//! Bounded single-workstation studies. Test split is never used for training or monitoring.
use crate::{
    batch::{ResidentScene, SampleSchedule, encode_batch, sample_indices},
    encoder::{FrozenViews, visible_mask},
    loss::{reconstruction_loss, rgb_patches},
    model::{DecoderConfig, GekkoDecoder},
    train::{
        CheckpointMetadata, EncoderSource, StepMetric, TrainConfig, TrainReport, clip,
        load_encoder, scalar,
    },
};
use anyhow::{Result, ensure};
use burn::{
    module::{AutodiffModule, Module},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::{
        Tensor,
        backend::{AutodiffBackend, Backend},
    },
};
use burn_gekko_data::{
    Split, fingerprint, load_rgb, open_dataset, sha256_file, write_config, write_json,
};
use burn_vjepa::SparseTokenMask;
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PilotConfig {
    pub training: TrainConfig,
    pub batch_size: usize,
    pub cache_full_views: bool,
    pub fixed_example: bool,
    pub train_rooms: usize,
    pub eval_every: usize,
    pub profile_phases: bool,
    pub max_seconds: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub schedule: Option<LearningRateSchedule>,
    #[serde(default)]
    pub shuffle_samples: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LearningRateSchedule {
    pub warmup_steps: usize,
    /// Fixed horizon, independent of a resume invocation's stop step.
    pub decay_steps: usize,
    pub min_lr_ratio: f64,
}
impl LearningRateSchedule {
    pub fn learning_rate(&self, base: f64, step: usize) -> f64 {
        if step < self.warmup_steps {
            return base * (step + 1) as f64 / self.warmup_steps as f64;
        }
        let progress = ((step - self.warmup_steps) as f64
            / (self.decay_steps - self.warmup_steps - 1).max(1) as f64)
            .min(1.0);
        base * (self.min_lr_ratio
            + (1.0 - self.min_lr_ratio) * 0.5 * (1.0 + (std::f64::consts::PI * progress).cos()))
    }
}
impl PilotConfig {
    pub fn validate(&self) -> Result<()> {
        self.training.validate()?;
        ensure!(
            (1..=16).contains(&self.batch_size),
            "pilot batch size must be 1..=16"
        );
        ensure!(
            (1..=4096).contains(&self.train_rooms),
            "pilot train room limit must be 1..=4096"
        );
        ensure!(
            (1..=7200).contains(&self.max_seconds),
            "pilot wall limit must be 1..=7200 seconds"
        );
        ensure!(self.eval_every <= 100_000, "invalid probe interval");
        if let Some(schedule) = &self.schedule {
            ensure!(
                schedule.decay_steps > schedule.warmup_steps + 1
                    && schedule.decay_steps <= 100_000
                    && self.training.steps <= schedule.decay_steps
                    && schedule.min_lr_ratio.is_finite()
                    && (0.0..=1.0).contains(&schedule.min_lr_ratio),
                "invalid learning-rate schedule"
            );
        }
        ensure!(
            !self.fixed_example || self.batch_size == 1,
            "fixed-example overfit uses batch size one"
        );
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProbeMetrics {
    pub total: f64,
    pub cross: f64,
    pub mae: f64,
    pub ri: f64,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProbePoint {
    pub step: usize,
    pub train: ProbeMetrics,
    pub validation: ProbeMetrics,
    pub examples_per_split: usize,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PilotStep {
    #[serde(flatten)]
    pub metric: StepMetric,
    pub samples: Vec<(u64, usize)>,
    pub batch_size: usize,
    pub encoder_ms: Option<f64>,
    pub forward_loss_ms: Option<f64>,
    pub backward_clip_ms: Option<f64>,
    pub optimizer_ms: Option<f64>,
    pub learning_rate: f64,
    pub elapsed_ms_precise: f64,
}
#[derive(Debug, Serialize)]
pub struct PilotReport {
    pub schema: u32,
    pub config: PilotConfig,
    pub stop_reason: String,
    pub completed_steps: usize,
    pub prepare_seconds: f64,
    pub optimize_seconds: f64,
    pub run_seconds: f64,
    pub warmup_steps_excluded: usize,
    pub warm_step_median_ms: f64,
    pub warm_step_p90_ms: f64,
    pub warm_examples_per_second: f64,
    pub resident_rgb_bytes: usize,
    pub cached_full_feature_bytes: usize,
    pub probes: Vec<ProbePoint>,
    pub steps: Vec<PilotStep>,
    pub fixed_train_loss_reduction: f64,
    pub validation_loss_reduction: f64,
    pub diagnostic_only: bool,
}

fn sync<B: Backend>(device: &B::Device) -> Result<()> {
    B::sync(device).map_err(|e| anyhow::anyhow!("backend sync: {e}"))
}
fn stats<B: Backend>(
    total: Tensor<B, 1>,
    cross: Tensor<B, 1>,
    mae: Tensor<B, 1>,
    ri: Tensor<B, 1>,
) -> Result<ProbeMetrics> {
    let values = Tensor::cat(vec![total, cross, mae, ri], 0)
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()?;
    ensure!(values.iter().all(|v| v.is_finite()), "nonfinite losses");
    Ok(ProbeMetrics {
        total: values[0] as f64,
        cross: values[1] as f64,
        mae: values[2] as f64,
        ri: values[3] as f64,
    })
}
fn probe<B: AutodiffBackend>(
    model: &GekkoDecoder<B>,
    views: &FrozenViews<B>,
    mask: &SparseTokenMask,
    grid: [usize; 2],
    config: &TrainConfig,
) -> Result<ProbeMetrics> {
    let output = model.valid().forward(
        views.masked.clone().inner(),
        views.full.clone().inner(),
        views.references.iter().map(|r| r.clone().inner()).collect(),
        mask,
        grid,
    )?;
    let loss = reconstruction_loss(
        output,
        rgb_patches(views.target_rgb.clone().inner(), 16, false),
        mask,
        config.normalize_targets,
        config.predict_patch_stats,
    );
    stats(loss.total, loss.cross, loss.mae, loss.ri)
}
fn save_checkpoint<B: AutodiffBackend, O: Optimizer<GekkoDecoder<B>, B>>(
    model: &GekkoDecoder<B>,
    optim: &O,
    run: &Path,
    step: usize,
    identity: &str,
    dataset: &str,
    encoder: &str,
) -> Result<PathBuf> {
    let path = run.join(format!("checkpoint-{step:06}"));
    ensure!(!path.exists(), "checkpoint already exists");
    let stage = run.join(format!("checkpoint-{step:06}.partial"));
    fs::create_dir(&stage)?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    model.clone().save_file(stage.join("decoder"), &recorder)?;
    recorder.record(optim.to_record(), stage.join("optimizer"))?;
    write_json(
        &stage.join("metadata.json"),
        &CheckpointMetadata {
            schema: 1,
            identity: identity.into(),
            dataset_id: dataset.into(),
            encoder_id: encoder.into(),
            completed_steps: step,
            model_sha256: sha256_file(&stage.join("decoder.mpk"))?,
            optimizer_sha256: sha256_file(&stage.join("optimizer.mpk"))?,
        },
    )?;
    fs::rename(stage, &path)?;
    Ok(path)
}
fn percentile(values: &[f64], p: f64) -> f64 {
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[((values.len() - 1) as f64 * p).round() as usize]
}

pub fn run_pilot<B: AutodiffBackend>(
    dataset: &Path,
    run: &Path,
    pilot: &PilotConfig,
    resume: Option<&Path>,
    device: &B::Device,
) -> Result<PilotReport> {
    pilot.validate()?;
    let wall = Instant::now();
    let config = &pilot.training;
    let manifest = open_dataset(dataset)?;
    ensure!(
        manifest.config.cameras > config.references
            && manifest.config.width <= 512
            && manifest.config.height <= 512,
        "pilot dataset exceeds bounds"
    );
    ensure!(!run.exists(), "choose a new run directory");
    fs::create_dir_all(run)?;
    write_config(&run.join("config.toml"), config)?;
    write_config(&run.join("pilot-config.toml"), pilot)?;
    let (encoder, enc_config, encoder_id) =
        load_encoder::<B::InnerBackend>(&config.encoder, config.seed, device)?;
    B::seed(device, config.seed.wrapping_add(1));
    let model = GekkoDecoder::<B>::with_reconstruction(
        &DecoderConfig {
            encoder_dim: config.image_features.width(enc_config.encoder.embed_dim),
            width: config.decoder_width,
            depth: config.decoder_depth,
            heads: config.decoder_heads,
            patch: 16,
        },
        config.decoder_position,
        config.predict_patch_stats,
        device,
    )?
    .with_mae_context_before_self(config.mae_context_before_self);
    let mut model = model.clone().load_record(model.into_record());
    let mut optim = AdamWConfig::new()
        .with_weight_decay(config.weight_decay)
        .init::<B, GekkoDecoder<B>>();
    let backend = B::name(device);
    let mut identity_config = pilot.clone();
    identity_config.training.steps = 0;
    identity_config.max_seconds = 0;
    let code = crate::provenance::identity()?;
    let identity = fingerprint(&(
        &manifest.dataset_id,
        &encoder_id,
        &backend,
        &identity_config,
        code,
    ))?;
    let start_step = if let Some(path) = resume {
        let meta: CheckpointMetadata =
            serde_json::from_slice(&fs::read(path.join("metadata.json"))?)?;
        ensure!(
            meta.schema == 1 && meta.identity == identity && meta.completed_steps < config.steps,
            "resume identity/budget mismatch"
        );
        ensure!(
            sha256_file(&path.join("decoder.mpk"))? == meta.model_sha256
                && sha256_file(&path.join("optimizer.mpk"))? == meta.optimizer_sha256,
            "checkpoint corruption"
        );
        let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
        model = model.load_file(path.join("decoder"), &recorder, device)?;
        optim = optim.load_record(recorder.load(path.join("optimizer"), device)?);
        meta.completed_steps
    } else {
        0
    };
    let initial = model.ri.weight.val().detach();
    let mut training = Vec::new();
    let mut validation = Vec::new();
    let mut rgb_bytes = 0;
    for entry in &manifest.scenes {
        if entry.split == Split::Test
            || (entry.split == Split::Train && training.len() >= pilot.train_rooms)
        {
            continue;
        }
        let rgb = load_rgb(&dataset.join("raw").join(&entry.file))?;
        rgb_bytes += rgb.views.iter().map(|v| v.len() * 4).sum::<usize>();
        let prepared =
            ResidentScene::new(&rgb, &encoder, &enc_config, pilot.cache_full_views, device);
        if entry.split == Split::Train {
            training.push(prepared);
        } else {
            validation.push(prepared);
        }
        if (training.len() + validation.len()).is_multiple_of(32) {
            sync::<B>(device)?;
            eprintln!(
                "prepared {} train and {} validation rooms in {:.1}s",
                training.len(),
                validation.len(),
                wall.elapsed().as_secs_f64()
            );
        }
        ensure!(
            wall.elapsed().as_secs() < pilot.max_seconds,
            "wall budget expired while preparing data"
        );
    }
    ensure!(
        training.len() == pilot.train_rooms && !validation.is_empty(),
        "not enough train/validation rooms"
    );
    let grid = [manifest.config.height / 16, manifest.config.width / 16];
    let n = grid[0] * grid[1];
    let views = manifest.config.cameras;
    let probe_mask = visible_mask(n, config.mask_ratio, config.seed, 0)?;
    let count = if pilot.fixed_example {
        1
    } else {
        pilot.batch_size.min(16)
    };
    let train_indices: Vec<_> = (0..count)
        .map(|i| {
            sample_indices(
                i * (training.len() * views / count).max(1),
                training.len(),
                views,
                pilot.fixed_example,
            )
        })
        .collect();
    let valid_indices: Vec<_> = (0..count)
        .map(|i| {
            sample_indices(
                i * (validation.len() * views / count).max(1),
                validation.len(),
                views,
                false,
            )
        })
        .collect();
    let train_panel = encode_batch::<B>(
        &encoder,
        &training,
        &train_indices,
        config.references,
        &probe_mask,
        config.image_features,
    )?;
    let valid_panel = encode_batch::<B>(
        &encoder,
        &validation,
        &valid_indices,
        config.references,
        &probe_mask,
        config.image_features,
    )?;
    sync::<B>(device)?;
    let mut probes = vec![ProbePoint {
        step: start_step,
        train: probe(&model, &train_panel, &probe_mask, grid, config)?,
        validation: probe(&model, &valid_panel, &probe_mask, grid, config)?,
        examples_per_split: count,
    }];
    save_checkpoint(
        &model,
        &optim,
        run,
        start_step,
        &identity,
        &manifest.dataset_id,
        &encoder_id,
    )?;
    let prepare_seconds = wall.elapsed().as_secs_f64();
    eprintln!(
        "pilot prepared in {prepare_seconds:.2}s; initial train {:.5}, validation {:.5}",
        probes[0].train.total, probes[0].validation.total
    );
    let mut log = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(run.join("metrics.jsonl"))?;
    write_json(&run.join("probes.json"), &probes)?;
    let mut steps = Vec::new();
    let mut stop_reason = "step_budget".to_string();
    let mut optimize_seconds = 0.0;
    let mut sampler =
        SampleSchedule::new(training.len(), views, config.seed, pilot.shuffle_samples);
    for step in start_step..config.steps {
        if wall.elapsed().as_secs() >= pilot.max_seconds {
            stop_reason = "wall_time_limit".into();
            break;
        }
        let start = Instant::now();
        let samples: Vec<_> = (0..pilot.batch_size)
            .map(|j| sampler.sample(step * pilot.batch_size + j, pilot.fixed_example))
            .collect();
        let mask = visible_mask(
            n,
            config.mask_ratio,
            config.seed,
            if pilot.fixed_example { 0 } else { step },
        )?;
        let batch = encode_batch::<B>(
            &encoder,
            &training,
            &samples,
            config.references,
            &mask,
            config.image_features,
        )?;
        if pilot.profile_phases {
            sync::<B>(device)?;
        }
        let encoder_ms = start.elapsed().as_secs_f64() * 1000.0;
        let forward_start = Instant::now();
        let output = model.forward(batch.masked, batch.full, batch.references, &mask, grid)?;
        let losses = reconstruction_loss(
            output,
            rgb_patches(batch.target_rgb, 16, false),
            &mask,
            config.normalize_targets,
            config.predict_patch_stats,
        );
        let values = stats(losses.total.clone(), losses.cross, losses.mae, losses.ri)?;
        ensure!(values.total < 1e6, "loss exploded");
        let forward_ms = forward_start.elapsed().as_secs_f64() * 1000.0;
        let backward_start = Instant::now();
        let mut grads = GradientsParams::from_grads(losses.total.backward(), &model);
        let gradient_norm = clip(&model, &mut grads, config.global_grad_clip)?;
        ensure!(gradient_norm < 1e6, "gradient norm exploded");
        let backward_ms = backward_start.elapsed().as_secs_f64() * 1000.0;
        let optimizer_start = Instant::now();
        let learning_rate = pilot.schedule.as_ref().map_or(config.learning_rate, |s| {
            s.learning_rate(config.learning_rate, step)
        });
        model = optim.step(learning_rate, model, grads);
        sync::<B>(device)?;
        let optimizer_ms = optimizer_start.elapsed().as_secs_f64() * 1000.0;
        optimize_seconds += start.elapsed().as_secs_f64();
        let sample_ids: Vec<_> = samples
            .iter()
            .map(|&(s, v)| (training[s].seed, v))
            .collect();
        let row = PilotStep {
            metric: StepMetric {
                step: step + 1,
                room_seed: sample_ids[0].0,
                target_view: sample_ids[0].1,
                total: values.total,
                cross: values.cross,
                mae: values.mae,
                ri: values.ri,
                gradient_norm,
                elapsed_ms: start.elapsed().as_millis(),
            },
            samples: sample_ids,
            batch_size: pilot.batch_size,
            encoder_ms: pilot.profile_phases.then_some(encoder_ms),
            forward_loss_ms: pilot.profile_phases.then_some(forward_ms),
            backward_clip_ms: pilot.profile_phases.then_some(backward_ms),
            optimizer_ms: pilot.profile_phases.then_some(optimizer_ms),
            learning_rate,
            elapsed_ms_precise: start.elapsed().as_secs_f64() * 1000.0,
        };
        writeln!(log, "{}", serde_json::to_string(&row)?)?;
        log.flush()?;
        eprintln!(
            "pilot step {}/{} loss {:.5} {:.1}ms",
            step + 1,
            config.steps,
            row.metric.total,
            row.metric.elapsed_ms
        );
        steps.push(row);
        if pilot.eval_every > 0 && (step + 1).is_multiple_of(pilot.eval_every) {
            probes.push(ProbePoint {
                step: step + 1,
                train: probe(&model, &train_panel, &probe_mask, grid, config)?,
                validation: probe(&model, &valid_panel, &probe_mask, grid, config)?,
                examples_per_split: count,
            });
            write_json(&run.join("probes.json"), &probes)?;
            save_checkpoint(
                &model,
                &optim,
                run,
                step + 1,
                &identity,
                &manifest.dataset_id,
                &encoder_id,
            )?;
        }
    }
    ensure!(
        !steps.is_empty(),
        "budget expired during preparation; no optimizer steps executed"
    );
    let completed = steps.last().unwrap().metric.step;
    if probes.last().unwrap().step != completed {
        probes.push(ProbePoint {
            step: completed,
            train: probe(&model, &train_panel, &probe_mask, grid, config)?,
            validation: probe(&model, &valid_panel, &probe_mask, grid, config)?,
            examples_per_split: count,
        });
    }
    let checkpoint = run.join(format!("checkpoint-{completed:06}"));
    if !checkpoint.exists() {
        save_checkpoint(
            &model,
            &optim,
            run,
            completed,
            &identity,
            &manifest.dataset_id,
            &encoder_id,
        )?;
    }
    let delta = scalar((model.ri.weight.val().detach() - initial).abs().max())?;
    ensure!(delta > 0.0, "no RI weight update");
    let train_report = TrainReport {
        artifact_kind: if config.encoder == EncoderSource::DiagnosticTiny {
            "random_encoder_pilot"
        } else {
            "pretrained_encoder_pilot"
        }
        .into(),
        dataset_id: manifest.dataset_id,
        encoder_id,
        backend,
        start_step,
        completed_steps: completed,
        checkpoint,
        train: steps.iter().map(|s| s.metric.clone()).collect(),
        validation_total: probes.last().unwrap().validation.total,
        decoder_weight_delta: delta,
    };
    write_json(&run.join("report.json"), &train_report)?;
    let warmup = 2.min(steps.len().saturating_sub(1));
    let warm: Vec<_> = steps[warmup..]
        .iter()
        .map(|s| s.elapsed_ms_precise)
        .collect();
    let warm_seconds = warm.iter().sum::<f64>() / 1000.0;
    let report = PilotReport {
        schema: 1,
        config: pilot.clone(),
        stop_reason,
        completed_steps: completed,
        prepare_seconds,
        optimize_seconds,
        run_seconds: wall.elapsed().as_secs_f64(),
        warmup_steps_excluded: warmup,
        warm_step_median_ms: percentile(&warm, 0.5),
        warm_step_p90_ms: percentile(&warm, 0.9),
        warm_examples_per_second: (warm.len() * pilot.batch_size) as f64 / warm_seconds,
        resident_rgb_bytes: rgb_bytes * 2,
        cached_full_feature_bytes: if pilot.cache_full_views {
            (training.len() + validation.len()) * views * n * enc_config.encoder.embed_dim * 4
        } else {
            0
        },
        fixed_train_loss_reduction: 1.0
            - probes.last().unwrap().train.total / probes[0].train.total,
        validation_loss_reduction: 1.0
            - probes.last().unwrap().validation.total / probes[0].validation.total,
        probes,
        steps,
        diagnostic_only: true,
    };
    write_json(&run.join("probes.json"), &report.probes)?;
    write_json(&run.join("pilot-report.json"), &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn learning_rate_schedule_has_fixed_absolute_endpoints() {
        let schedule = LearningRateSchedule {
            warmup_steps: 2,
            decay_steps: 6,
            min_lr_ratio: 0.1,
        };
        assert!((schedule.learning_rate(1.0, 0) - 0.5).abs() < 1e-12);
        assert_eq!(schedule.learning_rate(1.0, 1), 1.0);
        assert_eq!(schedule.learning_rate(1.0, 2), 1.0);
        assert!((schedule.learning_rate(1.0, 5) - 0.1).abs() < 1e-12);
        assert!((schedule.learning_rate(1.0, 7) - 0.1).abs() < 1e-12);
    }
}
