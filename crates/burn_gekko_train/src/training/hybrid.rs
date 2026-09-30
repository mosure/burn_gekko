//! Bounded calibration and selective decoder adaptation with frozen encoders.
use crate::{
    batch::{ResidentScene, SampleSchedule, encode_batch},
    encoder::{ImageFeatures, visible_mask},
    hybrid::{HybridFusion, hybrid_loss},
    released::{ReleasedDecoder, ReleasedEncoder},
    train::{EncoderSource, clip, load_encoder, scalar},
};
use anyhow::{Result, ensure};
use burn::{
    module::{AutodiffModule, Module},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{
        Tensor, TensorData,
        backend::{AutodiffBackend, Backend},
    },
};
use burn_gekko_data::{Split, load_rgb, open_dataset, sha256_file, write_config, write_json};
use burn_vjepa::{SparseTokenMask, VJepaEncoder};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HybridConfig {
    pub dataset: PathBuf,
    pub weights: PathBuf,
    pub encoder: EncoderSource,
    pub checkpoint: Option<PathBuf>,
    pub train_rooms: usize,
    pub batch_size: usize,
    pub steps: usize,
    pub max_seconds: u64,
    pub eval_every: usize,
    pub learning_rate: f64,
    pub seed: u64,
    pub mask_ratio: f32,
    pub references: usize,
    pub adapter_start: usize,
    #[serde(default)]
    pub normalize_predicted_content: bool,
    #[serde(default)]
    pub train_mae: bool,
    #[serde(default)]
    pub trainable_decoder_blocks: usize,
    #[serde(default)]
    pub edge_loss_weight: f64,
    #[serde(default)]
    pub gradient_energy_weight: f64,
    /// Export every evaluated target for independent full-split RGB/edge metrics.
    #[serde(default)]
    pub export_all_views: bool,
}
impl HybridConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            self.trainable_decoder_blocks <= 2
                && self.edge_loss_weight.is_finite()
                && (0.0..=2.0).contains(&self.edge_loss_weight)
                && self.gradient_energy_weight.is_finite()
                && (0.0..=1.0).contains(&self.gradient_energy_weight),
            "invalid fine-tune scope"
        );
        ensure!(
            (1..=4096).contains(&self.train_rooms) && (1..=16).contains(&self.batch_size),
            "invalid batch bounds"
        );
        ensure!(
            (1..=100000).contains(&self.steps)
                && (1..=7200).contains(&self.max_seconds)
                && self.eval_every > 0,
            "invalid budget bounds"
        );
        ensure!(
            self.learning_rate > 0.
                && self.learning_rate <= 0.01
                && self.mask_ratio > 0.
                && self.mask_ratio < 1.
                && (1..=3).contains(&self.references),
            "invalid training parameters"
        );
        let root = fs::canonicalize(".data")?;
        for p in [&self.dataset, &self.weights]
            .into_iter()
            .chain(self.checkpoint.iter())
        {
            ensure!(
                fs::canonicalize(p)?.starts_with(&root),
                "inputs must stay inside .data"
            );
        }
        Ok(())
    }
}

struct Scenes<B: Backend> {
    rgb: Vec<ResidentScene<B>>,
    appearance: Vec<Vec<Tensor<B, 3>>>,
}
fn appearance_batch<B: AutodiffBackend>(
    encoder: &ReleasedEncoder<B::InnerBackend>,
    scenes: &Scenes<B::InnerBackend>,
    samples: &[(usize, usize)],
    mask: &SparseTokenMask,
    references: usize,
) -> (Tensor<B, 3>, Vec<Tensor<B, 3>>) {
    let rgb = Tensor::cat(
        samples
            .iter()
            .map(|&(s, v)| scenes.rgb[s].rgb[v].clone())
            .collect(),
        0,
    );
    let target = Tensor::from_inner(encoder.forward(rgb, Some(mask)));
    let refs = (1..=references)
        .map(|offset| {
            Tensor::from_inner(Tensor::cat(
                samples
                    .iter()
                    .map(|&(s, v)| {
                        scenes.appearance[s][(v + offset) % scenes.appearance[s].len()].clone()
                    })
                    .collect(),
                0,
            ))
        })
        .collect();
    (target, refs)
}
fn export<B: Backend, const D: usize>(p: &Path, t: Tensor<B, D>) -> Result<()> {
    let values = t.into_data().convert::<f32>().to_vec::<f32>()?;
    // Full-split exports must not issue a syscall per float. The on-disk
    // little-endian F32 ordering is identical to the diagnostic exporter.
    let bytes: Vec<_> = values.into_iter().flat_map(f32::to_le_bytes).collect();
    fs::write(p, bytes)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn evaluate<B: AutodiffBackend>(
    model: &HybridFusion<B>,
    encoder: &VJepaEncoder<B::InnerBackend>,
    appearance_encoder: &ReleasedEncoder<B::InnerBackend>,
    scenes: &Scenes<B::InnerBackend>,
    c: &HybridConfig,
    grid: [usize; 2],
    output: &Path,
    limit: usize,
    unrelated: bool,
    adapter: bool,
) -> Result<f64> {
    fs::create_dir_all(output)?;
    let mask = visible_mask(grid[0] * grid[1], c.mask_ratio, c.seed, usize::MAX)?;
    let [h, w] = [grid[0] * 16, grid[1] * 16];
    let hidden: Vec<f32> = (0..h * w)
        .map(|i| {
            if mask
                .indices()
                .contains(&(i / w / 16 * grid[1] + i % w / 16))
            {
                0.
            } else {
                1.
            }
        })
        .collect();
    let hidden = Tensor::<B::InnerBackend, 4>::from_data(
        TensorData::new(hidden, [1, 1, h, w]),
        &model.statistics_out.weight.val().device(),
    );
    let valid = model.valid();
    let mut rows = Vec::new();
    for (s, scene) in scenes.rgb.iter().enumerate().take(limit) {
        for v in 0..scene.rgb.len() {
            let views = encode_batch::<B>(
                encoder,
                &scenes.rgb,
                &[(s, v)],
                c.references,
                &mask,
                ImageFeatures::SemanticRgb,
            )?;
            let (target, mut references) =
                appearance_batch::<B>(appearance_encoder, scenes, &[(s, v)], &mask, c.references);
            let mut jepa_references = views.references;
            let ref_room = if unrelated {
                (s + 1) % scenes.rgb.len()
            } else {
                s
            };
            if unrelated {
                references = appearance_batch::<B>(
                    appearance_encoder,
                    scenes,
                    &[(ref_room, v)],
                    &mask,
                    c.references,
                )
                .1;
                jepa_references = encode_batch::<B>(
                    encoder,
                    &scenes.rgb,
                    &[(ref_room, v)],
                    c.references,
                    &mask,
                    ImageFeatures::SemanticRgb,
                )?
                .references;
            }
            let mono = c.train_mae.then(|| {
                valid
                    .monocular(
                        target.clone().inner(),
                        views.masked.clone().inner(),
                        &mask,
                        grid,
                        adapter,
                    )
                    .1
            });
            let out = valid.forward(
                target.inner(),
                views.masked.inner(),
                references.into_iter().map(|x| x.inner()).collect(),
                jepa_references.into_iter().map(|x| x.inner()).collect(),
                &mask,
                grid,
                adapter,
            );
            let target = views.target_rgb.inner();
            let mse = scalar(
                ((out.rgb.clone() - target.clone()).powf_scalar(2.) * hidden.clone()).sum()
                    / (hidden.clone().sum() * 3.),
            )?;
            let mono_mse = mono
                .as_ref()
                .map(|m| {
                    scalar(
                        ((m.clone() - target.clone()).powf_scalar(2.) * hidden.clone()).sum()
                            / (hidden.clone().sum() * 3.),
                    )
                })
                .transpose()?;
            rows.push(
                serde_json::json!({"room_seed":scene.seed,"target_view":v,"hidden_rgb_mse":mse,"monocular_hidden_rgb_mse":mono_mse}),
            );
            if c.export_all_views || (v == 0 && s < 16) {
                let dir = output.join(format!("room-{}-view-{v}", scene.seed));
                fs::create_dir_all(&dir)?;
                export(&dir.join("target.f32"), target.permute([0, 2, 3, 1]))?;
                export(&dir.join("prediction.f32"), out.rgb.permute([0, 2, 3, 1]))?;
                if let Some(mono) = mono {
                    export(&dir.join("monocular.f32"), mono.permute([0, 2, 3, 1]))?;
                }
                for i in 0..c.references {
                    export(
                        &dir.join(format!("reference-{i}.f32")),
                        scenes.rgb[ref_room].rgb[(v + i + 1) % scene.rgb.len()]
                            .clone()
                            .permute([0, 2, 3, 1]),
                    )?;
                }
                write_json(
                    &dir.join("sample.json"),
                    &serde_json::json!({"height":h,"width":w,"room_seed":scene.seed,"target_view":v,"visible_patch_ids":mask.indices(),"hidden_rgb_mse":mse,"monocular_hidden_rgb_mse":mono_mse,"reference_count":c.references,"unrelated":unrelated,"jepa_adapter":adapter,"method":"hybrid Gekko appearance + V-JEPA adapter + predicted patch statistics","trainable_decoder_blocks":c.trainable_decoder_blocks,"oracle_statistics":false}),
                )?;
            }
        }
    }
    let mean = rows
        .iter()
        .map(|r| r["hidden_rgb_mse"].as_f64().unwrap())
        .sum::<f64>()
        / rows.len() as f64;
    write_json(
        &output.join("evaluation.json"),
        &serde_json::json!({"mean_hidden_rgb_mse":mean,"targets":rows,"unrelated":unrelated,"jepa_adapter":adapter}),
    )?;
    Ok(mean)
}

pub fn run<B: AutodiffBackend>(
    c: &HybridConfig,
    run: &Path,
    split: Option<Split>,
    unrelated: bool,
    disable_adapter: bool,
    device: &B::Device,
) -> Result<()> {
    c.validate()?;
    let wall = Instant::now();
    ensure!(!run.exists(), "choose a fresh run path");
    let ancestor = run.ancestors().find(|p| p.exists()).unwrap();
    ensure!(
        fs::canonicalize(ancestor)?.starts_with(fs::canonicalize(".data")?),
        "outputs must stay in .data"
    );
    fs::create_dir_all(run)?;
    write_config(&run.join("config.toml"), c)?;
    let manifest = open_dataset(&c.dataset)?;
    ensure!(
        manifest.config.cameras > c.references
            && manifest.config.width <= 512
            && manifest.config.height <= 512,
        "invalid dataset bounds"
    );
    let grid = [manifest.config.height / 16, manifest.config.width / 16];
    let (encoder, enc_config, encoder_id) =
        load_encoder::<B::InnerBackend>(&c.encoder, c.seed, device)?;
    let (appearance_encoder, decoder) =
        crate::released::load::<B::InnerBackend>(&c.weights, device)?;
    B::seed(device, c.seed.wrapping_add(5));
    let mut model = HybridFusion::<B>::new(
        ReleasedDecoder::<B>::from_inner(decoder),
        ImageFeatures::SemanticRgb.width(enc_config.encoder.embed_dim),
        c.normalize_predicted_content,
        device,
    );
    model = model.clone().load_record(model.into_record());
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    if let Some(path) = &c.checkpoint {
        model = model.load_file(path, &recorder, device)?;
    }
    model.decoder = model.decoder.no_grad();
    model.decoder = model.decoder.train_tail(c.trainable_decoder_blocks);
    ensure!(
        !model.decoder.cross_head.weight.val().is_require_grad(),
        "released decoder output head must be frozen"
    );
    let frozen_before = model.decoder.cross_head.weight.val().detach();
    let [first_before, last_before] = model.decoder.optimization_markers();
    let mut training = Scenes {
        rgb: Vec::new(),
        appearance: Vec::new(),
    };
    let mut validation = Scenes {
        rgb: Vec::new(),
        appearance: Vec::new(),
    };
    for entry in &manifest.scenes {
        let wanted = if let Some(split) = split {
            entry.split == split
        } else {
            entry.split == Split::Validation
                || (entry.split == Split::Train && training.rgb.len() < c.train_rooms)
        };
        if !wanted {
            continue;
        }
        let rgb = load_rgb(&c.dataset.join("raw").join(&entry.file))?;
        let resident = ResidentScene::new(&rgb, &encoder, &enc_config, true, device);
        let appearance = resident
            .rgb
            .iter()
            .map(|v| appearance_encoder.forward(v.clone(), None))
            .collect();
        let dest = if split.is_none() && entry.split == Split::Train {
            &mut training
        } else {
            &mut validation
        };
        dest.rgb.push(resident);
        dest.appearance.push(appearance);
        if (training.rgb.len() + validation.rgb.len()).is_multiple_of(32) {
            B::sync(device).map_err(|e| anyhow::anyhow!("{e}"))?;
            eprintln!(
                "prepared {} train / {} evaluation rooms in {:.1}s",
                training.rgb.len(),
                validation.rgb.len(),
                wall.elapsed().as_secs_f64()
            );
        }
        ensure!(
            wall.elapsed().as_secs() < c.max_seconds,
            "preparation exceeded wall limit"
        );
    }
    write_json(
        &run.join("provenance.json"),
        &serde_json::json!({"dataset_id":manifest.dataset_id,"encoder_id":encoder_id,"appearance_weights_sha256":sha256_file(&c.weights)?,"initial_checkpoint_sha256":c.checkpoint.as_ref().map(|p|sha256_file(p)).transpose()?,"source_sha256":crate::provenance::identity()?,"evaluation_split":split,"frozen_released_decoder":c.trainable_decoder_blocks==0,"trainable_decoder_blocks":c.trainable_decoder_blocks}),
    )?;
    if split.is_some() {
        let mse = evaluate(
            &model,
            &encoder,
            &appearance_encoder,
            &validation,
            c,
            grid,
            &run.join("samples"),
            validation.rgb.len(),
            unrelated,
            !disable_adapter,
        )?;
        write_json(
            &run.join("report.json"),
            &serde_json::json!({"mean_hidden_rgb_mse":mse,"seconds":wall.elapsed().as_secs_f64()}),
        )?;
        return Ok(());
    }
    ensure!(
        training.rgb.len() == c.train_rooms && !validation.rgb.is_empty(),
        "insufficient scenes"
    );
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.01)
        .init::<B, HybridFusion<B>>();
    let mut sampler =
        SampleSchedule::new(training.rgb.len(), manifest.config.cameras, c.seed, true);
    let mut log = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(run.join("metrics.jsonl"))?;
    let initial = evaluate(
        &model,
        &encoder,
        &appearance_encoder,
        &validation,
        c,
        grid,
        &run.join("step-000000"),
        4,
        false,
        c.adapter_start == 0,
    )?;
    let mut probes = vec![serde_json::json!({"step":0,"mse":initial})];
    let mut completed = 0;
    let mut times = Vec::new();
    for step in 0..c.steps {
        if wall.elapsed().as_secs() >= c.max_seconds {
            break;
        }
        let start = Instant::now();
        let samples: Vec<_> = (0..c.batch_size)
            .map(|j| sampler.sample(step * c.batch_size + j, false))
            .collect();
        let mask = visible_mask(grid[0] * grid[1], c.mask_ratio, c.seed, step)?;
        let views = encode_batch::<B>(
            &encoder,
            &training.rgb,
            &samples,
            c.references,
            &mask,
            ImageFeatures::SemanticRgb,
        )?;
        let (target, refs) = appearance_batch::<B>(
            &appearance_encoder,
            &training,
            &samples,
            &mask,
            c.references,
        );
        let adapter = step >= c.adapter_start;
        let mut out = model.forward(
            target.clone(),
            views.masked.clone(),
            refs,
            views.references,
            &mask,
            grid,
            adapter,
        );
        if c.train_mae {
            let (prediction, rgb) = model.monocular(target, views.masked, &mask, grid, adapter);
            out.pair_predictions.push(prediction);
            out.auxiliary_rgb.push(rgb);
        }
        let loss = hybrid_loss(
            &out,
            views.target_rgb,
            &mask,
            c.edge_loss_weight,
            c.gradient_energy_weight,
        );
        let value = scalar(loss.clone())?;
        let mut gradients = GradientsParams::from_grads(loss.backward(), &model);
        let norm = clip(&model, &mut gradients, 1.)?;
        let lr = c.learning_rate
            * ((step + 1) as f64 / 100.).min(1.)
            * (0.1 + 0.45 * (1. + (std::f64::consts::PI * step as f64 / c.steps as f64).cos()));
        model = optimizer.step(lr, model, gradients);
        B::sync(device).map_err(|e| anyhow::anyhow!("{e}"))?;
        completed = step + 1;
        times.push(start.elapsed().as_secs_f64());
        writeln!(
            log,
            "{}",
            serde_json::json!({"step":completed,"loss":value,"gradient_norm":norm,"learning_rate":lr,"adapter_enabled":adapter,"seconds":times.last(),"samples":samples.iter().map(|&(s,v)|(training.rgb[s].seed,v)).collect::<Vec<_>>()})
        )?;
        log.flush()?;
        if completed.is_multiple_of(50) {
            eprintln!(
                "hybrid step {completed}/{} loss {value:.6} {:.1}ms",
                c.steps,
                times.last().unwrap() * 1000.
            );
        }
        if completed.is_multiple_of(c.eval_every) {
            let mse = evaluate(
                &model,
                &encoder,
                &appearance_encoder,
                &validation,
                c,
                grid,
                &run.join(format!("step-{completed:06}")),
                4,
                false,
                adapter,
            )?;
            probes.push(serde_json::json!({"step":completed,"mse":mse}));
            write_json(&run.join("probes.json"), &probes)?;
            model
                .clone()
                .save_file(run.join(format!("checkpoint-{completed:06}")), &recorder)?;
            eprintln!("validation step {completed}: {mse:.6}");
        }
    }
    ensure!(completed > 0, "no steps completed");
    let frozen_delta = scalar(
        (model.decoder.cross_head.weight.val().detach() - frozen_before)
            .abs()
            .max(),
    )?;
    ensure!(frozen_delta == 0., "frozen decoder head changed");
    let adapter_weight = scalar(model.adapter_down.weight.val().detach().abs().max())?;
    let [first, last] = model.decoder.optimization_markers();
    let first_delta = scalar((first - first_before).abs().max())?;
    let last_delta = scalar((last - last_before).abs().max())?;
    ensure!(first_delta == 0., "frozen first decoder block changed");
    ensure!(
        c.trainable_decoder_blocks == 0 || last_delta > 0.,
        "decoder tail did not update"
    );
    model.save_file(run.join("final"), &recorder)?;
    write_json(
        &run.join("report.json"),
        &serde_json::json!({"completed_steps":completed,"seconds":wall.elapsed().as_secs_f64(),"step_seconds":times,"probes":probes,"frozen_head_max_abs_delta":frozen_delta,"first_decoder_block_max_abs_delta":first_delta,"last_decoder_block_max_abs_delta":last_delta,"adapter_weight_max_abs":adapter_weight,"model_sha256":sha256_file(&run.join("final.mpk"))?,"stop_reason":if completed==c.steps{"step_budget"}else{"wall_limit"}}),
    )?;
    Ok(())
}
