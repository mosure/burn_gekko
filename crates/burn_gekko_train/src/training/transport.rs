//! Bounded experiments for the explicit RGB transport branch.
use crate::{
    batch::{ResidentScene, SampleSchedule, encode_batch},
    encoder::visible_mask,
    model::{DecoderConfig, GekkoDecoder},
    train::{TrainConfig, clip, load_encoder, scalar},
    transport::{TransportDecoder, transport_loss},
};
use anyhow::{Result, ensure};
use burn::{
    module::{AutodiffModule, Module},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{
        Tensor,
        backend::{AutodiffBackend, Backend},
    },
};
use burn_gekko_data::{
    Split, load_rgb, open_dataset, read_config, sha256_file, write_config, write_json,
};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportConfig {
    pub dataset: PathBuf,
    pub backbone_config: PathBuf,
    pub initial_backbone: Option<PathBuf>,
    pub initial_transport: Option<PathBuf>,
    pub train_rooms: usize,
    pub batch_size: usize,
    pub steps: usize,
    pub max_seconds: u64,
    pub eval_every: usize,
    pub learning_rate: f64,
    pub max_displacement: f32,
}
impl TransportConfig {
    fn validate(&self) -> Result<()> {
        ensure!(
            (1..=4096).contains(&self.train_rooms) && (1..=16).contains(&self.batch_size),
            "invalid resident batch bounds"
        );
        ensure!(
            (1..=100000).contains(&self.steps) && (1..=7200).contains(&self.max_seconds),
            "invalid experiment bounds"
        );
        ensure!(
            self.eval_every > 0
                && self.learning_rate.is_finite()
                && self.learning_rate > 0.
                && self.learning_rate <= 0.01,
            "invalid optimizer config"
        );
        ensure!(
            self.max_displacement.is_finite()
                && self.max_displacement > 0.
                && self.max_displacement <= 128.,
            "invalid displacement bound"
        );
        let root = fs::canonicalize(".data")?;
        ensure!(
            fs::canonicalize(&self.dataset)?.starts_with(&root),
            "dataset must stay in .data"
        );
        for p in [&self.initial_backbone, &self.initial_transport]
            .into_iter()
            .flatten()
        {
            ensure!(
                fs::canonicalize(p)?.starts_with(&root),
                "checkpoint must stay in .data"
            );
        }
        ensure!(
            self.initial_backbone.is_none() || self.initial_transport.is_none(),
            "choose one initialization"
        );
        Ok(())
    }
}
fn build<B: Backend>(
    c: &TrainConfig,
    enc_dim: usize,
    config: &TransportConfig,
    d: &B::Device,
) -> Result<TransportDecoder<B>> {
    let mut backbone = GekkoDecoder::with_reconstruction(
        &DecoderConfig {
            encoder_dim: c.image_features.width(enc_dim),
            width: c.decoder_width,
            depth: c.decoder_depth,
            heads: c.decoder_heads,
            patch: 16,
        },
        c.decoder_position,
        c.predict_patch_stats,
        d,
    )?
    .with_mae_context_before_self(c.mae_context_before_self);
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    if let Some(path) = &config.initial_backbone {
        backbone = backbone.load_file(path, &recorder, d)?;
    }
    let mut model = TransportDecoder::new(backbone, c.decoder_width, config.max_displacement, d);
    model = model.clone().load_record(model.into_record());
    if let Some(path) = &config.initial_transport {
        model = model.load_file(path, &recorder, d)?;
    }
    Ok(model)
}
fn refs<B: AutodiffBackend>(
    scenes: &[ResidentScene<B::InnerBackend>],
    samples: &[(usize, usize)],
    n: usize,
) -> Vec<Tensor<B, 4>> {
    (1..=n)
        .map(|offset| {
            Tensor::from_inner(Tensor::cat(
                samples
                    .iter()
                    .map(|&(s, v)| scenes[s].rgb[(v + offset) % scenes[s].rgb.len()].clone())
                    .collect(),
                0,
            ))
        })
        .collect()
}
fn export_tensor<B: Backend, const D: usize>(path: &Path, t: Tensor<B, D>) -> Result<()> {
    let data = t.into_data().convert::<f32>().to_vec::<f32>()?;
    let mut f = fs::File::create(path)?;
    for value in data {
        f.write_all(&value.to_le_bytes())?;
    }
    Ok(())
}
#[allow(clippy::too_many_arguments)]
fn evaluate<B: AutodiffBackend>(
    model: &TransportDecoder<B>,
    encoder: &burn_vjepa::VJepaEncoder<B::InnerBackend>,
    scenes: &[ResidentScene<B::InnerBackend>],
    c: &TrainConfig,
    grid: [usize; 2],
    output: &Path,
    limit: usize,
    unrelated: bool,
) -> Result<f64> {
    fs::create_dir_all(output)?;
    let mask = visible_mask(grid[0] * grid[1], c.mask_ratio, c.seed, usize::MAX)?;
    let mut rows = Vec::new();
    let device = model.flow_head.weight.val().device();
    let hidden: Vec<f32> = (0..grid[0] * 16)
        .flat_map(|y| {
            (0..grid[1] * 16).map({
                let mask = &mask;
                move |x| {
                    if mask.indices().contains(&(y / 16 * grid[1] + x / 16)) {
                        0.
                    } else {
                        1.
                    }
                }
            })
        })
        .collect();
    let hidden = Tensor::<B::InnerBackend, 4>::from_data(
        burn::tensor::TensorData::new(hidden, [1, 1, grid[0] * 16, grid[1] * 16]),
        &device,
    );
    let valid = model.valid();
    for (s, scene) in scenes.iter().enumerate().take(limit) {
        for v in 0..scene.rgb.len() {
            let views = encode_batch::<B>(
                encoder,
                scenes,
                &[(s, v)],
                c.references,
                &mask,
                c.image_features,
            )?;
            let reference_samples = if unrelated {
                vec![((s + 1) % scenes.len(), v)]
            } else {
                vec![(s, v)]
            };
            let reference_rgb = refs::<B>(scenes, &reference_samples, c.references);
            let reference_features = if unrelated {
                encode_batch::<B>(
                    encoder,
                    scenes,
                    &reference_samples,
                    c.references,
                    &mask,
                    c.image_features,
                )?
                .references
            } else {
                views.references
            };
            let out = valid.forward(
                views.masked.inner(),
                reference_features.into_iter().map(|r| r.inner()).collect(),
                reference_rgb.iter().map(|r| r.clone().inner()).collect(),
                &mask,
                grid,
            )?;
            let target = views.target_rgb.inner();
            let mse = scalar(
                ((out.rgb.clone() - target.clone()).powf_scalar(2.) * hidden.clone()).sum()
                    / (hidden.clone().sum() * 3.),
            )?;
            rows.push(
                serde_json::json!({"room_seed":scene.seed,"target_view":v,"hidden_rgb_mse":mse}),
            );
            if v == 0 && s < 16 {
                let dir = output.join(format!("room-{}-view-{v}", scene.seed));
                fs::create_dir_all(&dir)?;
                export_tensor(&dir.join("target.f32"), target.permute([0, 2, 3, 1]))?;
                export_tensor(&dir.join("prediction.f32"), out.rgb.permute([0, 2, 3, 1]))?;
                for (i, r) in reference_rgb.into_iter().enumerate() {
                    export_tensor(
                        &dir.join(format!("reference-{i}.f32")),
                        r.inner().permute([0, 2, 3, 1]),
                    )?;
                }
                for (i, f) in out.flows.into_iter().enumerate() {
                    export_tensor(&dir.join(format!("flow-{i}.f32")), f.permute([0, 2, 3, 1]))?;
                }
                export_tensor(&dir.join("weights.f32"), out.weights.permute([0, 2, 3, 1]))?;
                write_json(
                    &dir.join("sample.json"),
                    &serde_json::json!({"height":grid[0]*16,"width":grid[1]*16,"room_seed":scene.seed,"target_view":v,"visible_patch_ids":mask.indices(),"hidden_rgb_mse":mse,"reference_count":c.references,"prediction":"HWC RGB, no target statistics or geometry","unrelated":unrelated}),
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
        &serde_json::json!({"mean_hidden_rgb_mse":mean,"targets":rows,"unrelated":unrelated}),
    )?;
    Ok(mean)
}

pub fn run<B: AutodiffBackend>(
    config: &TransportConfig,
    run: &Path,
    evaluation: Option<Split>,
    unrelated: bool,
    device: &B::Device,
) -> Result<()> {
    config.validate()?;
    let wall = Instant::now();
    ensure!(!run.exists(), "choose a fresh output directory");
    let ancestor = run.ancestors().find(|p| p.exists()).unwrap();
    ensure!(
        fs::canonicalize(ancestor)?.starts_with(fs::canonicalize(".data")?),
        "outputs must stay in .data"
    );
    fs::create_dir_all(run)?;
    write_config(&run.join("config.toml"), config)?;
    let c: TrainConfig = read_config(&config.backbone_config)?;
    c.validate()?;
    write_config(&run.join("backbone-config.toml"), &c)?;
    let manifest = open_dataset(&config.dataset)?;
    ensure!(
        manifest.config.cameras > c.references
            && manifest.config.width <= 512
            && manifest.config.height <= 512,
        "invalid dataset shape"
    );
    let grid = [manifest.config.height / 16, manifest.config.width / 16];
    let (encoder, enc_config, encoder_id) =
        load_encoder::<B::InnerBackend>(&c.encoder, c.seed, device)?;
    B::seed(device, c.seed.wrapping_add(4));
    let mut model = build::<B>(&c, enc_config.encoder.embed_dim, config, device)?;
    let mut training = Vec::new();
    let mut validation = Vec::new();
    for entry in &manifest.scenes {
        let wanted = if let Some(split) = evaluation {
            entry.split == split
        } else {
            entry.split == Split::Validation
                || (entry.split == Split::Train && training.len() < config.train_rooms)
        };
        if !wanted {
            continue;
        }
        let rgb = load_rgb(&config.dataset.join("raw").join(&entry.file))?;
        let scene = ResidentScene::new(&rgb, &encoder, &enc_config, true, device);
        if evaluation.is_none() && entry.split == Split::Train {
            training.push(scene);
        } else {
            validation.push(scene);
        }
        ensure!(
            wall.elapsed().as_secs() < config.max_seconds,
            "preparation exceeded wall bound"
        );
    }
    B::sync(device).map_err(|e| anyhow::anyhow!("{e}"))?;
    write_json(
        &run.join("provenance.json"),
        &serde_json::json!({"dataset_id":manifest.dataset_id,"encoder_id":encoder_id,"initial_backbone_sha256":config.initial_backbone.as_ref().map(|p|sha256_file(p)).transpose()?,"initial_transport_sha256":config.initial_transport.as_ref().map(|p|sha256_file(p)).transpose()?,"source_sha256":crate::provenance::identity()?,"evaluation_split":evaluation}),
    )?;
    if evaluation.is_some() {
        let mse = evaluate(
            &model,
            &encoder,
            &validation,
            &c,
            grid,
            &run.join("samples"),
            validation.len(),
            unrelated,
        )?;
        write_json(
            &run.join("report.json"),
            &serde_json::json!({"mean_hidden_rgb_mse":mse,"seconds":wall.elapsed().as_secs_f64()}),
        )?;
        return Ok(());
    }
    ensure!(
        training.len() == config.train_rooms && !validation.is_empty(),
        "insufficient scenes"
    );
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.01)
        .init::<B, TransportDecoder<B>>();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    let mut log = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(run.join("metrics.jsonl"))?;
    let mut sampler = SampleSchedule::new(training.len(), manifest.config.cameras, c.seed, true);
    let mut completed = 0;
    let mut probes = Vec::new();
    let mut times = Vec::new();
    let initial = evaluate(
        &model,
        &encoder,
        &validation,
        &c,
        grid,
        &run.join("step-000000"),
        4,
        false,
    )?;
    probes.push(serde_json::json!({"step":0,"mse":initial}));
    for step in 0..config.steps {
        if wall.elapsed().as_secs() >= config.max_seconds {
            break;
        }
        let start = Instant::now();
        let samples: Vec<_> = (0..config.batch_size)
            .map(|j| sampler.sample(step * config.batch_size + j, false))
            .collect();
        let mask = visible_mask(grid[0] * grid[1], c.mask_ratio, c.seed, step)?;
        let views = encode_batch::<B>(
            &encoder,
            &training,
            &samples,
            c.references,
            &mask,
            c.image_features,
        )?;
        let out = model.forward(
            views.masked,
            views.references,
            refs::<B>(&training, &samples, c.references),
            &mask,
            grid,
        )?;
        let loss = transport_loss(&out, views.target_rgb);
        let value = scalar(loss.clone())?;
        let mut grads = GradientsParams::from_grads(loss.backward(), &model);
        let norm = clip(&model, &mut grads, 1.)?;
        let warm = ((step + 1) as f64 / 100.).min(1.);
        let decay = 0.1
            + 0.9 * 0.5 * (1. + (std::f64::consts::PI * step as f64 / config.steps as f64).cos());
        let lr = config.learning_rate * warm * decay;
        model = optimizer.step(lr, model, grads);
        B::sync(device).map_err(|e| anyhow::anyhow!("{e}"))?;
        completed = step + 1;
        times.push(start.elapsed().as_secs_f64());
        writeln!(
            log,
            "{}",
            serde_json::json!({"step":completed,"loss":value,"gradient_norm":norm,"learning_rate":lr,"seconds":times.last(),"samples":samples.iter().map(|&(s,v)|(training[s].seed,v)).collect::<Vec<_>>()})
        )?;
        log.flush()?;
        if completed.is_multiple_of(50) {
            eprintln!(
                "transport step {completed}/{} loss {value:.6} {:.1}ms",
                config.steps,
                times.last().unwrap() * 1000.
            );
        }
        if completed.is_multiple_of(config.eval_every) {
            let mse = evaluate(
                &model,
                &encoder,
                &validation,
                &c,
                grid,
                &run.join(format!("step-{completed:06}")),
                4,
                false,
            )?;
            probes.push(serde_json::json!({"step":completed,"mse":mse}));
            write_json(&run.join("probes.json"), &probes)?;
            model
                .clone()
                .save_file(run.join(format!("checkpoint-{completed:06}")), &recorder)?;
            eprintln!("validation step {completed}: hidden RGB MSE {mse:.6}");
        }
    }
    ensure!(completed > 0, "no steps completed");
    model.save_file(run.join("final"), &recorder)?;
    write_json(
        &run.join("report.json"),
        &serde_json::json!({"completed_steps":completed,"seconds":wall.elapsed().as_secs_f64(),"step_seconds":times,"probes":probes,"model_sha256":sha256_file(&run.join("final.mpk"))?,"stop_reason":if completed==config.steps {"step_budget"}else{"wall_limit"}}),
    )?;
    Ok(())
}
