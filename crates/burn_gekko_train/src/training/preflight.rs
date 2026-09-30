use crate::{
    encoder::{ImageFeatures, encode_views, load_checkpoint, visible_mask},
    loss::{reconstruction_loss, rgb_patches},
    model::{DecoderConfig, DecoderPosition, GekkoDecoder},
};
use anyhow::{Result, ensure};
use burn::{
    module::{AutodiffModule, Module, ModuleVisitor, Param},
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
use burn_vjepa::{VJepaConfig, VJepaEncoder, load_burnpack_parts};
use serde::{Deserialize, Serialize};
use std::{
    fs::{self, OpenOptions},
    io::Write,
    path::{Component, Path, PathBuf},
    time::Instant,
};

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EncoderSource {
    DiagnosticTiny,
    Checkpoint { directory: PathBuf },
    Burnpack { directory: PathBuf },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TrainConfig {
    pub seed: u64,
    pub steps: usize,
    pub learning_rate: f64,
    pub weight_decay: f32,
    pub mask_ratio: f32,
    pub references: usize,
    pub decoder_width: usize,
    pub decoder_depth: usize,
    pub decoder_heads: usize,
    pub normalize_targets: bool,
    pub global_grad_clip: f64,
    pub encoder: EncoderSource,
    #[serde(default)]
    pub image_features: ImageFeatures,
    #[serde(default)]
    pub decoder_position: DecoderPosition,
    #[serde(default)]
    pub predict_patch_stats: bool,
    #[serde(default)]
    pub mae_context_before_self: bool,
}
impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            seed: 17,
            steps: 2,
            learning_rate: 1e-4,
            weight_decay: 0.05,
            mask_ratio: 0.75,
            references: 1,
            decoder_width: 32,
            decoder_depth: 1,
            decoder_heads: 4,
            normalize_targets: true,
            global_grad_clip: 1.0,
            encoder: EncoderSource::DiagnosticTiny,
            image_features: ImageFeatures::Semantic,
            decoder_position: DecoderPosition::Absolute,
            predict_patch_stats: false,
            mae_context_before_self: false,
        }
    }
}
impl TrainConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.predict_patch_stats || self.normalize_targets,
            "predicted patch statistics require normalized content targets"
        );
        ensure!(
            (1..=100_000).contains(&self.steps),
            "bounded training is limited to 1..=100000 optimizer steps"
        );
        ensure!(
            self.learning_rate.is_finite()
                && self.learning_rate > 0.0
                && self.learning_rate <= 1e-2,
            "invalid learning rate"
        );
        ensure!(
            self.weight_decay.is_finite() && (0.0..=1.0).contains(&self.weight_decay),
            "invalid weight decay"
        );
        ensure!(
            self.mask_ratio.is_finite() && self.mask_ratio > 0.0 && self.mask_ratio < 1.0,
            "invalid mask ratio"
        );
        ensure!(
            (1..=3).contains(&self.references),
            "reference count must be 1..=3"
        );
        ensure!(
            self.global_grad_clip.is_finite() && self.global_grad_clip > 0.0,
            "invalid gradient clipping threshold"
        );
        DecoderConfig {
            encoder_dim: 32,
            width: self.decoder_width,
            depth: self.decoder_depth,
            heads: self.decoder_heads,
            patch: 16,
        }
        .validate()?;
        ensure!(
            self.decoder_position == DecoderPosition::Absolute
                || (self.decoder_width / self.decoder_heads).is_multiple_of(4),
            "2D rotary attention requires head width divisible by four"
        );
        Ok(())
    }
}

#[derive(Deserialize)]
struct Package {
    version: u32,
    jepa_config: VJepaConfig,
    parts_manifest: String,
}
#[derive(Deserialize)]
struct Parts {
    version: u32,
    parts: Vec<Part>,
}
#[derive(Deserialize)]
struct Part {
    path: String,
    bytes: u64,
    sha256: String,
}
fn local_file(directory: &Path, name: &str) -> Result<PathBuf> {
    let mut it = Path::new(name).components();
    ensure!(
        matches!(it.next(), Some(Component::Normal(_))) && it.next().is_none(),
        "invalid checkpoint member path"
    );
    Ok(directory.join(name))
}
pub fn load_encoder<B: Backend>(
    source: &EncoderSource,
    seed: u64,
    device: &B::Device,
) -> Result<(VJepaEncoder<B>, VJepaConfig, String)> {
    B::seed(device, seed);
    let (encoder, config, identity) = match source {
        EncoderSource::DiagnosticTiny => {
            let config = VJepaConfig::tiny_for_tests();
            (
                VJepaEncoder::new(&config, device),
                config,
                fingerprint(&("random-diagnostic-v1", seed))?,
            )
        }
        EncoderSource::Checkpoint { directory } => {
            let identity = fingerprint(&(
                sha256_file(&directory.join("config.json"))?,
                sha256_file(&directory.join("model.safetensors"))?,
            ))?;
            let (encoder, config) = load_checkpoint(directory, device)?;
            (encoder, config, identity)
        }
        EncoderSource::Burnpack { directory } => {
            let package_bytes = fs::read(directory.join("manifest.json"))?;
            let package: Package = serde_json::from_slice(&package_bytes)?;
            ensure!(
                package.version == 1 && package.jepa_config.model_type == "vjepa2_1",
                "unsupported pretrained package"
            );
            let parts_path = local_file(directory, &package.parts_manifest)?;
            let parts_bytes = fs::read(parts_path)?;
            let parts: Parts = serde_json::from_slice(&parts_bytes)?;
            ensure!(
                parts.version == 1 && !parts.parts.is_empty() && parts.parts.len() <= 128,
                "invalid shard manifest"
            );
            let mut buffers = Vec::new();
            for part in &parts.parts {
                let path = local_file(directory, &part.path)?;
                ensure!(
                    fs::metadata(&path)?.len() == part.bytes && sha256_file(&path)? == part.sha256,
                    "checkpoint shard checksum mismatch"
                );
                buffers.push(fs::read(path)?);
            }
            let identity = fingerprint(&(package_bytes, parts_bytes))?;
            let model = load_burnpack_parts::<B>(&package.jepa_config, buffers, device)?;
            (model.encoder, package.jepa_config, identity)
        }
    };
    ensure!(
        config.patch_size == 16 && config.encoder.embed_dim <= 1024 && config.encoder.depth <= 24,
        "encoder exceeds preflight budget"
    );
    // Force lazy parameters now, before decoder seeding or checkpoint loading can alter RNG order.
    let encoder = encoder.clone().load_record(encoder.into_record());
    Ok((encoder, config, identity))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StepMetric {
    pub step: usize,
    pub room_seed: u64,
    pub target_view: usize,
    pub total: f64,
    pub cross: f64,
    pub mae: f64,
    pub ri: f64,
    pub gradient_norm: f64,
    pub elapsed_ms: u128,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CheckpointMetadata {
    pub schema: u32,
    pub identity: String,
    pub dataset_id: String,
    pub encoder_id: String,
    pub completed_steps: usize,
    pub model_sha256: String,
    pub optimizer_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TrainReport {
    pub artifact_kind: String,
    pub dataset_id: String,
    pub encoder_id: String,
    pub backend: String,
    pub start_step: usize,
    pub completed_steps: usize,
    pub checkpoint: PathBuf,
    pub train: Vec<StepMetric>,
    pub validation_total: f64,
    pub decoder_weight_delta: f64,
}
pub use burn_gekko::tensor::scalar;

struct GradientNorm<'a, B: Backend> {
    grads: &'a GradientsParams,
    sums: Vec<Tensor<B, 1>>,
}
impl<B: AutodiffBackend> ModuleVisitor<B> for GradientNorm<'_, B::InnerBackend> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(g) = self.grads.get::<B::InnerBackend, D>(param.id) {
            self.sums.push(g.powf_scalar(2.0).sum());
        }
    }
}
struct Clip<'a> {
    grads: &'a mut GradientsParams,
    scale: f64,
}
impl<B: AutodiffBackend> ModuleVisitor<B> for Clip<'_> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        if let Some(g) = self.grads.remove::<B::InnerBackend, D>(param.id) {
            self.grads.register(param.id, g * self.scale);
        }
    }
}
pub(crate) fn clip<B: AutodiffBackend>(
    model: &impl Module<B>,
    grads: &mut GradientsParams,
    max: f64,
) -> Result<f64> {
    ensure!(!grads.is_empty(), "no trainable gradients");
    let mut norm = GradientNorm {
        grads,
        sums: Vec::new(),
    };
    model.visit(&mut norm);
    ensure!(!norm.sums.is_empty(), "no model gradients");
    // One device reduction/readback for the whole model, instead of one per parameter.
    let squared = scalar(Tensor::cat(norm.sums, 0).sum())?;
    ensure!(squared > 0.0, "invalid/zero model gradient norm");
    let norm = squared.sqrt();
    if norm > max {
        model.visit(&mut Clip {
            grads,
            scale: max / norm,
        });
    }
    Ok(norm)
}

pub fn train<B: AutodiffBackend>(
    dataset: &Path,
    run: &Path,
    config: &TrainConfig,
    resume: Option<&Path>,
    device: &B::Device,
) -> Result<TrainReport> {
    config.validate()?;
    ensure!(
        config.steps <= 16 && config.decoder_width <= 64 && config.decoder_depth <= 2,
        "train-preflight allows at most 16 steps and a 64-wide two-layer decoder; use train-pilot for a bounded study"
    );
    let manifest = open_dataset(dataset)?;
    ensure!(
        manifest.config.cameras > config.references
            && manifest.config.width <= 128
            && manifest.config.height <= 128,
        "training preflight input budget exceeded"
    );
    ensure!(
        !run.exists(),
        "run directory already exists; choose a fresh run name"
    );
    let (encoder, encoder_config, encoder_id) =
        load_encoder::<B::InnerBackend>(&config.encoder, config.seed, device)?;
    let decoder_config = DecoderConfig {
        encoder_dim: config
            .image_features
            .width(encoder_config.encoder.embed_dim),
        width: config.decoder_width,
        depth: config.decoder_depth,
        heads: config.decoder_heads,
        patch: encoder_config.patch_size,
    };
    B::seed(device, config.seed.wrapping_add(1));
    let model = GekkoDecoder::<B>::with_reconstruction(
        &decoder_config,
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
    let mut identity_config = config.clone();
    identity_config.steps = 0;
    let code = crate::provenance::identity()?;
    let identity = fingerprint(&(
        &manifest.dataset_id,
        &encoder_id,
        &identity_config,
        &backend,
        code,
    ))?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    let start_step = if let Some(path) = resume {
        let meta: CheckpointMetadata =
            serde_json::from_slice(&fs::read(path.join("metadata.json"))?)?;
        ensure!(
            meta.schema == 1 && meta.identity == identity && meta.completed_steps < config.steps,
            "checkpoint config/source/backend mismatch or no remaining steps"
        );
        ensure!(
            sha256_file(&path.join("decoder.mpk"))? == meta.model_sha256
                && sha256_file(&path.join("optimizer.mpk"))? == meta.optimizer_sha256,
            "checkpoint checksum mismatch"
        );
        model = model.load_file(path.join("decoder"), &recorder, device)?;
        optim = optim.load_record(recorder.load(path.join("optimizer"), device)?);
        meta.completed_steps
    } else {
        0
    };
    fs::create_dir_all(run)?;
    write_config(&run.join("config.toml"), config)?;
    let initial = model.ri.weight.val().detach();
    let training: Vec<_> = manifest
        .scenes
        .iter()
        .filter(|s| s.split == Split::Train)
        .collect();
    let mut metrics = Vec::new();
    let mut log = OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(run.join("metrics.jsonl"))?;
    for step in start_step..config.steps {
        let start = Instant::now();
        let scene = load_rgb(
            &dataset
                .join("raw")
                .join(&training[step % training.len()].file),
        )?;
        let grid = [
            scene.height / encoder_config.patch_size,
            scene.width / encoder_config.patch_size,
        ];
        let mask = visible_mask(grid[0] * grid[1], config.mask_ratio, config.seed, step)?;
        let target_view = step % scene.views.len();
        let views = encode_views::<B>(
            &encoder,
            &encoder_config,
            &scene,
            target_view,
            config.references,
            &mask,
            config.image_features,
            device,
        )?;
        let target = rgb_patches(views.target_rgb, encoder_config.patch_size, false);
        let predictions = model.forward(views.masked, views.full, views.references, &mask, grid)?;
        let losses = reconstruction_loss(
            predictions,
            target,
            &mask,
            config.normalize_targets,
            config.predict_patch_stats,
        );
        let (total, cross, mae, ri) = (
            scalar(losses.total.clone())?,
            scalar(losses.cross)?,
            scalar(losses.mae)?,
            scalar(losses.ri)?,
        );
        let mut grads = GradientsParams::from_grads(losses.total.backward(), &model);
        let gradient_norm = clip(&model, &mut grads, config.global_grad_clip)?;
        model = optim.step(config.learning_rate, model, grads);
        let metric = StepMetric {
            step: step + 1,
            room_seed: scene.seed,
            target_view,
            total,
            cross,
            mae,
            ri,
            gradient_norm,
            elapsed_ms: start.elapsed().as_millis(),
        };
        writeln!(log, "{}", serde_json::to_string(&metric)?)?;
        log.flush()?;
        eprintln!(
            "step {}/{} loss {:.6} gradient_norm {:.4}",
            step + 1,
            config.steps,
            total,
            gradient_norm
        );
        metrics.push(metric);
    }
    let delta = scalar((model.ri.weight.val().detach() - initial).abs().max())?;
    ensure!(delta > 0.0, "optimizer did not change RI head weights");
    let mut validation = Vec::new();
    let validation_model = model.valid();
    for entry in manifest
        .scenes
        .iter()
        .filter(|s| s.split == Split::Validation)
    {
        let scene = load_rgb(&dataset.join("raw").join(&entry.file))?;
        let grid = [
            scene.height / encoder_config.patch_size,
            scene.width / encoder_config.patch_size,
        ];
        let mask = visible_mask(
            grid[0] * grid[1],
            config.mask_ratio,
            config.seed,
            usize::MAX,
        )?;
        let views = encode_views::<B>(
            &encoder,
            &encoder_config,
            &scene,
            0,
            config.references,
            &mask,
            config.image_features,
            device,
        )?;
        let predictions = validation_model.forward(
            views.masked.inner(),
            views.full.inner(),
            views.references.into_iter().map(Tensor::inner).collect(),
            &mask,
            grid,
        )?;
        validation.push(scalar(
            reconstruction_loss(
                predictions,
                rgb_patches(views.target_rgb.inner(), encoder_config.patch_size, false),
                &mask,
                config.normalize_targets,
                config.predict_patch_stats,
            )
            .total,
        )?);
    }
    let stage = run.join("checkpoint.partial");
    fs::create_dir(&stage)?;
    model.save_file(stage.join("decoder"), &recorder)?;
    recorder.record(optim.to_record(), stage.join("optimizer"))?;
    let metadata = CheckpointMetadata {
        schema: 1,
        identity,
        dataset_id: manifest.dataset_id.clone(),
        encoder_id: encoder_id.clone(),
        completed_steps: config.steps,
        model_sha256: sha256_file(&stage.join("decoder.mpk"))?,
        optimizer_sha256: sha256_file(&stage.join("optimizer.mpk"))?,
    };
    write_json(&stage.join("metadata.json"), &metadata)?;
    let checkpoint = run.join(format!("checkpoint-{:06}", config.steps));
    fs::rename(stage, &checkpoint)?;
    let report = TrainReport {
        artifact_kind: if config.encoder == EncoderSource::DiagnosticTiny {
            "random_encoder_diagnostic"
        } else {
            "pretrained_encoder_preflight"
        }
        .into(),
        dataset_id: manifest.dataset_id,
        encoder_id,
        backend,
        start_step,
        completed_steps: config.steps,
        checkpoint,
        train: metrics,
        validation_total: validation.iter().sum::<f64>() / validation.len() as f64,
        decoder_weight_delta: delta,
    };
    write_json(&run.join("report.json"), &report)?;
    Ok(report)
}
