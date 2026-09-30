//! Bounded end-to-end RGB-only studies. Raw scenes stay on disk in .data;
//! resident tensors contain RGB only, never cached trainable encoder features.
use crate::{
    batch::SampleSchedule,
    e2e::{ReconstructionModel, RgbHead, UnfreezeGate, take_gradients},
    encoder::{ImageFeatures, image_tensor, visible_mask},
    model::{DecoderConfig, DecoderPosition, GekkoDecoder},
    train::{EncoderSource, clip, load_encoder, scalar},
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
use burn_vjepa::{VJepaConfig, VJepaEncoder};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

mod config;
pub(crate) use config::REVIEWED_VJEPA_ID;
pub use config::{E2eConfig, Initialization, RgbCache};

pub fn initialize<B: Backend>(
    c: &E2eConfig,
    device: &B::Device,
) -> Result<(ReconstructionModel<B>, String)> {
    c.validate()?;
    B::seed(device, c.seed);
    let (encoder, encoder_config, encoder_id) = match &c.initialization {
        Initialization::Scratch {
            width,
            depth,
            heads,
        } => {
            let mut config = VJepaConfig::default();
            config.encoder.embed_dim = *width;
            config.encoder.depth = *depth;
            config.encoder.num_heads = *heads;
            config.encoder.n_output_distillation = 1;
            let encoder = VJepaEncoder::new(&config, device);
            let id = fingerprint(&("random-vjepa21-image-encoder", &config, c.seed))?;
            (encoder, config, id)
        }
        Initialization::Vjepa21 {
            directory,
            expected_encoder_id,
        } => {
            let (enc, cfg, id) = load_encoder(
                &EncoderSource::Burnpack {
                    directory: directory.clone(),
                },
                c.seed,
                device,
            )?;
            ensure!(
                &id == expected_encoder_id,
                "encoder package identity changed"
            );
            (enc, cfg, id)
        }
    };
    let encoder = encoder.clone().load_record(encoder.into_record());
    B::seed(device, c.seed.wrapping_add(5));
    let input = if c.appearance_bypass {
        ImageFeatures::SemanticRgb
    } else {
        ImageFeatures::Semantic
    };
    let mut decoder = GekkoDecoder::with_reconstruction(
        &DecoderConfig {
            encoder_dim: input.width(encoder_config.encoder.embed_dim),
            width: c.decoder_width,
            depth: c.decoder_depth,
            heads: c.decoder_heads,
            patch: 16,
        },
        DecoderPosition::Rope2d,
        c.rgb_head == RgbHead::Calibrated,
        device,
    )?
    .with_mae_context_before_self(true)
    .with_attention_normalization(c.decoder_qk_norm, device)
    .with_stable_attention(c.decoder_stable_attention);
    if c.appearance_transport {
        decoder.appearance_head = Some(
            crate::appearance::AppearanceHead::new(
                c.decoder_width,
                c.transport_max_displacement,
                c.transport_loss_weight,
                device,
            )
            .with_pyramid_loss(c.transport_pyramid_loss)
            .with_coarse_smoothness(c.transport_coarse_smoothness_weight),
        );
        if c.transport_matching {
            decoder.appearance_head.as_mut().unwrap().matcher =
                Some(crate::matching::RgbMatcher::new(device));
        }
    }
    let model = ReconstructionModel {
        encoder,
        decoder,
        encoder_config,
        appearance_bypass: c.appearance_bypass,
        rgb_head: c.rgb_head,
    };
    // Materialize once before any validation clone; Burn initializers are lazy.
    Ok((model.clone().load_record(model.into_record()), encoder_id))
}

pub struct Scene<B: Backend> {
    pub seed: u64,
    pub rgb: Vec<Tensor<B, 4>>,
}
enum TrainingScenes<B: Backend> {
    Device(Vec<Scene<B>>),
    Host(Vec<burn_gekko_data::RgbScene>),
}
impl<B: Backend> TrainingScenes<B> {
    fn len(&self) -> usize {
        match self {
            Self::Device(s) => s.len(),
            Self::Host(s) => s.len(),
        }
    }
    fn seed(&self, index: usize) -> u64 {
        match self {
            Self::Device(s) => s[index].seed,
            Self::Host(s) => s[index].seed,
        }
    }
    fn batch(
        &self,
        samples: &[(usize, usize)],
        references: usize,
        device: &B::Device,
    ) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        match self {
            Self::Device(s) => batch(s, samples, references, false),
            Self::Host(s) => host_batch(s, samples, references, device),
        }
    }
}
/// Raw HWC floats stay in bounded host memory; each batch uploads only its RGB.
/// No dense/masked trainable encoder features are cached across updates.
pub fn host_batch<B: Backend>(
    scenes: &[burn_gekko_data::RgbScene],
    samples: &[(usize, usize)],
    references: usize,
    device: &B::Device,
) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
    let h = scenes[0].height;
    let w = scenes[0].width;
    let pack = |offset: usize| {
        let data = samples
            .iter()
            .flat_map(|&(s, v)| {
                scenes[s].views[(v + offset) % scenes[s].views.len()]
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        crate::encoder::upload_rgb(data, [samples.len(), h, w], device)
    };
    (pack(0), (1..=references).map(pack).collect())
}
fn batch<B: Backend>(
    scenes: &[Scene<B>],
    samples: &[(usize, usize)],
    refs: usize,
    unrelated: bool,
) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
    let target = Tensor::cat(
        samples
            .iter()
            .map(|&(s, v)| scenes[s].rgb[v].clone())
            .collect(),
        0,
    );
    let refs = (1..=refs)
        .map(|i| {
            Tensor::cat(
                samples
                    .iter()
                    .map(|&(s, v)| {
                        let s = if unrelated { (s + 1) % scenes.len() } else { s };
                        scenes[s].rgb[(v + i) % scenes[s].rgb.len()].clone()
                    })
                    .collect(),
                0,
            )
        })
        .collect();
    (target, refs)
}
fn export<B: Backend>(path: &Path, x: Tensor<B, 4>) -> Result<()> {
    let values = x
        .permute([0, 2, 3, 1])
        .into_data()
        .convert::<f32>()
        .to_vec::<f32>()?;
    let bytes: Vec<_> = values.into_iter().flat_map(f32::to_le_bytes).collect();
    fs::write(path, bytes)?;
    Ok(())
}
#[allow(clippy::too_many_arguments)]
pub fn evaluate<B: Backend>(
    model: &ReconstructionModel<B>,
    scenes: &[Scene<B>],
    c: &E2eConfig,
    output: &Path,
    rooms: usize,
    unrelated: bool,
    save_all: bool,
) -> Result<f64> {
    fs::create_dir_all(output)?;
    let [_, _, h, w] = scenes[0].rgb[0].dims();
    // Shared evaluation mask across study arms, independent of train RNG.
    let mask = visible_mask(h / 16 * (w / 16), c.mask_ratio, 79, 0)?;
    let pixels: Vec<f32> = (0..h * w)
        .map(|i| {
            if mask
                .indices()
                .contains(&(i / w / 16 * (w / 16) + i % w / 16))
            {
                0.
            } else {
                1.
            }
        })
        .collect();
    let hidden = Tensor::<B, 4>::from_data(
        burn::tensor::TensorData::new(pixels, [1, 1, h, w]),
        &scenes[0].rgb[0].device(),
    );
    let mut rows = Vec::new();
    for (s, scene) in scenes.iter().take(rooms).enumerate() {
        for v in 0..scene.rgb.len() {
            let (target, refs) = batch(scenes, &[(s, v)], c.references, unrelated);
            let out = model.complete(target.clone(), &refs, &mask)?;
            let mse = scalar(
                ((out.rgb.clone() - target.clone()).powf_scalar(2.) * hidden.clone()).sum()
                    / (hidden.clone().sum() * 3.),
            )?;
            let mono = scalar(
                ((out.monocular.clone() - target.clone()).powf_scalar(2.) * hidden.clone()).sum()
                    / (hidden.clone().sum() * 3.),
            )?;
            rows.push(serde_json::json!({"room_seed":scene.seed,"target_view":v,"hidden_rgb_mse":mse,"monocular_hidden_rgb_mse":mono}));
            // Every requested probe room contributes a view-zero image. The
            // room limit controls cost; silently limiting exports to four
            // rooms would make detailed quality curves narrower than the
            // configured validation probe.
            if save_all || v == 0 {
                let dir = output.join(format!("room-{}-view-{v}", scene.seed));
                fs::create_dir_all(&dir)?;
                export(&dir.join("target.f32"), target)?;
                export(&dir.join("prediction.f32"), out.rgb)?;
                export(&dir.join("monocular.f32"), out.monocular)?;
                if let Some(transport) = out.transport {
                    export(&dir.join("transported.f32"), transport.rgb)?;
                    export(&dir.join("generated.f32"), out.generated_rgb.unwrap())?;
                    export(&dir.join("mixture.f32"), out.appearance_mixture.unwrap())?;
                    for (i, flow) in transport.flows.into_iter().enumerate() {
                        export(&dir.join(format!("flow-{i}.f32")), flow)?;
                    }
                    for (i, flow) in transport.coarse_flows.into_iter().enumerate() {
                        export(&dir.join(format!("coarse-flow-{i}.f32")), flow)?;
                    }
                }
                for (i, r) in refs.into_iter().enumerate() {
                    export(&dir.join(format!("reference-{i}.f32")), r)?;
                }
                write_json(
                    &dir.join("sample.json"),
                    &serde_json::json!({"room_seed":scene.seed,"target_view":v,"height":h,"width":w,"visible_patch_ids":mask.indices(),"hidden_rgb_mse":mse,"monocular_hidden_rgb_mse":mono,"reference_count":c.references,"unrelated":unrelated,"oracle_statistics":false,"appearance_transport":c.appearance_transport,"method":if c.appearance_transport {"end-to-end Gekko fusion with RGB reference transport"}else{"end-to-end Gekko fusion"}}),
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
#[derive(Clone, Serialize, Deserialize)]
struct Snapshot {
    identity: String,
    dataset_id: String,
    encoder_id: String,
    completed_steps: usize,
    gate: UnfreezeGate,
    model_sha256: String,
    encoder_optimizer_sha256: String,
    decoder_optimizer_sha256: String,
    noncommercial_weight_dependencies: Vec<String>,
    #[serde(default)]
    backend: String,
    #[serde(default)]
    warm_start: Option<WarmStart>,
}
#[derive(Clone, Serialize, Deserialize)]
struct WarmStart {
    checkpoint: PathBuf,
    model_sha256: String,
    source_metadata_sha256: String,
    source_identity: String,
    source_dataset_id: String,
    source_completed_steps: usize,
    optimizer_reset: bool,
}
fn checkpoint<
    B: AutodiffBackend,
    E: Optimizer<VJepaEncoder<B>, B>,
    D: Optimizer<GekkoDecoder<B>, B>,
>(
    model: &ReconstructionModel<B>,
    enc_opt: &E,
    dec_opt: &D,
    path: &Path,
    mut metadata: Snapshot,
) -> Result<()> {
    ensure!(!path.exists(), "checkpoint exists");
    let stage = path.with_extension("partial");
    ensure!(!stage.exists(), "partial checkpoint exists");
    fs::create_dir(&stage)?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    model.clone().save_file(stage.join("model"), &recorder)?;
    recorder.record(enc_opt.to_record(), stage.join("encoder-optimizer"))?;
    recorder.record(dec_opt.to_record(), stage.join("decoder-optimizer"))?;
    metadata.model_sha256 = sha256_file(&stage.join("model.mpk"))?;
    metadata.encoder_optimizer_sha256 = sha256_file(&stage.join("encoder-optimizer.mpk"))?;
    metadata.decoder_optimizer_sha256 = sha256_file(&stage.join("decoder-optimizer.mpk"))?;
    if metadata.encoder_id == REVIEWED_VJEPA_ID {
        fs::write(
            stage.join("LICENSE.vjepa.txt"),
            burn_vjepa::provenance::WEIGHTS_LICENSE,
        )?;
        fs::write(
            stage.join("NOTICE.md"),
            "This checkpoint adapts the MIT-licensed V-JEPA 2.1 image encoder from Meta.\nOriginal source: https://github.com/facebookresearch/vjepa2/tree/204698b45b3712590f06245fbfba32d3be539812\nOfficial weights: https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt\nFusion and RGB/RI heads were trained in this workspace. No released Gekko parameters or noncommercial teacher weights were used.\nRetain LICENSE.vjepa.txt when redistributing this checkpoint. See metadata.json for checkpoint lineage.\n",
        )?;
    }
    write_json(&stage.join("metadata.json"), &metadata)?;
    fs::rename(stage, path)?;
    Ok(())
}
mod runner;
pub use runner::{run, run_with_warm_start};
