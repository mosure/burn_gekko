//! Standalone, weights-only assessment with common data, masks and readouts.
use crate::{
    latent::LatentModel,
    latent_pilot::{LatentConfig, WeightAncestor, audited_checkpoint},
    masking::MaskPattern,
    model::DecoderConfig,
    train::load_encoder,
};
use anyhow::{Result, ensure};
use burn::{
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::{Tensor, TensorData, backend::Backend},
};
use burn_gekko_data::{
    Split, load_dataset_rgb, read_config, read_dataset_manifest, sha256_file, write_config,
    write_json,
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssessmentModel {
    pub name: String,
    #[serde(flatten)]
    pub weights: WeightAncestor,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssessmentConfig {
    pub dataset: PathBuf,
    pub split: Split,
    pub rooms: usize,
    pub export_rooms: usize,
    pub mask_ratio: f32,
    pub mask_pattern: MaskPattern,
    pub seed: u64,
    pub stable_attention: bool,
    pub correspondence: bool,
    /// Optional count ablation using the same learned model and target views.
    #[serde(default)]
    pub references: Option<usize>,
    pub models: Vec<AssessmentModel>,
}
pub struct AssessedModel<B: Backend> {
    pub model: LatentModel<B>,
    pub teacher: burn_vjepa::VJepaEncoder<B>,
    pub config: LatentConfig,
    pub teacher_id: String,
}
pub fn load_assessed_model<B: Backend>(
    weights: &WeightAncestor,
    device: &B::Device,
) -> Result<AssessedModel<B>> {
    let parent = weights
        .checkpoint
        .parent()
        .ok_or_else(|| anyhow::anyhow!("checkpoint has no run parent"))?;
    let config: LatentConfig = read_config(&parent.join("config.toml"))?;
    let (teacher, ec, teacher_id) = load_encoder::<B>(&config.teacher, config.seed, device)?;
    ensure!(
        teacher_id == crate::e2e_pilot::REVIEWED_VJEPA_ID
            || matches!(config.teacher, crate::train::EncoderSource::DiagnosticTiny),
        "unreviewed teacher"
    );
    audited_checkpoint(weights, &teacher_id)?;
    let (encoder, _, _) = load_encoder::<B>(&config.teacher, config.seed, device)?;
    let mut model = LatentModel::new(
        encoder,
        ec.clone(),
        &DecoderConfig {
            encoder_dim: ec.encoder.embed_dim,
            width: config.decoder_width,
            depth: config.decoder_depth,
            heads: config.decoder_heads,
            patch: 16,
        },
        device,
    )?
    .prepare_spatial_descriptor(config.spatial_descriptor.as_ref())?
    .load_checked_record(
        NamedMpkFileRecorder::<FullPrecisionSettings>::default()
            .load(weights.checkpoint.join("model"), device)?,
    )?;
    model = model
        .with_spatial_input_layer(config.spatial_input_layer, false)?
        .with_spatial_input_scale(config.spatial_input_scale)?;
    model.validate_spatial_descriptor()?;
    model.fusion.decoder = model
        .fusion
        .decoder
        .with_cross_view_rope(config.cross_view_rope)
        .with_stable_attention(config.stable_attention);
    Ok(AssessedModel {
        model,
        teacher,
        config,
        teacher_id,
    })
}
pub fn run<B: Backend>(c: &AssessmentConfig, out: &Path, device: &B::Device) -> Result<()> {
    ensure!(
        !(c.stable_attention && std::any::type_name::<B>().contains("Fusion")),
        "float64 stable attention is unsupported by Burn CUDA Fusion in full evaluation; use float32"
    );
    ensure!(!out.exists(), "assessment output exists");
    ensure!(
        fs::canonicalize(out.ancestors().find(|p| p.exists()).unwrap())?
            .starts_with(fs::canonicalize(".data")?),
        "output outside .data"
    );
    ensure!(
        !c.models.is_empty() && c.rooms > 0 && c.rooms <= 256 && c.export_rooms <= c.rooms,
        "invalid assessment size"
    );
    ensure!(
        c.split != Split::Train && c.mask_ratio > 0. && c.mask_ratio < 1.,
        "invalid assessment split/mask"
    );
    let names: std::collections::BTreeSet<_> = c.models.iter().map(|m| m.name.as_str()).collect();
    ensure!(
        names.len() == c.models.len()
            && names.iter().all(|n| !n.is_empty()
                && n.chars()
                    .all(|x| x.is_ascii_alphanumeric() || x == '-' || x == '_')),
        "invalid/duplicate model names"
    );
    fs::create_dir_all(out)?;
    write_config(&out.join("config.toml"), c)?;
    let manifest = read_dataset_manifest(&c.dataset)?;
    let entries: Vec<_> = manifest
        .scenes
        .iter()
        .filter(|s| s.split == c.split)
        .take(c.rooms)
        .cloned()
        .collect();
    ensure!(entries.len() == c.rooms, "not enough assessment rooms");
    let scenes = entries
        .iter()
        .map(|e| load_dataset_rgb(&c.dataset, &manifest, e))
        .collect::<Result<Vec<_>>>()?;
    for item in &c.models {
        eprintln!("assessing {}", item.name);
        let parent = item
            .weights
            .checkpoint
            .parent()
            .ok_or_else(|| anyhow::anyhow!("checkpoint has no run parent"))?;
        let AssessedModel {
            mut model,
            teacher,
            mut config,
            teacher_id,
        } = load_assessed_model::<B>(&item.weights, device)?;
        let ec = model.encoder_config.clone();
        // Audit every recorded phase, including the legacy exact-resume path.
        fn audit_seeds(
            path: &Path,
            entries: &[burn_gekko_data::SceneEntry],
            test: bool,
            depth: usize,
        ) -> Result<()> {
            ensure!(depth < 32, "ancestry too deep");
            let p: serde_json::Value =
                serde_json::from_slice(&fs::read(path.join("provenance.json"))?)?;
            for key in if test {
                vec!["training_room_seeds", "validation_room_seeds"]
            } else {
                vec!["training_room_seeds"]
            } {
                let seen = p[key]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("missing seed ancestry"))?;
                ensure!(
                    !entries
                        .iter()
                        .any(|e| seen.iter().any(|x| x.as_u64() == Some(e.seed))),
                    "assessment overlaps prior training/selection seeds"
                );
            }
            let old: LatentConfig = read_config(&path.join("config.toml"))?;
            if let Some(w) = old.warm_start {
                audit_seeds(w.checkpoint.parent().unwrap(), entries, test, depth + 1)?;
            }
            if let Some(p) = p["resume"].as_str() {
                audit_seeds(Path::new(p).parent().unwrap(), entries, test, depth + 1)?;
            }
            Ok(())
        }
        audit_seeds(parent, &entries, c.split == Split::Test, 0)?;
        model.fusion.decoder = model
            .fusion
            .decoder
            .with_stable_attention(c.stable_attention);
        let bytes = fs::read(parent.join("train-position-mean.f32"))?;
        let n = scenes[0].height / 16 * (scenes[0].width / 16);
        ensure!(
            bytes.len() == n * ec.encoder.embed_dim * 4,
            "training mean shape mismatch"
        );
        let mean: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        ensure!(mean.iter().all(|x| x.is_finite()), "invalid training mean");
        config.dataset = c.dataset.clone();
        config.export_rooms = c.export_rooms;
        config.references = c.references.unwrap_or(config.references);
        ensure!(
            (1..=3).contains(&config.references) && config.references < manifest.config.cameras,
            "assessment reference count exceeds captured views"
        );
        config.eval_mask_ratio = Some(c.mask_ratio);
        config.eval_mask = c.mask_pattern;
        config.seed = c.seed;
        let dest = out.join(&item.name);
        crate::latent_eval::evaluate(
            &model,
            &teacher,
            &scenes,
            &entries,
            &config,
            Tensor::from_data(TensorData::new(mean, [1, n, ec.encoder.embed_dim]), device),
            &dest,
            true,
            true,
        )?;
        if c.correspondence {
            crate::correspondence::evaluate(
                &model,
                &teacher,
                &scenes,
                &entries,
                &c.dataset,
                &dest.join("correspondence.json"),
            )?;
        }
        write_json(
            &dest.join("provenance.json"),
            &serde_json::json!({"checkpoint":item.weights,"dataset_id":manifest.dataset_id,"teacher_id":teacher_id,
            "assessment_config_sha256":sha256_file(&out.join("config.toml"))?,"training_mean_sha256":sha256_file(&parent.join("train-position-mean.f32"))?,
            "noncommercial_weight_dependencies":[],"geometry_training_supervision":false,"status":"diagnostic; no external benchmark claim"}),
        )?;
    }
    Ok(())
}
