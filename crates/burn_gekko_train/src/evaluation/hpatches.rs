//! External RGB-only correspondence export. Homographies are intentionally absent.
use crate::{
    correspondence::{fixed_views, standard_readouts},
    latent_assess::{AssessmentModel, load_assessed_model},
};
use anyhow::{Result, ensure};
use burn::tensor::backend::Backend;
use burn_gekko_data::{sha256_file, write_config, write_json};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
};
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HpatchesConfig {
    pub images: PathBuf,
    pub images_sha256: String,
    pub models: Vec<AssessmentModel>,
    #[serde(default)]
    pub diagnostic_readouts: bool,
    #[serde(default)]
    pub encoder_layer_audit: bool,
    /// A single, preselected one-based encoder layer retained as a baseline.
    #[serde(default)]
    pub spatial_layer: Option<usize>,
    #[serde(default)]
    pub self_conditioned_readouts: bool,
    #[serde(default)]
    pub local_refinement: bool,
    #[serde(default)]
    pub evaluation_use: EvaluationUse,
}

#[derive(Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationUse {
    #[default]
    Development,
    HeldOut,
}
#[derive(Deserialize)]
struct Images {
    image_size: usize,
    sequences: Vec<Sequence>,
}
#[derive(Deserialize)]
struct Sequence {
    name: String,
    views: Vec<View>,
}
#[derive(Deserialize)]
struct View {
    file: PathBuf,
    sha256: String,
}

pub fn run<B: Backend>(c: &HpatchesConfig, out: &Path, device: &B::Device) -> Result<()> {
    ensure!(!out.exists(), "output exists");
    ensure!(
        !c.local_refinement
            || (c.self_conditioned_readouts
                && c.spatial_layer.is_some()
                && !c.diagnostic_readouts
                && !c.encoder_layer_audit),
        "local readout requires fixed spatial and same-image controls"
    );
    ensure!(
        fs::canonicalize(out.ancestors().find(|p| p.exists()).unwrap())?
            .starts_with(fs::canonicalize(".data")?),
        "output outside .data"
    );
    ensure!(
        sha256_file(&c.images)? == c.images_sha256,
        "external image manifest checksum mismatch"
    );
    let data: Images = serde_json::from_slice(&fs::read(&c.images)?)?;
    ensure!(
        data.sequences.len() == 116 && data.image_size == 256,
        "expected complete HPatches at 256"
    );
    ensure!(!c.models.is_empty(), "no assessment model");
    ensure!(
        c.spatial_layer.is_none_or(|x| x > 0)
            && !(c.encoder_layer_audit && c.spatial_layer.is_some()),
        "choose either a layer sweep or one fixed spatial control"
    );
    ensure!(
        !c.encoder_layer_audit || matches!(c.evaluation_use, EvaluationUse::Development),
        "encoder layer selection is a development-only diagnostic"
    );
    fs::create_dir_all(out)?;
    write_config(&out.join("config.toml"), c)?;
    for item in &c.models {
        ensure!(
            !item.name.is_empty()
                && item
                    .name
                    .chars()
                    .all(|x| x.is_ascii_alphanumeric() || x == '-' || x == '_'),
            "invalid model name"
        );
        let loaded = load_assessed_model::<B>(&item.weights, device)?;
        let (model, teacher) = (loaded.model, loaded.teacher);
        ensure!(
            !c.local_refinement
                || (loaded.config.spatial_input_layer == c.spatial_layer
                    && loaded.config.spatial_input_scale == 1.),
            "refined encoder route differs from checkpoint"
        );
        let file = out.join(format!("{}.jsonl", item.name));
        let mut log = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(file)?;
        for (s, seq) in data.sequences.iter().enumerate() {
            ensure!(seq.views.len() == 6, "HPatches requires six views");
            let mut rgb = Vec::new();
            for view in &seq.views {
                ensure!(
                    fs::canonicalize(&view.file)?.starts_with(fs::canonicalize(".data")?),
                    "image outside .data"
                );
                ensure!(
                    sha256_file(&view.file)? == view.sha256,
                    "external image checksum mismatch"
                );
                let bytes = fs::read(&view.file)?;
                ensure!(
                    bytes.len() == 256 * 256 * 3 * 4,
                    "invalid external RGB dimensions"
                );
                let values: Vec<f32> = bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|x| f32::from_le_bytes(*x))
                    .collect();
                ensure!(
                    values
                        .iter()
                        .all(|&x| x.is_finite() && (0. ..=1.).contains(&x)),
                    "invalid external RGB"
                );
                rgb.push(crate::encoder::upload_rgb(values, [1, 256, 256], device));
            }
            let student = model.encode_references(&rgb);
            let independent = c
                .self_conditioned_readouts
                .then(|| {
                    student
                        .iter()
                        .map(|x| {
                            crate::correspondence::self_conditioned_descriptor(
                                &model,
                                x.clone(),
                                [16, 16],
                            )
                        })
                        .collect::<Result<Vec<_>>>()
                })
                .transpose()?;
            let frozen =
                (!c.local_refinement).then(|| fixed_views(&teacher, &model.encoder_config, &rgb));
            let hierarchy =
                if !c.local_refinement && (c.encoder_layer_audit || c.spatial_layer.is_some()) {
                    let layers = c.spatial_layer.map_or_else(
                        || model.encoder_config.encoder.hierarchical_layers(),
                        |x| vec![x - 1],
                    );
                    Some((
                        crate::encoder_audit::capture_view_layers(
                            &model.encoder,
                            &model.encoder_config,
                            &rgb,
                            &layers,
                        )?,
                        crate::encoder_audit::capture_view_layers(
                            &teacher,
                            &model.encoder_config,
                            &rgb,
                            &layers,
                        )?,
                    ))
                } else {
                    None
                };
            for target in 1..6 {
                if c.local_refinement {
                    let features = independent.as_ref().unwrap();
                    let readouts = crate::evaluation::refinement::spatial_readouts(
                        &model,
                        student[target].clone(),
                        student[0].clone(),
                        [16, 16],
                        c.spatial_layer.unwrap(),
                        Some((features[target].clone(), features[0].clone())),
                    )?;
                    for r in readouts {
                        writeln!(
                            log,
                            "{}",
                            serde_json::json!({"sequence":seq.name,"target":target+1,"reference":1,"method":r.method,"grid":[16,16],"model_image_size":256,"indices":r.indices,"mutual":r.mutual,"coordinates":r.coordinates})
                        )?;
                    }
                    continue;
                }
                let mut readouts = standard_readouts(
                    &model,
                    student[target].clone(),
                    student[0].clone(),
                    frozen.as_ref().unwrap()[target].clone(),
                    frozen.as_ref().unwrap()[0].clone(),
                    [16, 16],
                )?;
                if let Some(features) = &independent {
                    readouts.extend(crate::correspondence::self_conditioned_readouts(
                        features[target].clone(),
                        features[0].clone(),
                    )?);
                }
                if c.diagnostic_readouts {
                    readouts.extend(crate::fusion_audit::readouts(
                        &model.fusion.decoder,
                        student[target].clone(),
                        student[0].clone(),
                        [16, 16],
                    )?);
                }
                if let Some((student, teacher)) = &hierarchy {
                    readouts.extend(crate::encoder_audit::readouts(
                        "student", student, target, 0,
                    )?);
                    readouts.extend(crate::encoder_audit::readouts(
                        "teacher", teacher, target, 0,
                    )?);
                }
                for (method, (indices, mutual)) in readouts {
                    writeln!(
                        log,
                        "{}",
                        serde_json::json!({"sequence":seq.name,"target":target+1,"reference":1,
                        "method":method,"grid":[16,16],"model_image_size":256,"indices":indices,"mutual":mutual})
                    )?;
                }
            }
            log.flush()?;
            if (s + 1) % 10 == 0 {
                eprintln!("HPatches {} sequence {}/116", item.name, s + 1);
            }
        }
        write_json(
            &out.join(format!("{}-provenance.json", item.name)),
            &serde_json::json!({"checkpoint":item.weights,
            "teacher_id":loaded.teacher_id,"image_manifest_sha256":c.images_sha256,"sequences":116,"pairs":580,
            "input":"RGB only; no homographies, depth or camera inputs", "readout":if c.local_refinement { "reciprocal conditional descriptors, hard indices and fixed local 3x3 probability centroid" } else { "hard nearest neighbour, final feature cosine or head/layer-mean reciprocal cross-attention" },
            "noncommercial_weight_dependencies":[],"evaluation_use":c.evaluation_use,
            "diagnostic_readouts":c.diagnostic_readouts,
            "encoder_layer_audit":c.encoder_layer_audit,
            "spatial_layer":c.spatial_layer,"local_refinement":c.local_refinement,"refinement":"optional 3x3 reciprocal probability centroid, temperature 0.07, no labels",
            "checkpoint_selection":"see experiment protocol; development readout screening is not held-out qualification"}),
        )?;
    }
    Ok(())
}
