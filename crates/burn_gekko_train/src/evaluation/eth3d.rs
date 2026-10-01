//! ETH3D interval correspondence export from RGB only. Point labels are absent
//! from the accepted manifest and are loaded only by the separate CPU scorer.
use crate::{
    correspondence::{fixed_views, standard_readouts},
    latent_assess::{AssessmentModel, load_assessed_model},
};
use anyhow::{Result, ensure};
use burn::tensor::{Tensor, backend::Backend};
use burn_gekko_data::{read_config, sha256_file, write_config, write_json};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
};
mod canonical;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Eth3dConfig {
    pub images: PathBuf,
    pub images_sha256: String,
    pub models: Vec<AssessmentModel>,
    pub selection_record: PathBuf,
    pub selection_sha256: String,
    #[serde(default)]
    pub spatial_layer: Option<usize>,
    #[serde(default)]
    pub self_conditioned_readouts: bool,
    /// Only the three registered spatial readouts, with live parity checks.
    #[serde(default)]
    pub focused_spatial_readouts: bool,
    /// Emit the same focused family from the full canonical computations.
    /// Hard and local coordinates share score arrays; no optimized-path parity is assumed.
    #[serde(default)]
    pub canonical_spatial_readouts: bool,
    /// Also export fixed 3x3 probability centroids from the same score matrices.
    #[serde(default)]
    pub local_refinement: bool,
    /// Fail after throughput qualification if the complete protocol cannot fit.
    #[serde(default)]
    pub max_seconds: Option<u64>,
    #[serde(default)]
    pub evaluation_use: crate::hpatches::EvaluationUse,
}
#[derive(Deserialize)]
struct Images {
    image_size: usize,
    images: BTreeMap<String, View>,
    pairs: Vec<Pair>,
}
#[derive(Deserialize)]
struct View {
    file: PathBuf,
    sha256: String,
}
#[derive(Deserialize)]
struct Pair {
    id: String,
    scene: String,
    interval: usize,
    target: String,
    reference: String,
}

#[derive(Deserialize)]
struct SealedModels {
    models: Vec<AssessmentModel>,
    spatial_layer: Option<usize>,
    #[serde(default)]
    self_conditioned_readouts: bool,
    #[serde(default)]
    focused_spatial_readouts: bool,
    #[serde(default)]
    canonical_spatial_readouts: bool,
    #[serde(default)]
    local_refinement: bool,
}

/// Bind actual inference inputs to the sealed record before loading any model.
/// The independent scorer additionally checks all declared readout names.
fn verify_selection(c: &Eth3dConfig) -> Result<()> {
    ensure!(
        sha256_file(&c.selection_record)? == c.selection_sha256,
        "candidate selection must be frozen before export"
    );
    let selection: SealedModels = read_config(&c.selection_record)?;
    ensure!(
        selection.spatial_layer == c.spatial_layer
            && selection.self_conditioned_readouts == c.self_conditioned_readouts
            && selection.focused_spatial_readouts == c.focused_spatial_readouts
            && selection.canonical_spatial_readouts == c.canonical_spatial_readouts
            && selection.local_refinement == c.local_refinement,
        "spatial control differs from sealed selection"
    );
    let names = |models: &[AssessmentModel]| {
        models
            .iter()
            .map(|m| m.name.clone())
            .collect::<std::collections::BTreeSet<_>>()
    };
    let selected_names = names(&selection.models);
    ensure!(
        selected_names.len() == selection.models.len()
            && selected_names.len() == c.models.len()
            && selected_names == names(&c.models),
        "model set differs from sealed selection"
    );
    for model in &c.models {
        let selected = selection
            .models
            .iter()
            .find(|item| item.name == model.name)
            .unwrap();
        ensure!(
            selected.weights.model_sha256 == model.weights.model_sha256
                && fs::canonicalize(&selected.weights.checkpoint)?
                    == fs::canonicalize(&model.weights.checkpoint)?,
            "model weights differ from sealed selection: {}",
            model.name
        );
    }
    Ok(())
}

fn rgb<B: Backend>(v: &View, device: &B::Device) -> Result<Tensor<B, 4>> {
    ensure!(
        fs::canonicalize(&v.file)?.starts_with(fs::canonicalize(".data")?),
        "image outside .data"
    );
    ensure!(sha256_file(&v.file)? == v.sha256, "RGB checksum mismatch");
    let bytes = fs::read(&v.file)?;
    ensure!(bytes.len() == 256 * 256 * 3 * 4, "invalid RGB dimensions");
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
        "invalid RGB range"
    );
    Ok(crate::encoder::upload_rgb(values, [1, 256, 256], device))
}

pub fn run<B: Backend>(c: &Eth3dConfig, out: &Path, device: &B::Device) -> Result<()> {
    let wall = std::time::Instant::now();
    ensure!(
        !c.canonical_spatial_readouts || (c.focused_spatial_readouts && c.local_refinement),
        "canonical spatial export requires the focused family and local refinement"
    );
    ensure!(
        !c.local_refinement || c.focused_spatial_readouts,
        "local readout requires focused controls"
    );
    ensure!(
        !c.focused_spatial_readouts
            || (c.self_conditioned_readouts && c.spatial_layer.is_some() && c.models.len() == 1),
        "focused export requires one model, a fixed spatial layer and same-image controls"
    );
    ensure!(
        c.max_seconds.is_none_or(|s| (30..=43200).contains(&s)),
        "invalid export time limit"
    );
    ensure!(!out.exists(), "preserve existing benchmark outputs");
    ensure!(
        fs::canonicalize(out.ancestors().find(|p| p.exists()).unwrap())?
            .starts_with(fs::canonicalize(".data")?),
        "output outside .data"
    );
    ensure!(
        sha256_file(&c.images)? == c.images_sha256,
        "image manifest checksum mismatch"
    );
    verify_selection(c)?;
    let manifest: Images = serde_json::from_slice(&fs::read(&c.images)?)?;
    ensure!(
        manifest.image_size == 256 && manifest.pairs.len() == 3365 && manifest.images.len() == 2448,
        "expected complete ETH3D interval data"
    );
    ensure!(!c.models.is_empty(), "no candidate models");
    ensure!(
        c.spatial_layer.is_none_or(|x| x > 0),
        "spatial control uses a one-based encoder layer"
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
        let model = loaded.model;
        ensure!(
            !c.focused_spatial_readouts
                || (loaded.config.spatial_input_layer == c.spatial_layer
                    && loaded.config.spatial_input_scale == 1.),
            "focused control differs from checkpoint spatial route"
        );
        let mut focused_times = Vec::new();
        let mut log = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(out.join(format!("{}.jsonl", item.name)))?;
        for (i, pair) in manifest.pairs.iter().enumerate() {
            let tick = std::time::Instant::now();
            ensure!(
                c.max_seconds.is_none_or(|s| wall.elapsed().as_secs() < s),
                "complete ETH3D export exceeded its time ceiling"
            );
            let views = vec![
                rgb::<B>(&manifest.images[&pair.target], device)?,
                rgb::<B>(&manifest.images[&pair.reference], device)?,
            ];
            let student = model.encode_references(&views);
            let refined = if c.canonical_spatial_readouts {
                Some(canonical::readouts(
                    &model,
                    &loaded.teacher,
                    &views,
                    &student,
                    c.spatial_layer.unwrap(),
                )?)
            } else {
                c.local_refinement
                    .then(|| {
                        crate::evaluation::refinement::spatial_readouts(
                            &model,
                            student[0].clone(),
                            student[1].clone(),
                            [16, 16],
                            c.spatial_layer.unwrap(),
                            None,
                        )
                    })
                    .transpose()?
            };
            let focused = if let Some(refined) = &refined {
                Some(
                    refined
                        .iter()
                        .filter(|r| r.coordinates.is_none())
                        .map(|r| (r.method.clone(), (r.indices.clone(), r.mutual.clone())))
                        .collect::<Vec<_>>(),
                )
            } else if c.focused_spatial_readouts {
                Some(crate::correspondence::focused_spatial_readouts(
                    &model,
                    student[0].clone(),
                    student[1].clone(),
                    [16, 16],
                    c.spatial_layer.unwrap(),
                )?)
            } else {
                None
            };
            if focused.is_some() {
                focused_times.push(tick.elapsed().as_secs_f64());
            }
            // First eight pairs qualify exact native-backend indices AND mutual
            // flags against the original implementation, without reading labels.
            let audit = !c.canonical_spatial_readouts && (!c.focused_spatial_readouts || i < 8);
            let mut readouts = if audit {
                let teacher = fixed_views(&loaded.teacher, &model.encoder_config, &views);
                standard_readouts(
                    &model,
                    student[0].clone(),
                    student[1].clone(),
                    teacher[0].clone(),
                    teacher[1].clone(),
                    [16, 16],
                )?
            } else {
                Vec::new()
            };
            if audit && c.self_conditioned_readouts {
                let a = crate::correspondence::self_conditioned_descriptor(
                    &model,
                    student[0].clone(),
                    [16, 16],
                )?;
                let b = crate::correspondence::self_conditioned_descriptor(
                    &model,
                    student[1].clone(),
                    [16, 16],
                )?;
                readouts.extend(crate::correspondence::self_conditioned_readouts(a, b)?);
            }
            if let Some(layer) = c.spatial_layer.filter(|_| audit) {
                for (name, encoder) in [("student", &model.encoder), ("teacher", &loaded.teacher)] {
                    let layers = crate::encoder_audit::capture_view_layers(
                        encoder,
                        &model.encoder_config,
                        &views,
                        &[layer - 1],
                    )?;
                    readouts.extend(crate::encoder_audit::readouts(name, &layers, 0, 1)?);
                }
            }
            if let Some(focused) = focused {
                if audit {
                    for (name, prediction) in &focused {
                        ensure!(
                            readouts.iter().any(|(n, p)| n == name && p == prediction),
                            "focused native parity failed for {name}"
                        );
                    }
                }
                readouts = focused;
                if i == 63 {
                    let mut times = focused_times[8..].to_vec();
                    times.sort_by(f64::total_cmp);
                    let p95 = times[(times.len() - 1) * 95 / 100];
                    let estimate = wall.elapsed().as_secs_f64()
                        + p95 * (manifest.pairs.len() - i - 1) as f64 * 1.15
                        + 2.;
                    write_json(
                        &out.join("focused-qualification.json"),
                        &serde_json::json!({"parity_pairs":if c.canonical_spatial_readouts {0}else{8},"parity_indices_and_mutual_flags":if c.canonical_spatial_readouts {"shared canonical score arrays; checked for every pair"}else{"exact"},"canonical_spatial_readouts":c.canonical_spatial_readouts,"qualification_pairs":64,"p95_pair_seconds":p95,"estimated_total_seconds_with_15_percent_margin":estimate,"ceiling_seconds":c.max_seconds,"backend":std::any::type_name::<B>()}),
                    )?;
                    ensure!(
                        c.max_seconds.is_none_or(|s| estimate < s as f64),
                        "complete ETH3D protocol cannot fit; export stopped after qualification"
                    );
                }
            }
            for (method, (indices, mutual)) in readouts {
                writeln!(
                    log,
                    "{}",
                    serde_json::json!({"pair":pair.id,"scene":pair.scene,"interval":pair.interval,"method":method,"grid":[16,16],"model_image_size":256,"indices":indices,"mutual":mutual})
                )?;
            }
            if let Some(refined) = refined {
                for r in refined.into_iter().filter(|r| r.coordinates.is_some()) {
                    writeln!(
                        log,
                        "{}",
                        serde_json::json!({"pair":pair.id,"scene":pair.scene,"interval":pair.interval,"method":r.method,"grid":[16,16],"model_image_size":256,"indices":r.indices,"mutual":r.mutual,"coordinates":r.coordinates})
                    )?;
                }
            }
            if (i + 1) % 100 == 0 {
                log.flush()?;
                eprintln!("ETH3D {} pair {}/3365", item.name, i + 1);
            }
        }
        write_json(
            &out.join(format!("{}-provenance.json", item.name)),
            &serde_json::json!({"checkpoint":item.weights,"teacher_id":loaded.teacher_id,"image_manifest_sha256":c.images_sha256,"selection_sha256":c.selection_sha256,"spatial_layer":c.spatial_layer,"pairs":3365,"input":"RGB only; no points, depth, homographies or cameras","evaluation_use":c.evaluation_use,"local_refinement":c.local_refinement,"canonical_spatial_readouts":c.canonical_spatial_readouts,"refinement":"3x3 reciprocal probability centroid, temperature 0.07, no labels","noncommercial_weight_dependencies":[]}),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::latent_pilot::WeightAncestor;

    #[test]
    fn sealed_model_selection_rejects_changes_before_inference() -> Result<()> {
        let temp = tempfile::tempdir()?;
        let checkpoint = temp.path().join("checkpoint");
        fs::create_dir(&checkpoint)?;
        let definition = AssessmentModel {
            name: "candidate".into(),
            weights: WeightAncestor {
                checkpoint,
                model_sha256: "a".repeat(64),
            },
        };
        let selection_record = temp.path().join("selection.toml");
        #[derive(Serialize)]
        struct Selection<'a> {
            models: &'a [AssessmentModel],
            spatial_layer: usize,
        }
        write_config(
            &selection_record,
            &Selection {
                models: std::slice::from_ref(&definition),
                spatial_layer: 6,
            },
        )?;
        let mut config = Eth3dConfig {
            evaluation_use: crate::hpatches::EvaluationUse::Development,
            images: temp.path().join("unopened-images.json"),
            images_sha256: "unused".into(),
            models: vec![definition],
            selection_sha256: sha256_file(&selection_record)?,
            selection_record,
            spatial_layer: Some(6),
            self_conditioned_readouts: false,
            focused_spatial_readouts: false,
            canonical_spatial_readouts: false,
            local_refinement: false,
            max_seconds: None,
        };
        verify_selection(&config)?;
        config.canonical_spatial_readouts = true;
        assert!(verify_selection(&config).is_err());
        config.canonical_spatial_readouts = false;
        config.local_refinement = true;
        assert!(verify_selection(&config).is_err());
        config.local_refinement = false;
        config.self_conditioned_readouts = true;
        assert!(verify_selection(&config).is_err());
        config.self_conditioned_readouts = false;
        config.focused_spatial_readouts = true;
        assert!(verify_selection(&config).is_err());
        config.focused_spatial_readouts = false;
        config.models[0].weights.model_sha256 = "b".repeat(64);
        assert!(verify_selection(&config).is_err());
        config.models[0].weights.model_sha256 = "a".repeat(64);
        config.spatial_layer = Some(9);
        assert!(verify_selection(&config).is_err());
        config.spatial_layer = Some(6);
        config.models.push(config.models[0].clone());
        assert!(verify_selection(&config).is_err());
        config.models.pop();
        config.models[0].name = "unselected".into();
        assert!(verify_selection(&config).is_err());
        Ok(())
    }
}
