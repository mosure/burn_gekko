//! Frozen RGB-only export for calibrated pose probes. No camera labels are loaded.
use crate::{
    correspondence,
    latent_assess::{AssessmentModel, load_assessed_model},
};
use anyhow::{Context, Result, ensure};
use burn::tensor::{Tensor, backend::Backend};
use burn_gekko_data::{real_views::PairImages, sha256_file, write_config, write_json};
use burn_gekko_eval::pose::benchmark::Prediction;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoseExportSelection {
    pub images_sha256: String,
    pub model: AssessmentModel,
    pub spatial_layer: usize,
    pub evaluation_use: String,
    pub methods: Vec<String>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoseExportConfig {
    pub images: PathBuf,
    pub images_sha256: String,
    pub model: AssessmentModel,
    pub spatial_layer: usize,
    pub selection: PathBuf,
    pub selection_sha256: String,
}
pub fn run<B: Backend>(c: &PoseExportConfig, out: &Path, device: &B::Device) -> Result<()> {
    ensure!(!out.exists(), "preserve existing RGB export");
    ensure!(
        fs::canonicalize(
            out.ancestors()
                .find(|p| p.exists())
                .context("output parent")?
        )?
        .starts_with(fs::canonicalize(".data")?),
        "output outside .data"
    );
    ensure!(
        sha256_file(&c.images)? == c.images_sha256
            && sha256_file(&c.selection)? == c.selection_sha256,
        "pose selection/input hash mismatch"
    );
    let sealed: PoseExportSelection = burn_gekko_data::read_config(&c.selection)?;
    ensure!(
        sealed.images_sha256 == c.images_sha256
            && sealed.spatial_layer == c.spatial_layer
            && sealed.model.weights.model_sha256 == c.model.weights.model_sha256
            && sealed.model.name == c.model.name
            && fs::canonicalize(&sealed.model.weights.checkpoint)?
                == fs::canonicalize(&c.model.weights.checkpoint)?,
        "pose selection differs from inference config"
    );
    let images: PairImages = serde_json::from_slice(&fs::read(&c.images)?)?;
    let grid = burn_gekko_data::real_views::descriptor_grid(images.image_size)?;
    let side = images.image_size;
    ensure!(
        images.schema == 1
            && !images.pairs.is_empty()
            && images.evaluation_use == sealed.evaluation_use
            && matches!(images.evaluation_use.as_str(), "held_out" | "development"),
        "invalid RGB pair manifest"
    );
    let methods = vec![
        "spatial_residual_conditional_local".into(),
        "spatial_self_conditional_local".into(),
        format!("student_l{:02}_centered_conditional_local", c.spatial_layer),
    ];
    ensure!(
        methods == sealed.methods,
        "readout set differs from sealed selection"
    );
    ensure!(
        images
            .pairs
            .iter()
            .map(|p| &p.id)
            .collect::<BTreeSet<_>>()
            .len()
            == images.pairs.len(),
        "duplicate RGB pair IDs"
    );
    let used = images
        .pairs
        .iter()
        .flat_map(|p| [&p.target, &p.reference])
        .collect::<BTreeSet<_>>();
    ensure!(
        used.len() == images.images.len() && used.iter().all(|k| images.images.contains_key(*k)),
        "missing or unused images"
    );
    fs::create_dir_all(out)?;
    write_config(&out.join("config.toml"), c)?;
    let started = std::time::Instant::now();
    let loaded = load_assessed_model::<B>(&c.model.weights, device)?;
    ensure!(
        loaded.config.spatial_input_layer == Some(c.spatial_layer)
            && loaded.config.spatial_input_scale == 1.,
        "spatial baseline differs from checkpoint"
    );
    let model = loaded.model;
    drop(loaded.teacher);
    let mut student: BTreeMap<String, Tensor<B, 3>> = BTreeMap::new();
    let mut independent = BTreeMap::new();
    for (key, view) in &images.images {
        ensure!(
            fs::canonicalize(&view.file)?.starts_with(fs::canonicalize(".data")?)
                && sha256_file(&view.file)? == view.sha256,
            "RGB checksum/path mismatch"
        );
        let bytes = fs::read(&view.file)?;
        ensure!(bytes.len() == side * side * 3 * 4, "invalid RGB size");
        let values = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect::<Vec<_>>();
        ensure!(
            values
                .iter()
                .all(|v| v.is_finite() && (0. ..=1.).contains(v)),
            "invalid RGB values"
        );
        let x = model
            .encode_references(&[crate::encoder::upload_rgb(values, [1, side, side], device)])
            .remove(0);
        ensure!(
            x.dims()[1] == grid[0] * grid[1],
            "encoder output grid mismatch"
        );
        let sa = correspondence::self_conditioned_descriptor(&model, x.clone(), grid)?;
        student.insert(key.clone(), x);
        independent.insert(key.clone(), sa);
    }
    eprintln!(
        "encoded {} RGB images in {:.2}s",
        student.len(),
        started.elapsed().as_secs_f64()
    );
    let path = out.join("predictions.jsonl");
    let mut log = std::io::BufWriter::new(
        fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?,
    );
    for (i, pair) in images.pairs.iter().enumerate() {
        let readouts = super::refinement::spatial_readouts(
            &model,
            student[&pair.target].clone(),
            student[&pair.reference].clone(),
            grid,
            c.spatial_layer,
            Some((
                independent[&pair.target].clone(),
                independent[&pair.reference].clone(),
            )),
        )?;
        for r in readouts.into_iter().filter(|r| r.coordinates.is_some()) {
            ensure!(methods.contains(&r.method), "unexpected pose readout");
            let prediction = Prediction {
                pair: pair.id.clone(),
                method: r.method,
                grid,
                indices: r.indices,
                mutual: r.mutual,
                coordinates: r.coordinates.unwrap(),
            };
            writeln!(log, "{}", serde_json::to_string(&prediction)?)?;
        }
        if (i + 1) % 64 == 0 {
            eprintln!(
                "exported {}/{} pairs in {:.2}s",
                i + 1,
                images.pairs.len(),
                started.elapsed().as_secs_f64()
            );
        }
    }
    log.flush()?;
    write_json(
        &out.join("provenance.json"),
        &serde_json::json!({"schema":1,"checkpoint_sha256":c.model.weights.model_sha256,"images_sha256":c.images_sha256,"predictions_sha256":sha256_file(&path)?,"selection_sha256":c.selection_sha256,"evaluation_use":images.evaluation_use,"image_size":side,"grid":grid,"methods":methods,"pairs":images.pairs.len(),"unique_images":images.images.len(),"model_input":"RGB only; strict manifest rejects geometry and calibration; all pose solving runs separately on CPU","refinement":"fixed reciprocal conditional T=0.07, 3x3 local probability centroid; mutual flags from unchanged hard readout","seconds":started.elapsed().as_secs_f64(),"noncommercial_weight_dependencies":[]}),
    )?;
    Ok(())
}
