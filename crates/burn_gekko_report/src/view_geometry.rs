//! Fixed first-room visualizations of actual camera-view correspondences.
use crate::{
    artifact::{Source, record},
    correspondence::{CorrespondenceFigure, line, point},
    experiment::Experiment,
};
use anyhow::{Context, Result, ensure};
use image::{Rgb, RgbImage};
use serde_json::Value;
use std::{fs, path::Path};

pub(crate) fn training_provenance(
    config: &Value,
    run: &Path,
    dataset_id: &Value,
    sources: &mut Vec<Source>,
) -> Result<Value> {
    if !config["view_geometry"].is_object() {
        return Ok(Value::Null);
    }
    let path = run.join("view-geometry-provenance.json");
    record(&path, sources)?;
    let evidence: Value = serde_json::from_slice(&fs::read(path)?)?;
    ensure!(
        evidence["config"] == config["view_geometry"],
        "geometry training provenance differs"
    );
    let root = Path::new(
        config["view_geometry"]["cache"]
            .as_str()
            .context("geometry cache path")?,
    );
    let manifest = root.join("manifest.json");
    ensure!(
        record(&manifest, sources)? == evidence["cache_manifest_sha256"],
        "training label cache manifest changed"
    );
    let cache = burn_gekko_data::view_targets::Cache::load(
        root,
        Path::new(config["dataset"].as_str().context("training dataset")?),
    )?;
    ensure!(
        cache.manifest.dataset_id == *dataset_id,
        "training label dataset differs"
    );
    ensure!(
        record(&root.join("targets.bin"), sources)? == cache.manifest.targets_sha256,
        "training target bytes changed"
    );
    Ok(evidence)
}

pub fn build(
    e: &Experiment,
    out: &Path,
    sources: &mut Vec<Source>,
) -> Result<Vec<CorrespondenceFigure>> {
    let Some(input) = &e.view_geometry else {
        return Ok(Vec::new());
    };
    ensure!(
        record(&input.path, sources)? == input.sha256,
        "view geometry audit changed"
    );
    let data: Value = serde_json::from_slice(&fs::read(&input.path)?)?;
    ensure!(
        data["task"] == "renderer_viewpoint_correspondence"
            && data["checkpoint"]["model_sha256"] == e.checkpoint_sha256,
        "mixed viewpoint checkpoint"
    );
    let config_path = input.path.parent().unwrap().join("config.toml");
    record(&config_path, sources)?;
    let config: Value = burn_gekko_data::read_config(&config_path)?;
    ensure!(
        config["weights"]["model_sha256"] == e.checkpoint_sha256,
        "mixed viewpoint configuration"
    );
    let dataset = Path::new(config["dataset"].as_str().context("viewpoint dataset")?);
    record(&dataset.join("manifest.json"), sources)?;
    let manifest = burn_gekko_data::read_dataset_manifest(dataset)?;
    ensure!(
        manifest.dataset_id == data["dataset_id"],
        "viewpoint dataset differs"
    );
    let mut figures = Vec::new();
    let rows = data["records"].as_array().context("viewpoint records")?;
    for row in rows.iter().filter(|r| {
        r["sample"].as_u64().is_some_and(|i| i < 4)
            && r["direction"] == 0
            && r["method"] == "spatial_pair_local"
    }) {
        let seed = row["room_seed"].as_u64().context("room seed")?;
        let entry = manifest
            .scenes
            .iter()
            .find(|s| s.seed == seed && s.split == burn_gekko_data::Split::Validation)
            .context("viewpoint room missing")?;
        record(&dataset.join("raw").join(&entry.file), sources)?;
        let scene = burn_gekko_data::load_dataset_rgb(dataset, &manifest, entry)?;
        let (width, height) = (scene.width as u32, scene.height as u32);
        let mut image = RgbImage::from_pixel(width * 2 + 16, height, Rgb([235, 240, 244]));
        for (v, offset) in [(0, 0), (1, width + 16)] {
            let tile = RgbImage::from_raw(
                width,
                height,
                scene.views[v]
                    .iter()
                    .map(|x| (x.clamp(0., 1.) * 255.).round() as u8)
                    .collect(),
            )
            .unwrap();
            image::imageops::replace(&mut image, &tile, offset as i64, 0);
        }
        let points = row["score"]["points"]
            .as_array()
            .context("viewpoint points")?;
        ensure!(!points.is_empty(), "no drawable viewpoint points");
        let count = points.len().min(8);
        for i in 0..count {
            let p = &points[i * (points.len() - 1) / (count - 1).max(1)];
            let query = p[0].as_u64().context("query")?;
            let w = width as u64 / 16;
            let a = [(query % w) as f64 * 16. + 8., (query / w) as f64 * 16. + 8.];
            let xy = |p: &Value| -> Result<[f64; 2]> {
                Ok([
                    p[0].as_f64().context("x")? + width as f64 + 16.,
                    p[1].as_f64().context("y")?,
                ])
            };
            let (prediction, truth) = (xy(&p[1])?, xy(&p[2])?);
            line(&mut image, a, truth, Rgb([58, 155, 199]));
            line(&mut image, truth, prediction, Rgb([255, 187, 39]));
            point(&mut image, a[0], a[1], Rgb([87, 195, 251]));
            point(&mut image, truth[0], truth[1], Rgb([44, 242, 162]));
            point(
                &mut image,
                prediction[0],
                prediction[1],
                Rgb([255, 111, 61]),
            );
        }
        let file = format!("media/room-view-{}.png", row["sample"]);
        image.save(out.join(&file))?;
        figures.push(CorrespondenceFigure {file,title:format!("Actual camera views / room {seed}"),caption:format!("View 0 left, view 1 right. Blue lines join source queries to renderer-projected green targets; orange marks RGB-model predictions and yellow shows error. Mean endpoint error {:.2} input pixels over all known visible queries. First four validation rooms; eight evenly spaced valid queries. Synthetic development data, not external transfer evidence.",row["score"]["mean_epe"].as_f64().context("viewpoint AEPE")?)});
    }
    Ok(figures)
}
