//! Annotated known-transform examples from the same pinned training experiment.
use crate::{
    artifact::{Source, record},
    correspondence::{CorrespondenceFigure, line, point},
    experiment::Experiment,
    figures::floats,
};
use anyhow::{Context, Result, ensure};
use image::{Rgb, RgbImage};
use serde_json::Value;
use std::{fs, path::Path};

pub fn build(
    e: &Experiment,
    out: &Path,
    sources: &mut Vec<Source>,
) -> Result<Vec<CorrespondenceFigure>> {
    let Some(input) = &e.equivariance else {
        return Ok(Vec::new());
    };
    ensure!(
        record(&input.path, sources)? == input.sha256,
        "warp diagnostic changed"
    );
    let data: Value = serde_json::from_slice(&fs::read(&input.path)?)?;
    ensure!(
        data["checkpoint"]["model_sha256"] == e.checkpoint_sha256,
        "mixed warp checkpoint"
    );
    let width = data["width"].as_u64().context("warp width")? as u32;
    let height = data["height"].as_u64().context("warp height")? as u32;
    ensure!(
        (16..=512).contains(&width) && (16..=512).contains(&height),
        "invalid warp image size"
    );
    let mut figures = Vec::new();
    for sample in data["samples"].as_array().context("warp samples")? {
        let id = sample["sample"].as_u64().context("sample index")?;
        let mut image = RgbImage::from_pixel(width * 2 + 16, height, Rgb([235, 240, 244]));
        for (role, offset) in [("source", 0), ("warped", width + 16)] {
            let path = Path::new(sample[role]["file"].as_str().context("warp image path")?);
            ensure!(
                burn_gekko_data::sha256_file(path)?
                    == sample[role]["sha256"].as_str().context("warp RGB hash")?,
                "warp image changed"
            );
            let values = floats(path, sources)?;
            ensure!(
                values.len() == width as usize * height as usize * 3,
                "warp image dimensions"
            );
            let tile = RgbImage::from_raw(
                width,
                height,
                values
                    .iter()
                    .map(|x| (x.clamp(0., 1.) * 255.).round() as u8)
                    .collect(),
            )
            .unwrap();
            image::imageops::replace(&mut image, &tile, offset as i64, 0);
        }
        let method = if data["methods"].get("spatial_pair_local").is_some() {
            "spatial_pair_local"
        } else {
            "spatial_pair"
        };
        let row = data["records"]
            .as_array()
            .context("warp records")?
            .iter()
            .find(|r| r["sample"] == id && r["direction"] == 0 && r["method"] == method)
            .context("missing pair descriptor predictions")?;
        let points = row["score"]["points"].as_array().context("warp points")?;
        ensure!(!points.is_empty(), "no drawable correspondences");
        let count = points.len().min(8);
        for i in 0..count {
            let p = &points[i * (points.len() - 1) / (count - 1).max(1)];
            let query = p[0].as_u64().context("warp query index")?;
            let grid_width = width as u64 / 16;
            let a = [
                (query % grid_width) as f64 * 16. + 8.,
                (query / grid_width) as f64 * 16. + 8.,
            ];
            let xy = |p: &Value| -> Result<[f64; 2]> {
                Ok([
                    p[0].as_f64().context("point x")? + width as f64 + 16.,
                    p[1].as_f64().context("point y")?,
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
        let file = format!("media/known-warp-{id}.png");
        image.save(out.join(&file))?;
        figures.push(CorrespondenceFigure {title:format!("Known image transform / room {}",sample["room_seed"]),file,
            caption:format!("Original RGB left; transformed RGB right. Blue lines connect source queries to known green target locations; orange marks model predictions, with yellow error segments. {}, mean endpoint error {:.2} input pixels. First four validation rooms, eight evenly spaced valid queries per image. This is an augmentation diagnostic, not a real viewpoint benchmark.",crate::display::readout(method),row["score"]["mean_epe"].as_f64().context("warp AEPE")?) });
    }
    Ok(figures)
}
