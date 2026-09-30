use crate::{
    artifact::{Report, Source},
    figures::floats,
};
use anyhow::{Context, Result, ensure};
use image::{Rgb, RgbImage};
use serde::Serialize;
use serde_json::Value;
use std::path::Path;
#[derive(Debug, Serialize)]
pub struct CorrespondenceFigure {
    pub title: String,
    pub file: String,
    pub caption: String,
}
pub(crate) fn point(image: &mut RgbImage, x: f64, y: f64, color: Rgb<u8>) {
    for dx in -3..=3 {
        for dy in -3..=3 {
            if dx * dx + dy * dy > 9 {
                continue;
            }
            let (a, b) = (x.round() as i64 + dx, y.round() as i64 + dy);
            if a >= 0 && b >= 0 && a < image.width() as i64 && b < image.height() as i64 {
                image.put_pixel(a as u32, b as u32, color);
            }
        }
    }
}
pub(crate) fn line(image: &mut RgbImage, a: [f64; 2], b: [f64; 2], color: Rgb<u8>) {
    let steps = (a[0] - b[0])
        .abs()
        .max((a[1] - b[1]).abs())
        .ceil()
        .clamp(1., 2048.) as usize;
    for i in 0..=steps {
        let t = i as f64 / steps as f64;
        let x = (a[0] + t * (b[0] - a[0])).round() as i64;
        let y = (a[1] + t * (b[1] - a[1])).round() as i64;
        if x >= 0 && y >= 0 && x < image.width() as i64 && y < image.height() as i64 {
            image.put_pixel(x as u32, y as u32, color);
        }
    }
}
pub fn build(
    r: &Report,
    out: &Path,
    sources: &mut Vec<Source>,
) -> Result<Vec<CorrespondenceFigure>> {
    let mut figures = Vec::new();
    for benchmark in &r.benchmarks {
        let name = benchmark["benchmark"].as_str().context("benchmark name")?;
        let examples = benchmark["examples"]
            .as_array()
            .context("benchmark scoring must export deterministic visual examples")?;
        for (i, example) in examples.iter().enumerate() {
            let mut image = RgbImage::from_pixel(528, 256, Rgb([235, 240, 244]));
            for (role, offset) in [("target", 0), ("reference", 272)] {
                let p = Path::new(
                    example["images"][role]["file"]
                        .as_str()
                        .context("image path")?,
                );
                ensure!(
                    burn_gekko_data::sha256_file(p)?
                        == example["images"][role]["sha256"]
                            .as_str()
                            .context("image checksum")?,
                    "benchmark image changed"
                );
                let v = floats(p, sources)?;
                ensure!(v.len() == 256 * 256 * 3, "benchmark image shape");
                let tile = RgbImage::from_raw(
                    256,
                    256,
                    v.iter()
                        .map(|x| (x.clamp(0., 1.) * 255.).round() as u8)
                        .collect(),
                )
                .unwrap();
                image::imageops::replace(&mut image, &tile, offset, 0);
            }
            let h = example["metric_hw"][0].as_f64().context("metric height")?;
            let w = example["metric_hw"][1].as_f64().context("metric width")?;
            let xy = |v: &Value, offset: f64| -> [f64; 2] {
                [
                    offset + v[0].as_f64().unwrap() / w * 256.,
                    v[1].as_f64().unwrap() / h * 256.,
                ]
            };
            for v in example["vectors"]
                .as_array()
                .context("visual correspondences")?
            {
                let a = xy(&v["target"], 0.);
                let b = xy(&v["expected_reference"], 272.);
                let p = xy(&v["predicted_reference"], 272.);
                line(&mut image, a, b, Rgb([58, 155, 199]));
                line(&mut image, b, p, Rgb([255, 187, 39]));
                point(&mut image, a[0], a[1], Rgb([87, 195, 251]));
                point(&mut image, b[0], b[1], Rgb([44, 242, 162]));
                point(&mut image, p[0], p[1], Rgb([255, 111, 61]));
            }
            let file = format!("media/{name}-matches-{i}.png");
            image.save(out.join(&file))?;
            figures.push(CorrespondenceFigure {title:format!("{} / {}",name,example["sample"].as_str().context("sample")?),file,
                caption:format!("Target left; reference right. Blue: target/true-match connection. Green: ground-truth reference location. Orange: model match. Yellow: matching error. {} readout; pair AEPE {:.2} protocol pixels. Eight labels selected uniformly, not by error. {}.",crate::display::readout(example["method"].as_str().context("method")?),example["metrics"]["aepe"].as_f64().context("AEPE")?,example["selection"].as_str().context("selection")?)});
        }
    }
    Ok(figures)
}
