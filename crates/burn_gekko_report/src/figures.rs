//! Deterministic visualizations of native float exports. No fitted RGB decoder is implied.
use crate::{
    artifact::{Source, record},
    experiment::Experiment,
};
use anyhow::{Context, Result, ensure};
use image::{
    Rgb, RgbImage,
    imageops::{FilterType, resize},
};
use serde::Serialize;
use serde_json::Value;
use std::{fs, path::Path};

pub fn floats(path: &Path, sources: &mut Vec<Source>) -> Result<Vec<f32>> {
    record(path, sources)?;
    let bytes = fs::read(path)?;
    ensure!(bytes.len().is_multiple_of(4), "truncated float array");
    let v = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect::<Vec<_>>();
    ensure!(v.iter().all(|v| v.is_finite()), "nonfinite sample array");
    Ok(v)
}
fn rgb(v: &[f32], w: usize, h: usize) -> Result<RgbImage> {
    ensure!(v.len() == w * h * 3, "RGB shape mismatch");
    Ok(RgbImage::from_raw(
        w as u32,
        h as u32,
        v.iter()
            .map(|x| (x.clamp(0., 1.) * 255.).round() as u8)
            .collect(),
    )
    .unwrap())
}
fn heat(x: f64) -> Rgb<u8> {
    let t = x.clamp(0., 1.);
    let stops = [
        [17., 30., 63.],
        [29., 115., 155.],
        [40., 198., 157.],
        [241., 222., 78.],
        [232., 77., 64.],
    ];
    let p = t * 4.;
    let i = (p.floor() as usize).min(3);
    let a = p - i as f64;
    Rgb(std::array::from_fn(|c| {
        (stops[i][c] * (1. - a) + stops[i + 1][c] * a).round() as u8
    }))
}
fn signed_gain(x: f64) -> Rgb<u8> {
    let end = if x < 0. {
        [207., 65., 64.]
    } else {
        [22., 141., 120.]
    };
    let t = x.abs().min(1.);
    Rgb(std::array::from_fn(|i| {
        (245. * (1. - t) + end[i] * t).round() as u8
    }))
}
#[derive(Debug, Serialize)]
pub struct Sample {
    pub id: String,
    pub room_seed: u64,
    pub target_view: usize,
    pub mse: f64,
    pub cosine: f64,
    pub feature_snr_db: Option<f64>,
    pub panels: Vec<(String, String)>,
    pub contact_sheet: String,
}
#[derive(Serialize)]
struct Projection {
    mean: Vec<f64>,
    axes: Vec<Vec<f64>>,
    limits: Vec<[f64; 2]>,
    fit: String,
}
impl Projection {
    fn fit(truth: &[Vec<f32>], d: usize) -> Result<Self> {
        let points = truth.iter().map(|v| v.len() / d).sum::<usize>();
        ensure!(d >= 3 && points > 3, "insufficient PCA observations");
        let mut mean = vec![0.; d];
        for v in truth {
            for row in v.chunks_exact(d) {
                for j in 0..d {
                    mean[j] += row[j] as f64 / points as f64;
                }
            }
        }
        let mut axes: Vec<Vec<f64>> = Vec::new();
        for k in 0..3 {
            let mut a = (0..d)
                .map(|j| ((j * 17 + k * 37 + 1) as f64).sin())
                .collect::<Vec<_>>();
            for _ in 0..48 {
                let mut next = vec![0.; d];
                for v in truth {
                    for row in v.chunks_exact(d) {
                        let dot = (0..d)
                            .map(|j| (row[j] as f64 - mean[j]) * a[j])
                            .sum::<f64>();
                        for j in 0..d {
                            next[j] += (row[j] as f64 - mean[j]) * dot;
                        }
                    }
                }
                for previous in &axes {
                    let dot = next.iter().zip(previous).map(|(a, b)| a * b).sum::<f64>();
                    for j in 0..d {
                        next[j] -= dot * previous[j];
                    }
                }
                let norm = next.iter().map(|x| x * x).sum::<f64>().sqrt();
                ensure!(norm > 1e-10, "degenerate latent projection");
                for x in &mut next {
                    *x /= norm;
                }
                a = next;
            }
            let max = (0..d)
                .max_by(|&a0, &b| a[a0].abs().total_cmp(&a[b].abs()))
                .unwrap();
            if a[max] < 0. {
                for x in &mut a {
                    *x = -*x;
                }
            }
            axes.push(a);
        }
        let mut limits = Vec::new();
        for axis in &axes {
            let mut values = truth
                .iter()
                .flat_map(|v| v.chunks_exact(d))
                .map(|row| {
                    (0..d)
                        .map(|j| (row[j] as f64 - mean[j]) * axis[j])
                        .sum::<f64>()
                })
                .collect::<Vec<_>>();
            values.sort_by(f64::total_cmp);
            limits.push([
                values[(0.02 * (points - 1) as f64) as usize],
                values[(0.98 * (points - 1) as f64) as usize],
            ]);
        }
        Ok(Self {mean,axes,limits,fit:"Three PCA directions (48 deterministic power iterations each), fit on all displayed teacher tokens only. Shared teacher 2nd-98th percentile color limits for every prediction. Visualization only; no inference or metric fitting.".into()})
    }
    fn image(&self, v: &[f32], grid: [usize; 2]) -> RgbImage {
        let d = self.mean.len();
        let mut image = RgbImage::new(grid[1] as u32, grid[0] as u32);
        for (i, row) in v.chunks_exact(d).enumerate() {
            let color = std::array::from_fn(|c| {
                let x = (0..d)
                    .map(|j| (row[j] as f64 - self.mean[j]) * self.axes[c][j])
                    .sum::<f64>();
                let [low, high] = self.limits[c];
                (((x - low) / (high - low).max(1e-12)).clamp(0., 1.) * 255.).round() as u8
            });
            image.put_pixel((i % grid[1]) as u32, (i / grid[1]) as u32, Rgb(color));
        }
        resize(&image, 256, 256, FilterType::Nearest)
    }
}
pub fn build_samples(e: &Experiment, out: &Path, sources: &mut Vec<Source>) -> Result<Vec<Sample>> {
    let mut dirs = fs::read_dir(&e.latent.directory)?
        .map(|r| r.map(|r| r.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    dirs.retain(|p| {
        p.is_dir()
            && p.file_name()
                .is_some_and(|s| s.to_string_lossy().starts_with("room-"))
    });
    dirs.sort();
    ensure!(
        !dirs.is_empty(),
        "no sample exports; enable export_rooms during evaluation"
    );
    let n = e.sample_count.min(dirs.len());
    let chosen = (0..n)
        .map(|i| {
            if n == 1 {
                0
            } else {
                i * (dirs.len() - 1) / (n - 1)
            }
        })
        .collect::<Vec<_>>();
    let selected = chosen.iter().map(|&i| dirs[i].clone()).collect::<Vec<_>>();
    let mut truth = Vec::new();
    for dir in &selected {
        truth.push(floats(&dir.join("target-latent.f32"), sources)?);
    }
    let metadata: Value = serde_json::from_slice(&fs::read(selected[0].join("metadata.json"))?)?;
    let d = metadata["latent_shape"][1]
        .as_u64()
        .context("latent channels")? as usize;
    let projection = Projection::fit(&truth, d)?;
    burn_gekko_data::write_json(&out.join("media/latent-projection.json"), &projection)?;
    let metrics: Value =
        serde_json::from_slice(&fs::read(e.latent.directory.join("metrics.json"))?)?;
    let mut samples = Vec::new();
    for (index, dir) in selected.iter().enumerate() {
        record(&dir.join("metadata.json"), sources)?;
        let meta: Value = serde_json::from_slice(&fs::read(dir.join("metadata.json"))?)?;
        let seed = meta["room_seed"].as_u64().context("room seed")?;
        let view = meta["target_view"].as_u64().context("target view")? as usize;
        let h = meta["rgb_shape"][0].as_u64().context("height")? as usize;
        let w = meta["rgb_shape"][1].as_u64().context("width")? as usize;
        let grid = [
            meta["grid"][0].as_u64().context("grid h")? as usize,
            meta["grid"][1].as_u64().context("grid w")? as usize,
        ];
        ensure!(
            meta["latent_shape"][1].as_u64() == Some(d as u64)
                && grid[0] * grid[1] * d == truth[index].len(),
            "inconsistent latent shape"
        );
        let hidden = meta["hidden_tokens"]
            .as_array()
            .context("hidden tokens")?
            .iter()
            .map(|x| x.as_u64().map(|x| x as usize).context("token id"))
            .collect::<Result<Vec<_>>>()?;
        let pred = floats(&dir.join("cross-latent.f32"), sources)?;
        let check = burn_gekko_eval::metrics::completion(&pred, &truth[index], d, &hidden)?;
        let row = metrics["rows"]
            .as_array()
            .context("metric rows")?
            .iter()
            .find(|r| r["room_seed"] == seed && r["target_view"] == view)
            .context("sample absent from evaluation population")?;
        ensure!(
            (check.mse - row["cross_mse"].as_f64().context("sample MSE")?).abs() < 2e-6,
            "exported arrays disagree with measured MSE"
        );
        let target = rgb(&floats(&dir.join("target-rgb.f32"), sources)?, w, h)?;
        let mut masked = target.clone();
        for &id in &hidden {
            for y in id / grid[1] * 16..(id / grid[1] + 1) * 16 {
                for x in id % grid[1] * 16..(id % grid[1] + 1) * 16 {
                    masked.put_pixel(x as u32, y as u32, Rgb([23, 29, 39]));
                }
            }
        }
        let reference = |i, sources: &mut Vec<Source>| -> Result<RgbImage> {
            rgb(
                &floats(&dir.join(format!("reference-{i}-rgb.f32")), sources)?,
                w,
                h,
            )
        };
        let mut error = RgbImage::new(grid[1] as u32, grid[0] as u32);
        let mut visibility = error.clone();
        let labels = fs::read(dir.join("visibility.u8"))?;
        record(&dir.join("visibility.u8"), sources)?;
        ensure!(
            labels.len() == grid[0] * grid[1],
            "visibility shape mismatch"
        );
        for (i, &label) in labels.iter().enumerate() {
            let mse = (0..d)
                .map(|j| (pred[i * d + j] as f64 - truth[index][i * d + j] as f64).powi(2))
                .sum::<f64>()
                / d as f64;
            error.put_pixel(
                (i % grid[1]) as u32,
                (i / grid[1]) as u32,
                if hidden.contains(&i) {
                    heat(mse / 1.0)
                } else {
                    Rgb([160, 160, 160])
                },
            );
            visibility.put_pixel(
                (i % grid[1]) as u32,
                (i / grid[1]) as u32,
                match label {
                    1 => Rgb([48, 195, 158]),
                    0 => Rgb([207, 95, 81]),
                    _ => Rgb([160, 160, 160]),
                },
            );
        }
        let mut images = vec![
            (
                "Sparse target input",
                resize(&masked, 256, 256, FilterType::Triangle),
            ),
            (
                "Reference view 1",
                resize(&reference(1, sources)?, 256, 256, FilterType::Triangle),
            ),
        ];
        if dir.join("reference-2-rgb.f32").exists() {
            images.push((
                "Reference view 2",
                resize(&reference(2, sources)?, 256, 256, FilterType::Triangle),
            ));
        } else {
            images.push((
                "No second reference",
                RgbImage::from_pixel(256, 256, Rgb([23, 29, 39])),
            ));
        }
        images.extend([
            (
                "Target RGB (evaluation only)",
                resize(&target, 256, 256, FilterType::Triangle),
            ),
            (
                "Teacher latent (evaluation only)",
                projection.image(&truth[index], grid),
            ),
            ("Predicted latent", projection.image(&pred, grid)),
            (
                "Hidden-token MSE (0 to 1)",
                resize(&error, 256, 256, FilterType::Nearest),
            ),
            (
                "Co-visibility truth",
                resize(&visibility, 256, 256, FilterType::Nearest),
            ),
        ]);
        if dir.join("ri.f32").exists() {
            let mono = floats(&dir.join("monocular-latent.f32"), sources)?;
            ensure!(mono.len() == pred.len(), "monocular latent shape mismatch");
            images.push((
                "Monocular latent (same model)",
                projection.image(&mono, grid),
            ));
            for (file, title, scale) in [
                ("ri.f32", "RI score (full target; 0 to 1)", 1.),
                ("gain.f32", "Reference benefit (-1 to +1)", 1.),
                (
                    "visibility-fraction.f32",
                    "Geometric visibility fraction",
                    1.,
                ),
            ] {
                let values = floats(&dir.join(file), sources)?;
                ensure!(
                    values.len() == grid[0] * grid[1],
                    "diagnostic map shape mismatch"
                );
                let mut map = RgbImage::new(grid[1] as u32, grid[0] as u32);
                for (i, &value) in values.iter().enumerate() {
                    map.put_pixel(
                        (i % grid[1]) as u32,
                        (i / grid[1]) as u32,
                        if hidden.contains(&i)
                            && (file != "visibility-fraction.f32" || labels[i] != 255)
                        {
                            if file == "gain.f32" {
                                let mse = |v: &[f32]| {
                                    (0..d)
                                        .map(|j| {
                                            (v[i * d + j] as f64 - truth[index][i * d + j] as f64)
                                                .powi(2)
                                        })
                                        .sum::<f64>()
                                        / d as f64
                                };
                                let mono_error = mse(&mono);
                                // Preserve harmful reference effects: the training RI target is clipped at zero.
                                signed_gain((mono_error - mse(&pred)) / mono_error.max(1e-6))
                            } else {
                                heat(value as f64 / scale)
                            }
                        } else {
                            Rgb([160, 160, 160])
                        },
                    );
                }
                images.push((title, resize(&map, 256, 256, FilterType::Nearest)));
            }
        }
        let id = format!("room-{seed}-view-{view}");
        let mut panels = Vec::new();
        let rows = images.len().div_ceil(4) as u32;
        let mut sheet = RgbImage::from_pixel(
            4 * 256 + 3 * 8,
            rows * 256 + (rows - 1) * 8,
            Rgb([245, 247, 250]),
        );
        for (i, (title, img)) in images.iter().enumerate() {
            let file = format!("media/{id}-{i}.png");
            img.save(out.join(&file))?;
            panels.push((title.to_string(), file));
            image::imageops::replace(&mut sheet, img, (i % 4 * 264) as i64, (i / 4 * 264) as i64);
        }
        let contact_sheet = format!("media/{id}-sheet.png");
        sheet.save(out.join(&contact_sheet))?;
        samples.push(Sample {
            id,
            room_seed: seed,
            target_view: view,
            mse: check.mse,
            cosine: check.cosine.unwrap_or(0.),
            feature_snr_db: check.signal_to_error_db,
            panels,
            contact_sheet,
        });
    }
    ensure!(
        samples
            .iter()
            .all(|s| s.panels.len() == samples[0].panels.len()),
        "inconsistent sample export panels"
    );
    burn_gekko_data::write_json(&out.join("media/samples.json"), &samples)?;
    Ok(samples)
}
