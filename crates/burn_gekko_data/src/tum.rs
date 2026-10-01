//! Deterministic TUM Freiburg 3 preparation. Geometry stays outside the RGB manifest.
use crate::{real_views::*, sha256_file, write_config, write_json};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::{Path, PathBuf},
};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TumSequence {
    pub name: String,
    pub directory: PathBuf,
    pub archive: PathBuf,
    pub archive_sha256: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TumConfig {
    pub sequences: Vec<TumSequence>,
    pub output: PathBuf,
    pub anchors_per_sequence: usize,
    pub frame_intervals: Vec<usize>,
    pub max_association_seconds: f64,
    pub evaluation_use: String,
    #[serde(default = "default_image_size")]
    pub image_size: usize,
}
#[derive(Clone)]
struct Pose {
    time: f64,
    position: [f64; 3],
    rotation: [[f64; 3]; 3],
}

fn lines(path: &Path) -> Result<Vec<Vec<String>>> {
    Ok(fs::read_to_string(path)?
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty() && !s.starts_with('#'))
        .map(|s| s.split_whitespace().map(str::to_owned).collect())
        .collect())
}
fn rotation(q: [f64; 4]) -> Result<[[f64; 3]; 3]> {
    let norm = q.iter().map(|v| v * v).sum::<f64>().sqrt();
    ensure!(
        norm.is_finite() && (norm - 1.).abs() < 1e-3,
        "invalid pose quaternion"
    );
    let [x, y, z, w] = q.map(|v| v / norm);
    Ok([
        [
            1. - 2. * (y * y + z * z),
            2. * (x * y - z * w),
            2. * (x * z + y * w),
        ],
        [
            2. * (x * y + z * w),
            1. - 2. * (x * x + z * z),
            2. * (y * z - x * w),
        ],
        [
            2. * (x * z - y * w),
            2. * (y * z + x * w),
            1. - 2. * (x * x + y * y),
        ],
    ])
}
fn relative(target: &Pose, reference: &Pose) -> ([[f64; 3]; 3], [f64; 3]) {
    let r = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (0..3)
                .map(|k| reference.rotation[k][i] * target.rotation[k][j])
                .sum()
        })
    });
    let t = std::array::from_fn(|i| {
        (0..3)
            .map(|k| reference.rotation[k][i] * (target.position[k] - reference.position[k]))
            .sum()
    });
    (r, t)
}
fn nearest(poses: &[Pose], time: f64, tolerance: f64) -> Option<&Pose> {
    poses
        .iter()
        .min_by(|a, b| (a.time - time).abs().total_cmp(&(b.time - time).abs()))
        .filter(|p| (p.time - time).abs() <= tolerance)
}
fn inside_data(p: &Path) -> Result<()> {
    ensure!(
        fs::canonicalize(p)?.starts_with(fs::canonicalize(".data")?),
        "input outside .data"
    );
    Ok(())
}
pub fn prepare(c: &TumConfig) -> Result<()> {
    descriptor_grid(c.image_size)?;
    ensure!(!c.output.exists(), "preserve existing external dataset");
    inside_data(
        c.output
            .ancestors()
            .find(|p| p.exists())
            .context("output parent")?,
    )?;
    ensure!(
        c.anchors_per_sequence >= 2
            && !c.frame_intervals.is_empty()
            && c.frame_intervals.iter().all(|x| *x > 0)
            && c.frame_intervals.iter().collect::<BTreeSet<_>>().len() == c.frame_intervals.len()
            && c.max_association_seconds.is_finite()
            && c.max_association_seconds > 0.
            && matches!(c.evaluation_use.as_str(), "held_out" | "development"),
        "invalid pair protocol"
    );
    ensure!(
        !c.sequences.is_empty()
            && c.sequences
                .iter()
                .map(|s| &s.name)
                .collect::<BTreeSet<_>>()
                .len()
                == c.sequences.len(),
        "invalid sequences"
    );
    let max_interval = *c.frame_intervals.iter().max().unwrap();
    let mut images = PairImages {
        schema: 1,
        image_size: c.image_size,
        evaluation_use: c.evaluation_use.clone(),
        images: BTreeMap::new(),
        pairs: Vec::new(),
    };
    let mut labels=PairLabels {schema:1,image_manifest_sha256:String::new(),intrinsics:[535.4,539.2,320.1,247.6],coordinate_frame:"X_reference = R * X_target + t; OpenCV optical axes; ground truth camera-to-world converted before scoring".into(),rows:Vec::new()};
    let mut sources = BTreeMap::new();
    let mut excluded = Vec::new();
    let mut counts = Vec::new();
    fs::create_dir_all(c.output.join("rgb"))?;
    write_config(&c.output.join("config.toml"), c)?;
    for seq in &c.sequences {
        ensure!(
            matches!(
                seq.name.as_str(),
                "long_office_household" | "structure_texture_far" | "structure_texture_near"
            ),
            "only registered undistorted Freiburg 3 sequences supported"
        );
        inside_data(&seq.directory)?;
        inside_data(&seq.archive)?;
        ensure!(
            sha256_file(&seq.archive)? == seq.archive_sha256,
            "archive checksum mismatch"
        );
        sources.insert(seq.archive.clone(), seq.archive_sha256.clone());
        let rgb_path = seq.directory.join("rgb.txt");
        let gt_path = seq.directory.join("groundtruth.txt");
        sources.insert(rgb_path.clone(), sha256_file(&rgb_path)?);
        sources.insert(gt_path.clone(), sha256_file(&gt_path)?);
        let mut rgb = Vec::new();
        for row in lines(&rgb_path)? {
            ensure!(row.len() == 2, "invalid RGB index");
            let t = row[0].parse::<f64>()?;
            ensure!(t.is_finite(), "nonfinite RGB timestamp");
            let file = seq.directory.join(&row[1]);
            ensure!(
                fs::canonicalize(&file)?.starts_with(fs::canonicalize(&seq.directory)?),
                "RGB path escapes sequence"
            );
            rgb.push((t, file));
        }
        ensure!(
            rgb.windows(2).all(|p| p[0].0 < p[1].0),
            "RGB timestamps must be unique and sorted"
        );
        ensure!(
            rgb.len() > max_interval + c.anchors_per_sequence,
            "sequence too short"
        );
        let mut poses = Vec::new();
        for row in lines(&gt_path)? {
            ensure!(row.len() == 8, "invalid ground-truth row");
            let v = row
                .iter()
                .map(|s| s.parse::<f64>())
                .collect::<std::result::Result<Vec<_>, _>>()?;
            ensure!(v.iter().all(|x| x.is_finite()), "nonfinite camera label");
            poses.push(Pose {
                time: v[0],
                position: [v[1], v[2], v[3]],
                rotation: rotation([v[4], v[5], v[6], v[7]])?,
            });
        }
        ensure!(
            !poses.is_empty() && poses.windows(2).all(|p| p[0].time < p[1].time),
            "invalid pose timestamps"
        );
        let before = images.pairs.len();
        for a in 0..c.anchors_per_sequence {
            let i = a * (rgb.len() - 1 - max_interval) / (c.anchors_per_sequence - 1);
            for &interval in &c.frame_intervals {
                let j = i + interval;
                let id = format!("{}-{i:06}-{j:06}", seq.name);
                let (Some(pa), Some(pb)) = (
                    nearest(&poses, rgb[i].0, c.max_association_seconds),
                    nearest(&poses, rgb[j].0, c.max_association_seconds),
                ) else {
                    excluded.push(serde_json::json!({"id":id,"reason":"no ground-truth association within tolerance"}));
                    continue;
                };
                let keys = [
                    format!("{}-{i:06}", seq.name),
                    format!("{}-{j:06}", seq.name),
                ];
                for (&index, key) in [i, j].iter().zip(&keys) {
                    if images.images.contains_key(key) {
                        continue;
                    }
                    let path = &rgb[index].1;
                    let image = image::open(path)?.to_rgb8();
                    ensure!(
                        image.dimensions() == (640, 480),
                        "expected original Freiburg RGB 640x480"
                    );
                    let small = image::imageops::resize(
                        &image,
                        c.image_size as u32,
                        c.image_size as u32,
                        image::imageops::FilterType::Triangle,
                    );
                    let file = c.output.join("rgb").join(format!("{key}.f32"));
                    let mut f = std::io::BufWriter::new(
                        fs::OpenOptions::new()
                            .write(true)
                            .create_new(true)
                            .open(&file)?,
                    );
                    for &v in small.as_raw() {
                        f.write_all(&(v as f32 / 255.).to_le_bytes())?;
                    }
                    f.flush()?;
                    images.images.insert(
                        key.clone(),
                        RgbView {
                            sha256: sha256_file(&file)?,
                            file,
                            original_file: path.clone(),
                            original_sha256: sha256_file(path)?,
                            original_hw: [480, 640],
                        },
                    );
                }
                images.pairs.push(ViewPair {
                    id: id.clone(),
                    sequence: seq.name.clone(),
                    interval,
                    target: keys[0].clone(),
                    reference: keys[1].clone(),
                });
                let (rotation, translation) = relative(pa, pb);
                labels.rows.push(PoseLabel {
                    id,
                    rotation,
                    translation,
                    association_seconds: [(pa.time - rgb[i].0).abs(), (pb.time - rgb[j].0).abs()],
                });
            }
        }
        counts.push(serde_json::json!({"sequence":seq.name,"rgb_frames":rgb.len(),"ground_truth_poses":poses.len(),"pairs":images.pairs.len()-before}));
    }
    ensure!(!images.pairs.is_empty(), "no eligible image pairs");
    write_json(&c.output.join("images.json"), &images)?;
    labels.image_manifest_sha256 = sha256_file(&c.output.join("images.json"))?;
    write_json(&c.output.join("labels.json"), &labels)?;
    write_json(
        &c.output.join("preparation.json"),
        &serde_json::json!({"schema":1,"config":c,"sources":sources,"sequences":counts,"excluded_pairs":excluded,"actual_pairs":images.pairs.len(),"intended_pairs":c.sequences.len()*c.anchors_per_sequence*c.frame_intervals.len(),"resize":format!("image crate RGB8 Triangle {}x{}; half-pixel coordinate mapping",c.image_size,c.image_size),"ground_truth_association":"nearest timestamp within configured tolerance; no replacement","source":"https://cvg.cit.tum.de/data/datasets/rgbd-dataset","license":"CC BY 4.0"}),
    )?;
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn camera_to_world_conversion_uses_reference_frame_and_signed_translation() {
        let a = Pose {
            time: 0.,
            position: [1., 0., 0.],
            rotation: rotation([0., 0., 0., 1.]).unwrap(),
        };
        let b = Pose {
            time: 1.,
            position: [0., 1., 0.],
            rotation: rotation([0., 0., 0.5_f64.sqrt(), 0.5_f64.sqrt()]).unwrap(),
        };
        let (r, t) = relative(&a, &b);
        assert!((r[0][1] - 1.).abs() < 1e-12 && (r[1][0] + 1.).abs() < 1e-12);
        assert!((t[0] + 1.).abs() < 1e-12 && (t[1] + 1.).abs() < 1e-12);
        assert!(nearest(std::slice::from_ref(&a), 0.03, 0.02).is_none());
        assert!(nearest(&[a], 0.01, 0.02).is_some());
    }
}
