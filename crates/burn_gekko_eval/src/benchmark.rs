//! ETH3D and HP-240 scoring of one checkpoint with immutable RGB-only exports.
use crate::{
    metrics::{CorrespondenceMetrics, correspondence},
    npy::Npz,
    statistics::bootstrap_mean,
};
use anyhow::{Context, Result, ensure};
use burn_gekko_data::{sha256_file, write_json};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScoreConfig {
    pub benchmark: String,
    pub dataset: PathBuf,
    pub predictions: PathBuf,
    pub provenance: PathBuf,
    pub checkpoint_sha256: String,
    pub images_sha256: String,
    pub geometry_sha256: String,
    pub predictions_sha256: String,
    pub methods: Vec<String>,
    #[serde(default)]
    pub contrasts: Vec<crate::contrasts::ReadoutContrast>,
    #[serde(default)]
    pub visual_method: Option<String>,
    /// Development or held_out: explicitly records whether outcomes informed selection.
    pub evaluation_use: String,
    pub output: PathBuf,
}
#[derive(Debug, Deserialize)]
struct Prediction {
    method: String,
    grid: [usize; 2],
    indices: Vec<usize>,
    #[serde(default)]
    coordinates: Option<Vec<[f64; 2]>>,
    pair: Option<String>,
    scene: Option<String>,
    interval: Option<u64>,
    sequence: Option<String>,
    target: Option<usize>,
    reference: Option<usize>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Row {
    pub sample: String,
    pub cluster: String,
    pub group: String,
    pub method: String,
    #[serde(flatten)]
    pub metrics: CorrespondenceMetrics,
}
fn hash(path: &Path, expected: &str) -> Result<()> {
    ensure!(
        sha256_file(path)? == expected,
        "checksum mismatch: {}",
        path.display()
    );
    Ok(())
}
fn hw(v: &Value) -> Result<[usize; 2]> {
    Ok([
        v[0].as_u64().context("missing height")? as usize,
        v[1].as_u64().context("missing width")? as usize,
    ])
}

/// OpenCV half-pixel bilinear displacement readout with replicated boundaries.
/// Patch flow is first constructed in f32, matching the historical evaluator.
pub fn flow_at(indices: &[usize], grid: usize, xy: [f64; 2], hw: [usize; 2]) -> Result<[f64; 2]> {
    ensure!(
        grid > 0
            && indices.len() == grid * grid
            && hw[0] > 0
            && hw[1] > 0
            && indices.iter().all(|i| *i < grid * grid),
        "invalid flow grid"
    );
    Ok(flow_unchecked(indices, grid, xy, hw))
}
fn flow_unchecked(indices: &[usize], grid: usize, xy: [f64; 2], hw: [usize; 2]) -> [f64; 2] {
    flow_values(grid, xy, hw, |i| {
        [(indices[i] % grid) as f64, (indices[i] / grid) as f64]
    })
}
/// Fractional grid coordinates use the identical half-pixel displacement protocol.
pub fn fractional_flow_at(
    points: &[[f64; 2]],
    grid: usize,
    xy: [f64; 2],
    hw: [usize; 2],
) -> Result<[f64; 2]> {
    ensure!(
        grid > 0
            && points.len() == grid * grid
            && hw.iter().all(|&v| v > 0)
            && points
                .iter()
                .flatten()
                .all(|&v| v.is_finite() && v >= 0. && v <= (grid - 1) as f64),
        "invalid fractional flow grid"
    );
    Ok(flow_values(grid, xy, hw, |i| points[i]))
}
fn flow_values(
    grid: usize,
    xy: [f64; 2],
    hw: [usize; 2],
    point: impl Fn(usize) -> [f64; 2],
) -> [f64; 2] {
    let u = (((xy[0] as f32 + 0.5) * (grid as f32 / hw[1] as f32) - 0.5) as f64)
        .clamp(0., (grid - 1) as f64);
    let v = (((xy[1] as f32 + 0.5) * (grid as f32 / hw[0] as f32) - 0.5) as f64)
        .clamp(0., (grid - 1) as f64);
    let (x, y) = (u.floor() as usize, v.floor() as usize);
    let (xx, yy) = ((x + 1).min(grid - 1), (y + 1).min(grid - 1));
    let (wx, wy) = (u - x as f64, v - y as f64);
    let mut flow = [0.; 2];
    for (a, b, w) in [
        (x, y, (1. - wx) * (1. - wy)),
        (xx, y, wx * (1. - wy)),
        (x, yy, (1. - wx) * wy),
        (xx, yy, wx * wy),
    ] {
        let p = point(b * grid + a);
        flow[0] += ((p[0] - a as f64) * (hw[1] as f64 / grid as f64)) as f32 as f64 * w;
        flow[1] += ((p[1] - b as f64) * (hw[0] as f64 / grid as f64)) as f32 as f64 * w;
    }
    flow
}
fn inverse(m: [[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    let mut c = [[0.; 3]; 3];
    for (i, row) in c.iter_mut().enumerate() {
        for (j, x) in row.iter_mut().enumerate() {
            *x = m[(j + 1) % 3][(i + 1) % 3] * m[(j + 2) % 3][(i + 2) % 3]
                - m[(j + 1) % 3][(i + 2) % 3] * m[(j + 2) % 3][(i + 1) % 3];
        }
    }
    let d = m[0][0] * c[0][0] + m[0][1] * c[1][0] + m[0][2] * c[2][0];
    ensure!(d.abs() > 1e-12, "singular homography");
    for row in &mut c {
        for x in row {
            *x /= d;
        }
    }
    Ok(c)
}
fn hp_truth(
    h: &[f64],
    reference: [usize; 2],
    target: [usize; 2],
) -> Result<Vec<([f64; 2], [f64; 2])>> {
    ensure!(h.len() == 9, "homography must be 3x3");
    let rs = [240. / reference[1] as f64, 240. / reference[0] as f64, 1.];
    let ts = [240. / target[1] as f64, 240. / target[0] as f64, 1.];
    let mut m = [[0.; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] = h[i * 3 + j] * ts[i] / rs[j];
        }
    }
    let m = inverse(m)?;
    let mut out = Vec::new();
    for y in 0..240 {
        for x in 0..240 {
            let q = m.map(|r| r[0] * x as f64 + r[1] * y as f64 + r[2]);
            if q[2].abs() <= 1e-12 {
                continue;
            }
            let xy = [q[0] / q[2], q[1] / q[2]];
            if xy
                .iter()
                .all(|v| v.is_finite() && (0.0..=239.0).contains(v))
            {
                out.push(([x as f64, y as f64], [xy[0] - x as f64, xy[1] - y as f64]));
            }
        }
    }
    Ok(out)
}
fn summary(rows: &[&Row]) -> Value {
    let n = rows.len() as f64;
    json!({"aepe":rows.iter().map(|r|r.metrics.aepe).sum::<f64>()/n,
        "pck1":rows.iter().map(|r|r.metrics.pck1).sum::<f64>()/n,
        "pck3":rows.iter().map(|r|r.metrics.pck3).sum::<f64>()/n,
        "pck5":rows.iter().map(|r|r.metrics.pck5).sum::<f64>()/n,"pairs":rows.len()})
}
pub fn score(c: &ScoreConfig) -> Result<Value> {
    ensure!(
        matches!(c.evaluation_use.as_str(), "development" | "held_out"),
        "explicit evaluation use required"
    );
    ensure!(
        matches!(c.benchmark.as_str(), "eth3d" | "hpatches"),
        "unknown benchmark"
    );
    ensure!(!c.output.exists(), "preserve existing evaluation output");
    ensure!(
        !c.methods.is_empty() && c.methods.iter().collect::<BTreeSet<_>>().len() == c.methods.len(),
        "empty or duplicate readouts"
    );
    let geometry = if c.benchmark == "eth3d" {
        "correspondences.npz"
    } else {
        "homographies.npz"
    };
    hash(&c.dataset.join("images.json"), &c.images_sha256)?;
    hash(&c.dataset.join(geometry), &c.geometry_sha256)?;
    hash(&c.predictions, &c.predictions_sha256)?;
    let provenance: Value = serde_json::from_slice(&fs::read(&c.provenance)?)?;
    ensure!(
        provenance["checkpoint"]["model_sha256"] == c.checkpoint_sha256,
        "prediction checkpoint provenance mismatch"
    );
    ensure!(
        provenance["image_manifest_sha256"] == c.images_sha256,
        "export image manifest mismatch"
    );
    ensure!(
        c.evaluation_use != "held_out" || provenance["evaluation_use"] != "development",
        "development export cannot become held-out evidence"
    );
    let manifest: Value = serde_json::from_slice(&fs::read(c.dataset.join("images.json"))?)?;
    let mut npz = Npz::open(&c.dataset.join(geometry))?;
    let eth = c.benchmark == "eth3d";
    let mut truths = BTreeMap::new();
    let mut meta = BTreeMap::new();
    let mut images = BTreeMap::new();
    if eth {
        for p in manifest["pairs"].as_array().context("missing ETH pairs")? {
            let key = p["id"].as_str().context("missing pair id")?;
            let target = p["target"].as_str().context("missing target index")?;
            let reference = p["reference"].as_str().context("missing reference index")?;
            let size = hw(&manifest["images"][target]["original_hw"])?;
            ensure!(
                size == hw(&manifest["images"][reference]["original_hw"])?,
                "pair dimensions differ"
            );
            let (shape, v) = npz.read(key)?;
            ensure!(
                shape.len() == 2 && shape[1] == 4,
                "invalid sparse truth shape"
            );
            // Rounded target coordinates, duplicates keep last. No geometry reaches inference.
            let mut unique = BTreeMap::new();
            for row in v.as_chunks::<4>().0 {
                let (x, y) = (
                    row[2].round_ties_even() as i64,
                    row[3].round_ties_even() as i64,
                );
                ensure!(
                    x >= 0 && y >= 0 && x < size[1] as i64 && y < size[0] as i64,
                    "sparse truth outside image"
                );
                unique.insert(
                    (y, x),
                    ([x as f64, y as f64], [row[0] - row[2], row[1] - row[3]]),
                );
            }
            truths.insert(key.to_string(), unique.into_values().collect::<Vec<_>>());
            images.insert(key.to_string(),json!({"target":manifest["images"][target],"reference":manifest["images"][reference]}));
            meta.insert(
                key.to_string(),
                (
                    size,
                    p["scene"].as_str().context("scene")?.to_string(),
                    p["interval"].as_u64().context("interval")?.to_string(),
                ),
            );
        }
        ensure!(meta.len() == 3365, "ETH3D requires all 3365 pairs");
    } else {
        for seq in manifest["sequences"]
            .as_array()
            .context("missing HP sequences")?
        {
            let name = seq["name"].as_str().context("sequence name")?;
            for target in 2..=6 {
                let key = format!("{name}_{target}");
                let (shape, v) = npz.read(&key)?;
                ensure!(shape == [3, 3], "invalid homography shape");
                truths.insert(
                    key.clone(),
                    hp_truth(
                        &v,
                        hw(&seq["views"][0]["original_hw"])?,
                        hw(&seq["views"][target - 1]["original_hw"])?,
                    )?,
                );
                images.insert(
                    key.clone(),
                    json!({"target":seq["views"][target-1],"reference":seq["views"][0]}),
                );
                meta.insert(
                    key,
                    (
                        [240, 240],
                        name.to_string(),
                        if name.starts_with("i_") {
                            "illumination"
                        } else {
                            "viewpoint"
                        }
                        .into(),
                    ),
                );
            }
        }
        ensure!(meta.len() == 580, "HPatches requires all 580 pairs");
    }
    let keys = meta
        .keys()
        .filter(|k| eth || meta[*k].2 == "viewpoint")
        .cloned()
        .collect::<Vec<_>>();
    let example_ids = [
        keys[0].clone(),
        keys[(keys.len() - 1) / 2].clone(),
        keys[keys.len() - 1].clone(),
    ];
    let display_method = if let Some(method) = &c.visual_method {
        ensure!(c.methods.contains(method), "visual readout must be scored");
        method
    } else if c.methods.iter().any(|s| s == "conditional_decoder") {
        "conditional_decoder"
    } else {
        &c.methods[0]
    };
    let mut examples = Vec::new();
    let mut rows = Vec::new();
    let mut seen = BTreeSet::new();
    let mut readout_protocols = BTreeMap::new();
    for line in BufReader::new(fs::File::open(&c.predictions)?).lines() {
        let p: Prediction = serde_json::from_str(&line?)?;
        if !c.methods.contains(&p.method) {
            continue;
        }
        ensure!(
            p.grid == [16, 16] && p.indices.len() == 256 && p.indices.iter().all(|x| *x < 256),
            "invalid prediction grid"
        );
        if let Some(points) = &p.coordinates {
            fractional_flow_at(points, 16, [0., 0.], [256, 256])?;
        }
        let kind = if p.coordinates.is_some() {
            "local_3x3_probability_centroid"
        } else {
            "hard_patch_index"
        };
        if let Some(previous) = readout_protocols.insert(p.method.clone(), kind) {
            ensure!(
                previous == kind,
                "mixed coordinate protocols within a readout"
            );
        }
        let flow = |xy, size| {
            p.coordinates.as_ref().map_or_else(
                || flow_unchecked(&p.indices, 16, xy, size),
                |points| flow_values(16, xy, size, |i| points[i]),
            )
        };
        let key = if eth {
            p.pair.clone().context("missing pair")?
        } else {
            ensure!(p.reference == Some(1), "HP reference must be 1");
            format!(
                "{}_{}",
                p.sequence.clone().context("missing sequence")?,
                p.target.context("missing target")?
            )
        };
        ensure!(
            seen.insert((key.clone(), p.method.clone())),
            "duplicate prediction"
        );
        let (size, cluster, group) = meta.get(&key).context("unknown prediction sample")?;
        if eth {
            ensure!(
                p.scene.as_ref() == Some(cluster)
                    && p.interval.map(|x| x.to_string()).as_ref() == Some(group),
                "prediction metadata mismatch"
            );
        }
        let errors = truths[&key]
            .iter()
            .map(|(xy, gt)| {
                let pred = flow(*xy, *size);
                Ok((pred[0] - gt[0]).hypot(pred[1] - gt[1]))
            })
            .collect::<Result<Vec<_>>>()?;
        let metrics = correspondence(&errors)?;
        if example_ids.contains(&key) && p.method == display_method {
            let truth = &truths[&key];
            let count = truth.len().min(8);
            let mut vectors = Vec::new();
            for i in 0..count {
                let j = if count == 1 {
                    0
                } else {
                    i * (truth.len() - 1) / (count - 1)
                };
                let (xy, gt) = truth[j];
                let flow = flow(xy, *size);
                vectors.push(json!({"target":xy,"expected_reference":[xy[0]+gt[0],xy[1]+gt[1]],"predicted_reference":[xy[0]+flow[0],xy[1]+flow[1]],"error_pixels":errors[j]}));
            }
            examples.push(json!({"sample":key,"method":p.method,"metric_hw":size,"metrics":metrics,"images":images[&key],"vectors":vectors,
                "selection":"first/middle/last lexicographic pair in the primary subset; eight evenly spaced valid labels, no quality filtering"}));
        }
        rows.push(Row {
            sample: key,
            cluster: cluster.clone(),
            group: group.clone(),
            method: p.method,
            metrics,
        });
    }
    ensure!(
        rows.len() == meta.len() * c.methods.len(),
        "missing readouts/pairs"
    );
    let mut methods = BTreeMap::new();
    for method in &c.methods {
        let part = rows
            .iter()
            .filter(|r| &r.method == method)
            .collect::<Vec<_>>();
        let mut groups = BTreeMap::new();
        for group in part
            .iter()
            .map(|r| r.group.clone())
            .collect::<BTreeSet<_>>()
        {
            let g = part
                .iter()
                .filter(|r| r.group == group)
                .copied()
                .collect::<Vec<_>>();
            if eth {
                let scenes = g.iter().map(|r| r.cluster.clone()).collect::<BTreeSet<_>>();
                ensure!(scenes.len() == 10, "missing ETH scenes");
                let scores = scenes
                    .iter()
                    .map(|scene| {
                        summary(
                            &g.iter()
                                .filter(|r| &r.cluster == scene)
                                .copied()
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect::<Vec<_>>();
                let mut s = json!({});
                for k in ["aepe", "pck1", "pck3", "pck5"] {
                    s[k] = json!(
                        scores.iter().map(|r| r[k].as_f64().unwrap()).sum::<f64>()
                            / scores.len() as f64
                    );
                }
                let points = g.iter().map(|r| r.metrics.points).sum::<usize>();
                s["point_weighted_pck3"] = json!(
                    g.iter().map(|r| r.metrics.correct3).sum::<usize>() as f64 / points as f64
                );
                groups.insert(group, s);
            } else {
                groups.insert(group, summary(&g));
            }
        }
        let primary = if eth {
            ensure!(
                groups.keys().cloned().collect::<BTreeSet<_>>()
                    == ["3", "5", "7", "9", "11", "13", "15"]
                        .map(str::to_string)
                        .into_iter()
                        .collect(),
                "wrong ETH intervals"
            );
            let mut s = json!({});
            for k in ["aepe", "pck1", "pck3", "pck5", "point_weighted_pck3"] {
                s[k] = json!(groups.values().map(|r| r[k].as_f64().unwrap()).sum::<f64>() / 7.);
            }
            s
        } else {
            groups
                .get("viewpoint")
                .context("missing viewpoint subset")?
                .clone()
        };
        let clusters = part
            .iter()
            .filter(|r| eth || r.group == "viewpoint")
            .map(|r| r.cluster.clone())
            .collect::<BTreeSet<_>>();
        let means = clusters
            .iter()
            .map(|cluster| {
                let mut g = BTreeMap::<String, Vec<f64>>::new();
                for r in &part {
                    if &r.cluster == cluster {
                        g.entry(r.group.clone()).or_default().push(r.metrics.aepe);
                    }
                }
                g.values()
                    .map(|v| v.iter().sum::<f64>() / v.len() as f64)
                    .sum::<f64>()
                    / g.len() as f64
            })
            .collect::<Vec<_>>();
        methods.insert(method.clone(),json!({"primary":primary,"groups":groups,"aepe_cluster_interval":bootstrap_mean(&means,719)?,"uncertainty_unit":if eth {"scene"}else{"sequence"}}));
    }
    examples.sort_by(|a, b| a["sample"].as_str().cmp(&b["sample"].as_str()));
    let contrasts = c
        .contrasts
        .iter()
        .map(|contrast| {
            ensure!(
                c.methods.contains(&contrast.candidate) && c.methods.contains(&contrast.control),
                "contrast readouts must be scored"
            );
            crate::contrasts::paired(&rows, contrast, eth)
        })
        .collect::<Result<Vec<_>>>()?;
    let report = json!({"schema":1,"checkpoint_sha256":c.checkpoint_sha256,"benchmark":c.benchmark,"evaluation_use":c.evaluation_use,"examples":examples,
        "pairs":meta.len(),"methods":methods,"contrasts":contrasts,"rows":rows,"protocol":if eth {"original pixels; pair then scene then interval means; patch-grid displacement"}else{"HP-240; pair then sequence means; primary viewpoint subset; patch-grid displacement"},
        "readout_protocols":readout_protocols,"comparability":"Local patch-grid readouts; optional 3x3 refinement is declared per method. No parity with published refinement or SOTA claim.","inputs":c,"provenance_sha256":sha256_file(&c.provenance)?});
    if let Some(p) = c.output.parent() {
        fs::create_dir_all(p)?;
    }
    write_json(&c.output, &report)?;
    Ok(report)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fractional_flow_preserves_integer_oracle_and_subpatch_units() {
        let indices = (0..256).collect::<Vec<_>>();
        let points = indices
            .iter()
            .map(|&i| [(i % 16) as f64, (i / 16) as f64])
            .collect::<Vec<_>>();
        for xy in [[0., 0.], [123.5, 211.], [639., 479.]] {
            assert_eq!(
                fractional_flow_at(&points, 16, xy, [480, 640]).unwrap(),
                flow_at(&indices, 16, xy, [480, 640]).unwrap()
            );
        }
        let moved = points
            .iter()
            .map(|p| [(p[0] + 0.25).min(15.), p[1]])
            .collect::<Vec<_>>();
        assert_eq!(
            fractional_flow_at(&moved, 16, [100., 100.], [480, 640]).unwrap(),
            [10., 0.]
        );
        assert_eq!(
            fractional_flow_at(&moved, 16, [100., 100.], [256, 256]).unwrap(),
            [4., 0.]
        );
        assert!(fractional_flow_at(&[[f64::NAN; 2]; 256], 16, [0., 0.], [256, 256]).is_err());
    }
    #[test]
    fn identity_and_translation_flow() {
        let ids = (0..256).collect::<Vec<_>>();
        assert_eq!(flow_at(&ids, 16, [1., 77.], [480, 640]).unwrap(), [0., 0.]);
        let ids = (0..256).map(|i| (i + 1).min(255)).collect::<Vec<_>>();
        assert_eq!(
            flow_at(&ids, 16, [60., 60.], [256, 256]).unwrap(),
            [16., 0.]
        );
        let points = hp_truth(
            &[1., 0., 0., 0., 1., 0., 0., 0., 1.],
            [240, 240],
            [240, 240],
        )
        .unwrap();
        assert_eq!(points.len(), 57600);
        assert!(points.iter().all(|(_, f)| *f == [0., 0.]));
    }
}
