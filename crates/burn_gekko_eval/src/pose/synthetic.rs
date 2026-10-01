//! Synthetic camera retention from dense RGB predictions, never visibility-filtered matches.
use super::{
    benchmark::{PoseRow, summary},
    solver::{SolverConfig, estimate},
};
use anyhow::{Context, Result, ensure};
use burn_gekko_data::{Split, load_geometry, read_dataset_manifest, sha256_file, write_json};
use burn_gekko_metrics::{
    calibration::{CameraTarget, camera_target},
    camera,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};

/// All query positions are retained; mutuality is determined from RGB descriptors alone.
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prediction {
    pub room_seed: u64,
    pub method: String,
    pub indices: Vec<usize>,
    pub mutual: Vec<bool>,
    /// Descriptor-grid indices, with integer coordinates at patch centers.
    pub coordinates: Vec<[f64; 2]>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Export {
    pub schema: u32,
    pub checkpoint_sha256: String,
    pub dataset_id: String,
    pub dataset_manifest_sha256: String,
    pub source_sha256: String,
    pub grid: [usize; 2],
    pub rooms: usize,
    pub target: usize,
    pub reference: usize,
    pub predictions: Vec<Prediction>,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub dataset: PathBuf,
    pub predictions: PathBuf,
    pub predictions_sha256: String,
    pub checkpoint_sha256: String,
    pub rooms: usize,
    pub methods: Vec<String>,
    pub seeds: Vec<u64>,
    pub max_trials: usize,
    pub min_trials: usize,
    pub confidence: f64,
    pub min_inliers: usize,
    pub threshold_pixels: f64,
    pub minimum_baseline_meters: f64,
    pub output: PathBuf,
}

/// Convert Bevy (+Y up, -Z forward) into calibrated computer-vision coordinates.
fn camera_frame(mut t: CameraTarget) -> CameraTarget {
    let sign = [1., -1., -1.];
    for i in 0..3 {
        t.translation[i] *= sign[i];
        for j in 0..3 {
            t.rotation[i][j] *= sign[i] * sign[j];
        }
    }
    t
}

fn matches(
    p: &Prediction,
    grid: [usize; 2],
    hw: [usize; 2],
    focal: [f64; 2],
) -> Result<Vec<[[f64; 2]; 2]>> {
    let [h, w] = grid;
    let n = h.checked_mul(w).context("invalid grid")?;
    ensure!(
        n > 0
            && p.indices.len() == n
            && p.mutual.len() == n
            && p.coordinates.len() == n
            && p.indices.iter().all(|&i| i < n)
            && hw.iter().all(|&v| v > 0)
            && focal.iter().all(|f| f.is_finite() && *f > 0.),
        "invalid dense prediction shape"
    );
    ensure!(
        p.coordinates.iter().all(|xy| xy
            .iter()
            .enumerate()
            .all(|(i, x)| x.is_finite() && (0. ..=(grid[1 - i] - 1) as f64).contains(x))),
        "invalid dense coordinates"
    );
    let normalize = |xy: [f64; 2], f: f64| {
        [
            ((xy[0] + 0.5) * hw[1] as f64 / w as f64 - hw[1] as f64 / 2.) / f,
            ((xy[1] + 0.5) * hw[0] as f64 / h as f64 - hw[0] as f64 / 2.) / f,
        ]
    };
    Ok((0..n)
        .filter(|&i| p.mutual[i])
        .map(|i| {
            [
                normalize([(i % w) as f64, (i / w) as f64], focal[0]),
                normalize(p.coordinates[i], focal[1]),
            ]
        })
        .collect())
}

fn row(
    p: &Prediction,
    points: &[[[f64; 2]; 2]],
    truth: &CameraTarget,
    solver: &SolverConfig,
    minimum_baseline: f64,
) -> Result<PoseRow> {
    camera::rotation_degrees(truth.rotation, truth.rotation)?;
    let baseline = truth.translation.iter().map(|x| x * x).sum::<f64>().sqrt();
    ensure!(baseline.is_finite(), "nonfinite camera label");
    let fit = estimate(points, solver)?;
    let rotation = fit.pose.as_ref().map_or(Ok(180.), |p| {
        camera::rotation_degrees(p.rotation, truth.rotation)
    })?;
    let translation = if baseline < minimum_baseline || baseline < 1e-8 {
        None
    } else {
        Some(
            fit.pose
                .as_ref()
                .map_or(Ok(Some(180.)), |p| {
                    camera::translation_degrees(p.translation, truth.translation)
                })?
                .unwrap(),
        )
    };
    Ok(PoseRow {
        pair: p.room_seed.to_string(),
        sequence: "synthetic_validation".into(),
        interval: 1,
        method: p.method.clone(),
        mutual_matches: points.len(),
        fit,
        rotation_degrees: rotation,
        translation_degrees: translation,
        pose_degrees: translation.map(|t| t.max(rotation)),
        baseline_meters: baseline,
    })
}

pub fn score(c: &Config) -> Result<Value> {
    ensure!(
        !c.output.exists(),
        "preserve existing synthetic pose report"
    );
    ensure!(
        fs::canonicalize(
            c.output
                .ancestors()
                .skip(1)
                .find(|p| p.exists())
                .context("missing output parent")?
        )?
        .starts_with(fs::canonicalize(".data")?),
        "pose report outside .data"
    );
    ensure!(
        (1..=128).contains(&c.rooms)
            && (2..=16).contains(&c.seeds.len())
            && c.seeds.iter().collect::<BTreeSet<_>>().len() == c.seeds.len()
            && c.methods.iter().collect::<BTreeSet<_>>().len() == c.methods.len()
            && !c.methods.is_empty()
            && c.threshold_pixels.is_finite()
            && c.threshold_pixels > 0.
            && c.minimum_baseline_meters.is_finite()
            && c.minimum_baseline_meters >= 0.,
        "invalid synthetic pose protocol"
    );
    ensure!(
        sha256_file(&c.predictions)? == c.predictions_sha256,
        "prediction hash differs"
    );
    let export: Export = serde_json::from_slice(&fs::read(&c.predictions)?)?;
    let dataset = read_dataset_manifest(&c.dataset)?;
    let manifest_path = c.dataset.join("manifest.json");
    ensure!(
        export.schema == 1
            && export.checkpoint_sha256 == c.checkpoint_sha256
            && export.dataset_id == dataset.dataset_id
            && sha256_file(&manifest_path)? == export.dataset_manifest_sha256
            && export.rooms == c.rooms
            && export.target == 0
            && export.reference == 1
            && export.grid == [dataset.config.height / 16, dataset.config.width / 16]
            && dataset.config.height.is_multiple_of(16)
            && dataset.config.width.is_multiple_of(16),
        "export identity differs"
    );
    let mut inputs = BTreeMap::from([
        (c.predictions.clone(), c.predictions_sha256.clone()),
        (manifest_path, export.dataset_manifest_sha256.clone()),
    ]);
    let entries = dataset
        .scenes
        .iter()
        .filter(|s| s.split == Split::Validation)
        .take(c.rooms)
        .collect::<Vec<_>>();
    ensure!(entries.len() == c.rooms, "insufficient validation rooms");
    let expected = entries
        .iter()
        .flat_map(|e| c.methods.iter().map(move |m| (e.seed, m.as_str())))
        .collect::<BTreeSet<_>>();
    let actual = export
        .predictions
        .iter()
        .map(|p| (p.room_seed, p.method.as_str()))
        .collect::<BTreeSet<_>>();
    ensure!(
        actual == expected && actual.len() == export.predictions.len(),
        "incomplete or duplicate prediction cohort"
    );
    let mut prepared = Vec::new();
    for entry in entries {
        let path = c.dataset.join("raw").join(&entry.file);
        ensure!(
            sha256_file(&path)? == entry.sha256,
            "geometry shard changed"
        );
        inputs.insert(path.clone(), entry.sha256.clone());
        let geometry = load_geometry(&path)?;
        ensure!(
            geometry.fovy.len() > 1 && geometry.world_from_view.len() > 1,
            "missing camera labels"
        );
        let truth = camera_frame(camera_target(
            &geometry.world_from_view[0],
            &geometry.world_from_view[1],
            geometry.fovy[0],
            geometry.width,
            geometry.height,
        )?);
        let focal =
            [0, 1].map(|i| geometry.height as f64 / (2. * (geometry.fovy[i] as f64 / 2.).tan()));
        for p in export
            .predictions
            .iter()
            .filter(|p| p.room_seed == entry.seed)
        {
            let points = matches(p, export.grid, [geometry.height, geometry.width], focal)?;
            prepared.push((
                p,
                points,
                truth.clone(),
                c.threshold_pixels / ((focal[0] + focal[1]) / 2.),
            ));
        }
    }
    let mut seeds = Vec::new();
    let mut aucs: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for &seed in &c.seeds {
        let mut rows = Vec::new();
        for (p, points, truth, threshold) in &prepared {
            rows.push(row(
                p,
                points,
                truth,
                &SolverConfig {
                    max_trials: c.max_trials,
                    min_trials: c.min_trials,
                    confidence: c.confidence,
                    min_inliers: c.min_inliers,
                    threshold: *threshold,
                    seed,
                },
                c.minimum_baseline_meters,
            )?);
        }
        let mut methods = BTreeMap::new();
        for method in &c.methods {
            let s = summary(
                &rows
                    .iter()
                    .filter(|r| &r.method == method)
                    .collect::<Vec<_>>(),
            )?;
            if let Some(auc) = s.pose_auc_10 {
                aucs.entry(method.clone()).or_default().push(auc);
            }
            methods.insert(method, s);
        }
        seeds.push(json!({"seed":seed,"methods":methods,"rows":rows}));
    }
    let methods:BTreeMap<_,_> = aucs.into_iter().map(|(method,values)| {
        let mean = values.iter().sum::<f64>()/values.len() as f64;
        (method,json!({"mean_pose_auc_10":mean,"minimum_pose_auc_10":values.iter().copied().reduce(f64::min),"maximum_pose_auc_10":values.iter().copied().reduce(f64::max),"solver_seeds":values.len()}))
    }).collect();
    let result = json!({"schema":1,"status":"development_diagnostic","checkpoint_sha256":c.checkpoint_sha256,"dataset_id":export.dataset_id,
        "protocol":"First declared validation rooms, view 0 to view 1. All dense RGB mutual matches enter calibrated eight-point RANSAC; geometry visibility never filters matches. Known focal lengths only normalize coordinates and the pixel threshold (divided by mean focal). Failed fits count as 180 degrees. Low baselines excluded from translation/pose only. Equal rooms within seed, then mean across all declared seeds; never best-seed selection.",
        "limitations":["Synthetic reused development data, not external transfer or SotA evidence.","Known calibration and a geometric solver; not learned camera-head or predicted intrinsic accuracy."],
        "rooms":c.rooms,"methods":methods,"seeds":seeds,"inputs":inputs,"config":c});
    if let Some(parent) = c.output.parent() {
        fs::create_dir_all(parent)?;
    }
    write_json(&c.output, &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renderer_projection_recovers_signed_motion_and_failed_fits_stay_in_denominator() {
        let target = [
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        ];
        let (s, c) = 0.06_f32.sin_cos();
        let reference = [
            c, 0., -s, 0., 0., 1., 0., 0., s, 0., c, 0., 0.24, 0.03, 0.015, 1.,
        ];
        let fov = 1.3;
        let focal = 256. / (2. * (fov as f64 / 2.).tan());
        let truth = camera_frame(camera_target(&target, &reference, fov, 256, 256).unwrap());
        let mut p = Prediction {
            room_seed: 1,
            method: "fixture".into(),
            indices: vec![0; 256],
            mutual: vec![false; 256],
            coordinates: vec![[0.; 2]; 256],
        };
        for i in 0..256 {
            let z = 2. + ((i * 31) % 47) as f32 / 11.;
            let x = ((i % 16) as f32 + 0.5) * 16. - 128.;
            let y = ((i / 16) as f32 + 0.5) * 16. - 128.;
            let xyz = [x * z / focal as f32, -y * z / focal as f32, -z];
            let [u, v, _] = burn_gekko_data::project(xyz, &reference, fov, 256, 256).unwrap();
            let xy = [u as f64 / 16. - 0.5, v as f64 / 16. - 0.5];
            if xy.iter().all(|x| (0. ..=15.).contains(x)) {
                p.mutual[i] = true;
                p.coordinates[i] = xy;
                p.indices[i] = xy[0].round() as usize + 16 * xy[1].round() as usize;
            }
        }
        let points = matches(&p, [16, 16], [256, 256], [focal; 2]).unwrap();
        let solver = SolverConfig {
            max_trials: 256,
            min_trials: 64,
            confidence: 0.999,
            min_inliers: 12,
            threshold: 0.001,
            seed: 871,
        };
        let good = row(&p, &points, &truth, &solver, 0.01).unwrap();
        assert!(good.rotation_degrees < 0.05, "{good:?}");
        assert!(good.translation_degrees.unwrap() < 0.05, "{good:?}");
        p.mutual.fill(false);
        let points = matches(&p, [16, 16], [256, 256], [focal; 2]).unwrap();
        let failed = row(&p, &points, &truth, &solver, 0.01).unwrap();
        assert_eq!(failed.rotation_degrees, 180.);
        assert_eq!(failed.translation_degrees, Some(180.));
        let both = summary(&[&good, &failed]).unwrap();
        assert_eq!(both.pairs, 2);
        assert_eq!(both.success_fraction, 0.5);
        assert!(both.pose_auc_10.unwrap() < 0.51);
        let mut zero = truth;
        zero.translation = [0.; 3];
        let excluded = row(&p, &points, &zero, &solver, 0.01).unwrap();
        assert_eq!(excluded.translation_degrees, None);
        assert_eq!(summary(&[&excluded]).unwrap().excluded_low_baseline, 1);
    }

    #[test]
    fn dense_coordinates_are_checked_even_for_nonmutual_queries_and_rectangular_grids() {
        let mut p = Prediction {
            room_seed: 1,
            method: "fixture".into(),
            indices: vec![0; 6],
            mutual: vec![false; 6],
            coordinates: vec![[2., 1.]; 6],
        };
        assert!(
            matches(&p, [2, 3], [32, 48], [40., 40.])
                .unwrap()
                .is_empty()
        );
        p.coordinates[0][1] = 2.;
        assert!(matches(&p, [2, 3], [32, 48], [40., 40.]).is_err());
        p.coordinates[0][1] = f64::NAN;
        assert!(matches(&p, [2, 3], [32, 48], [40., 40.]).is_err());
    }
}
