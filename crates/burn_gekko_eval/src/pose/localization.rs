//! CPU localization ablation. Camera truth diagnoses matches but never selects solver input.
mod publication;
use super::{
    benchmark::{PoseReport, PoseRow, Prediction, summarize},
    replay::{Export, load},
    solver::{MinimalSolver, estimate_with, sampson},
};
use crate::camera;
use anyhow::{Context, Result, ensure};
use burn_gekko_data::{
    real_views::{PairImages, PairLabels, PoseLabel, original_pixel},
    sha256_file, write_json,
};
use nalgebra::{Matrix3, Vector3};
pub use publication::publish;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub original: PathBuf,
    pub original_sha256: String,
    pub seeds: Vec<u64>,
    #[serde(default)]
    pub minimal_solver: MinimalSolver,
    pub output: PathBuf,
}

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum Coordinates {
    Recorded,
    Hard,
}
impl Coordinates {
    fn name(self) -> &'static str {
        match self {
            Self::Recorded => "recorded",
            Self::Hard => "hard",
        }
    }
}
type Match = [[f64; 2]; 2];

fn points(p: &Prediction, hw: [[usize; 2]; 2], k: [f64; 4], mode: Coordinates) -> Vec<Match> {
    let [fx, fy, cx, cy] = k;
    let normalize = |x: [f64; 2]| [(x[0] - cx) / fx, (x[1] - cy) / fy];
    (0..p.indices.len())
        .filter(|&i| p.mutual[i])
        .map(|i| {
            let a = [(i % p.grid[1]) as f64, (i / p.grid[1]) as f64];
            let j = p.indices[i];
            let b = match mode {
                Coordinates::Recorded => p.coordinates[i],
                Coordinates::Hard => [(j % p.grid[1]) as f64, (j / p.grid[1]) as f64],
            };
            [
                normalize(original_pixel(a, p.grid, hw[0])),
                normalize(original_pixel(b, p.grid, hw[1])),
            ]
        })
        .collect()
}

fn essential(gt: &PoseLabel) -> Matrix3<f64> {
    let [x, y, z] = gt.translation;
    let cross = Matrix3::new(0., -z, y, z, 0., -x, -y, x, 0.);
    cross * Matrix3::from_fn(|i, j| gt.rotation[i][j])
}

/// Probability of at least one entirely epipolar-consistent minimal sample at the
/// maximum budget. Sampling within each trial is without replacement. This is
/// not a solver-success probability: epipolar consistency does not prove a match.
fn sampling_opportunity(consistent: usize, total: usize, trials: usize, sample_size: usize) -> f64 {
    if consistent < sample_size || total < sample_size {
        return 0.;
    }
    let p = (0..sample_size)
        .map(|i| (consistent - i) as f64 / (total - i) as f64)
        .product::<f64>();
    if p >= 1. {
        1.
    } else {
        -((trials as f64) * (-p).ln_1p()).exp_m1()
    }
}

#[derive(Serialize)]
struct MatchDiagnostic {
    pair: String,
    sequence: String,
    interval: usize,
    method: String,
    coordinates: Coordinates,
    matches: usize,
    baseline_meters: f64,
    truth_rotation_degrees: f64,
    /// None for excluded low-baseline pairs; never silently treat them as perfect matches.
    mean_gt_sampson_original_pixels: Option<f64>,
    median_gt_sampson_original_pixels: Option<f64>,
    gt_consistent_fraction: Option<f64>,
    sampling_opportunity_at_max_trials: Option<f64>,
}

fn mean(x: impl Iterator<Item = f64>) -> Option<f64> {
    let v = x.collect::<Vec<_>>();
    (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
}
fn range(v: &[f64]) -> Value {
    json!({"mean":mean(v.iter().copied()),"minimum":v.iter().copied().reduce(f64::min),"maximum":v.iter().copied().reduce(f64::max),"count":v.len()})
}

pub fn analyze(c: &Config) -> Result<Value> {
    let started = std::time::Instant::now();
    ensure!(!c.output.exists(), "preserve existing localization study");
    ensure!(
        sha256_file(&c.original)? == c.original_sha256,
        "original pose report changed"
    );
    ensure!(
        (2..=16).contains(&c.seeds.len())
            && c.seeds.iter().collect::<BTreeSet<_>>().len() == c.seeds.len(),
        "declare 2 to 16 distinct solver seeds"
    );
    ensure!(
        fs::canonicalize(
            c.output
                .ancestors()
                .skip(1)
                .find(|p| p.exists())
                .context("output parent")?
        )?
        .starts_with(fs::canonicalize(".data")?),
        "localization output outside .data"
    );
    let original: PoseReport = serde_json::from_slice(&fs::read(&c.original)?)?;
    ensure!(
        original.evaluation_use == "development" && c.seeds.contains(&original.config.solver.seed),
        "diagnose development data and retain original seed"
    );
    let cfg = &original.config;
    let mut sources = original.inputs.clone();
    for (path, hash) in [
        (&cfg.images, &cfg.images_sha256),
        (&cfg.labels, &cfg.labels_sha256),
        (&cfg.predictions, &cfg.predictions_sha256),
        (&cfg.provenance, &cfg.provenance_sha256),
    ] {
        ensure!(
            sources.get(path) == Some(hash),
            "incomplete original source closure"
        );
    }
    for (path, hash) in &sources {
        ensure!(sha256_file(path)? == *hash, "original pose input changed");
    }
    sources.insert(c.original.clone(), c.original_sha256.clone());
    let images: PairImages = serde_json::from_slice(&fs::read(&cfg.images)?)?;
    let labels: PairLabels = serde_json::from_slice(&fs::read(&cfg.labels)?)?;
    let (predictions, provenance) = load(&Export {
        predictions: cfg.predictions.clone(),
        predictions_sha256: cfg.predictions_sha256.clone(),
        provenance: cfg.provenance.clone(),
        provenance_sha256: cfg.provenance_sha256.clone(),
    })?;
    ensure!(
        images.schema == 1
            && labels.schema == 1
            && labels.image_manifest_sha256 == cfg.images_sha256
            && images.evaluation_use == "development"
            && original.image_size == images.image_size,
        "pose dataset contract mismatch"
    );
    ensure!(
        original.checkpoint_sha256 == cfg.checkpoint_sha256
            && provenance["checkpoint_sha256"] == cfg.checkpoint_sha256
            && provenance["images_sha256"] == cfg.images_sha256
            && provenance["methods"] == json!(cfg.methods)
            && provenance["evaluation_use"] == images.evaluation_use,
        "pose export identity mismatch"
    );
    let focal = (labels.intrinsics[0] + labels.intrinsics[1]) / 2.;
    ensure!(
        labels.intrinsics.iter().all(|v| v.is_finite())
            && labels.intrinsics[0] > 0.
            && labels.intrinsics[1] > 0.
            && (cfg.solver.threshold - cfg.threshold_original_pixels / focal).abs() < 1e-12
            && cfg.minimum_baseline_meters.is_finite()
            && cfg.minimum_baseline_meters >= 0.,
        "invalid pose calibration or thresholds"
    );
    let grid = burn_gekko_data::real_views::descriptor_grid(images.image_size)?;
    let pairs = images
        .pairs
        .iter()
        .map(|p| (p.id.clone(), p))
        .collect::<BTreeMap<_, _>>();
    let truth = labels
        .rows
        .iter()
        .map(|p| (p.id.clone(), p))
        .collect::<BTreeMap<_, _>>();
    let old = original
        .rows
        .iter()
        .map(|p| ((p.pair.clone(), p.method.clone()), p))
        .collect::<BTreeMap<_, _>>();
    let expected = pairs
        .keys()
        .flat_map(|id| cfg.methods.iter().map(move |m| (id.clone(), m.clone())))
        .collect::<BTreeSet<_>>();
    ensure!(
        !pairs.is_empty()
            && pairs.len() == images.pairs.len()
            && truth.len() == labels.rows.len()
            && pairs.keys().eq(truth.keys())
            && old.len() == original.rows.len()
            && old.keys().cloned().collect::<BTreeSet<_>>() == expected
            && predictions.keys().cloned().collect::<BTreeSet<_>>() == expected
            && !cfg.methods.is_empty()
            && cfg.methods.iter().collect::<BTreeSet<_>>().len() == cfg.methods.len()
            && original.methods.keys().collect::<BTreeSet<_>>()
                == cfg.methods.iter().collect::<BTreeSet<_>>(),
        "incomplete or duplicate pose population"
    );
    fs::create_dir_all(&c.output)?;
    let mut diagnostics = Vec::new();
    let mut methods = BTreeMap::new();
    let mut contrasts = Vec::new();
    for mode in [Coordinates::Recorded, Coordinates::Hard] {
        let mut reports = Vec::new();
        let mut consensus = Vec::new();
        for &seed in &c.seeds {
            let mut rows = Vec::new();
            let mut solver = cfg.solver.clone();
            solver.seed = seed;
            for (key, p) in &predictions {
                ensure!(p.grid == grid, "prediction grid differs from images");
                let pair = pairs[&p.pair];
                let gt = truth[&p.pair];
                let a = images
                    .images
                    .get(&pair.target)
                    .context("missing target image")?;
                let b = images
                    .images
                    .get(&pair.reference)
                    .context("missing reference image")?;
                ensure!(
                    a.original_hw.iter().chain(&b.original_hw).all(|&d| d > 0),
                    "invalid original image shape"
                );
                let matches = points(p, [a.original_hw, b.original_hw], labels.intrinsics, mode);
                // All mutual RGB matches enter estimate, before any ground-truth diagnostic.
                let fit = estimate_with(&matches, &solver, c.minimal_solver)?;
                let baseline = Vector3::from_row_slice(&gt.translation).norm();
                ensure!(baseline.is_finite(), "nonfinite camera translation");
                let eligible = baseline >= cfg.minimum_baseline_meters && baseline >= 1e-8;
                let rotation = fit.pose.as_ref().map_or(Ok(180.), |p| {
                    camera::rotation_degrees(p.rotation, gt.rotation)
                })?;
                let translation = if eligible {
                    Some(fit.pose.as_ref().map_or(Ok(180.), |p| {
                        camera::translation_degrees(p.translation, gt.translation)
                            .map(|t| t.unwrap())
                    })?)
                } else {
                    None
                };
                let truth_rotation = camera::rotation_degrees(
                    gt.rotation,
                    [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                )?;
                let errors = matches
                    .iter()
                    .map(|&p| sampson(&essential(gt), p).sqrt() * focal)
                    .collect::<Vec<_>>();
                let consistent = errors
                    .iter()
                    .filter(|&&x| x <= cfg.threshold_original_pixels)
                    .count();
                if seed == c.seeds[0] {
                    let mut sorted = errors.clone();
                    sorted.sort_by(f64::total_cmp);
                    let valid = eligible && !matches.is_empty();
                    // Infinite residuals (epipole/zero denominator) count as inconsistent;
                    // a finite mean is undefined if any such point occurs.
                    diagnostics.push(MatchDiagnostic {
                        pair: p.pair.clone(),
                        sequence: pair.sequence.clone(),
                        interval: pair.interval,
                        method: p.method.clone(),
                        coordinates: mode,
                        matches: matches.len(),
                        baseline_meters: baseline,
                        truth_rotation_degrees: truth_rotation,
                        mean_gt_sampson_original_pixels: valid
                            .then(|| mean(errors.iter().copied()))
                            .flatten()
                            .filter(|v| v.is_finite()),
                        median_gt_sampson_original_pixels: valid
                            .then(|| {
                                (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.
                            })
                            .filter(|v| v.is_finite()),
                        gt_consistent_fraction: valid
                            .then(|| consistent as f64 / matches.len() as f64),
                        sampling_opportunity_at_max_trials: valid.then(|| {
                            sampling_opportunity(
                                consistent,
                                matches.len(),
                                solver.max_trials,
                                c.minimal_solver.sample_size(),
                            )
                        }),
                    });
                }
                let inliers = fit.inliers.iter().filter(|&&x| x).count();
                let agreement = fit
                    .inliers
                    .iter()
                    .zip(&errors)
                    .filter(|(inlier, e)| **inlier && **e <= cfg.threshold_original_pixels)
                    .count();
                consensus.push(json!({"seed":seed,"pair":p.pair,"sequence":pair.sequence,"interval":pair.interval,"method":p.method,"ransac_inliers":inliers,"gt_consistent_ransac_inlier_fraction":(eligible && inliers > 0).then(|| agreement as f64 / inliers as f64)}));
                let row = PoseRow {
                    pair: p.pair.clone(),
                    sequence: pair.sequence.clone(),
                    interval: pair.interval,
                    method: p.method.clone(),
                    mutual_matches: matches.len(),
                    fit,
                    rotation_degrees: rotation,
                    translation_degrees: translation,
                    pose_degrees: translation.map(|t| t.max(rotation)),
                    baseline_meters: baseline,
                };
                if matches!(mode, Coordinates::Recorded)
                    && seed == cfg.solver.seed
                    && c.minimal_solver == MinimalSolver::EightPoint
                {
                    let prior = old[key];
                    ensure!(
                        prior.fit.inliers == row.fit.inliers
                            && prior.fit.trials == row.fit.trials
                            && prior.fit.failure == row.fit.failure
                            && (prior.rotation_degrees - row.rotation_degrees).abs() < 1e-12
                            && match (prior.translation_degrees, row.translation_degrees) {
                                (Some(a), Some(b)) => (a - b).abs() < 1e-12,
                                (None, None) => true,
                                _ => false,
                            },
                        "original solver replay differs"
                    );
                }
                rows.push(row);
            }
            let summary = summarize(&rows, &cfg.methods)?;
            let intervals = rows
                .iter()
                .map(|r| r.interval)
                .collect::<BTreeSet<_>>()
                .into_iter()
                .map(|interval| {
                    let selected = rows
                        .iter()
                        .filter(|r| r.interval == interval)
                        .cloned()
                        .collect::<Vec<_>>();
                    Ok((interval, summarize(&selected, &cfg.methods)?))
                })
                .collect::<Result<BTreeMap<_, _>>>()?;
            let path = c.output.join(format!("{}-seed-{seed}.json", mode.name()));
            let report = json!({"schema":1,"minimal_solver":c.minimal_solver,"coordinates":mode,"seed":seed,"methods":summary,"intervals":intervals,"rows":rows});
            write_json(&path, &report)?;
            sources.insert(path.clone(), sha256_file(&path)?);
            reports.push(report);
        }
        let candidate = "spatial_residual_conditional_local";
        for control in cfg.methods.iter().filter(|m| m.as_str() != candidate) {
            let mut seed_rows = Vec::new();
            for report in &reports {
                let a = &report["methods"][candidate];
                let b = &report["methods"][control];
                let gain = a["macro_pose_auc_10"].as_f64().context("candidate AUC")?
                    - b["macro_pose_auc_10"].as_f64().context("control AUC")?;
                let sequence_gains = a["sequences"]
                    .as_object()
                    .context("sequence summary")?
                    .iter()
                    .map(|(seq, value)| {
                        Ok((
                            seq.clone(),
                            value["pose_auc_10"]
                                .as_f64()
                                .context("candidate sequence AUC")?
                                - b["sequences"][seq]["pose_auc_10"]
                                    .as_f64()
                                    .context("control sequence AUC")?,
                        ))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()?;
                let passed = sequence_gains.values().all(|v| *v > 0.)
                    && a["macro_success_fraction"]
                        .as_f64()
                        .context("candidate success")?
                        >= b["macro_success_fraction"]
                            .as_f64()
                            .context("control success")?;
                seed_rows.push(json!({"seed":report["seed"],"macro_auc10_gain":gain,"sequence_auc10_gains":sequence_gains,"passed":passed}));
            }
            contrasts.push(json!({"coordinates":mode,"candidate":candidate,"control":control,"gate":"positive AUC@10 gain in every sequence and no macro solver-success regression","passing_seeds":seed_rows.iter().filter(|r|r["passed"]==true).count(),"total_seeds":c.seeds.len(),"seeds":seed_rows}));
        }
        for name in &cfg.methods {
            let values = |field: &str| {
                reports
                    .iter()
                    .map(|r| {
                        r["methods"][name][field]
                            .as_f64()
                            .context("missing pose summary")
                    })
                    .collect::<Result<Vec<_>>>()
            };
            let stats = diagnostics
                .iter()
                .filter(|r| r.method == *name && r.coordinates.name() == mode.name())
                .collect::<Vec<_>>();
            let mut sequences = BTreeMap::new();
            for sequence in original.methods[name].sequences.keys() {
                let auc = reports
                    .iter()
                    .map(|r| {
                        r["methods"][name]["sequences"][sequence]["pose_auc_10"]
                            .as_f64()
                            .context("missing sequence AUC")
                    })
                    .collect::<Result<Vec<_>>>()?;
                sequences.insert(sequence.clone(), range(&auc));
            }
            methods.insert(format!("{}/{name}", mode.name()), json!({
                "auc10":range(&values("macro_pose_auc_10")?),"rotation_degrees":range(&values("macro_rotation_degrees")?),"translation_degrees":range(&values("macro_translation_degrees")?),"recall10":range(&values("macro_pose_recall_10")?),"solver_success_fraction":range(&values("macro_success_fraction")?),"sequence_auc10":sequences,
                "pairs":stats.len(),"epipolar_eligible_pairs":stats.iter().filter(|r|r.gt_consistent_fraction.is_some()).count(),
                "pair_mean_gt_consistent_fraction":mean(stats.iter().filter_map(|r|r.gt_consistent_fraction)),
                "pair_mean_sampling_opportunity_at_max_trials":mean(stats.iter().filter_map(|r|r.sampling_opportunity_at_max_trials)),
                "pair_seed_mean_gt_consistent_ransac_inlier_fraction":mean(consensus.iter().filter(|r|r["method"] == *name).filter_map(|r|r["gt_consistent_ransac_inlier_fraction"].as_f64()))
            }));
        }
        let path = c.output.join(format!("{}-consensus.json", mode.name()));
        write_json(&path, &consensus)?;
        sources.insert(path.clone(), sha256_file(&path)?);
    }
    let diagnostic_path = c.output.join("matches.json");
    write_json(&diagnostic_path, &diagnostics)?;
    sources.insert(diagnostic_path.clone(), sha256_file(&diagnostic_path)?);
    let result = json!({"schema":1,"status":"localization_and_consensus_diagnostic","checkpoint_sha256":cfg.checkpoint_sha256,"config":c,"solver":cfg.solver,"minimal_sample_points":c.minimal_solver.sample_size(),"original_eight_point_replay_checked":c.minimal_solver==MinimalSolver::EightPoint,"cpu_command_seconds":started.elapsed().as_secs_f64(),"threshold_original_pixels":cfg.threshold_original_pixels,"minimum_baseline_meters":cfg.minimum_baseline_meters,"methods":methods,"contrasts":contrasts,"sources":sources,
        "protocol":"Frozen RGB predictions; recorded local centroid versus hard patch center, with identical mutual flags, all methods/pairs and every declared solver seed. Known calibration only. All fits precede and are independent of truth diagnostics; failed poses count as 180 degrees. Mean camera metrics weight pairs within sequence, sequences within seed, then every seed. Epipolar summaries weight eligible pairs equally. Eight-point mode verifies the original seed replay; five-point mode changes only the minimal hypothesis solver and RANSAC sample size/stopping exponent. Both modes retain the same final linear refit and cheirality. No best-seed/readout selection or new trained weights.",
        "limitations":["Repeated TUM development data, not independent qualification.","Ground-truth epipolar consistency is a necessary, not sufficient, condition for a correct correspondence; along-line errors can pass.","Sampling opportunity assumes the maximum trial budget and independent trials; it is not pose-success probability or an estimate of real correspondence precision.","Solver-seed ranges are sensitivity ranges, not scene or training uncertainty.","Hard and recorded labels change coordinate readout only; exported descriptor method IDs retain their original names."]});
    write_json(&c.output.join("summary.json"), &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn without_replacement_sampling_handles_small_populations() {
        assert_eq!(sampling_opportunity(7, 100, 2048, 8), 0.);
        assert_eq!(sampling_opportunity(8, 8, 2048, 8), 1.);
        assert!((sampling_opportunity(8, 9, 1, 8) - 1. / 9.).abs() < 1e-14);
        assert!((sampling_opportunity(8, 9, 2, 8) - (1. - (8_f64 / 9.).powi(2))).abs() < 1e-14);
        assert!(sampling_opportunity(40, 100, 2048, 5) > sampling_opportunity(40, 100, 2048, 8));
    }
    #[test]
    fn coordinate_ablation_preserves_mutual_population_and_resize_geometry() {
        let p = Prediction {
            pair: "p".into(),
            method: "m".into(),
            grid: [2, 2],
            indices: vec![1, 0, 3, 2],
            mutual: vec![true, false, true, false],
            coordinates: vec![[0.75, 0.], [0., 0.], [0.8, 1.], [0., 1.]],
        };
        let hard = points(
            &p,
            [[480, 640]; 2],
            [500., 500., 320., 240.],
            Coordinates::Hard,
        );
        let fine = points(
            &p,
            [[480, 640]; 2],
            [500., 500., 320., 240.],
            Coordinates::Recorded,
        );
        assert_eq!(hard.len(), 2);
        assert_eq!(fine.len(), 2);
        assert_eq!(hard[0][0], fine[0][0]);
        assert_eq!(hard[1][0], fine[1][0]);
        assert!((hard[0][1][0] - fine[0][1][0] - 0.16).abs() < 1e-12);
    }
    #[test]
    fn epipolar_consistency_cannot_detect_along_line_errors() {
        let truth = PoseLabel {
            id: "p".into(),
            rotation: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            translation: [1., 0., 0.],
            association_seconds: [0., 0.],
        };
        let e = essential(&truth);
        assert_eq!(sampson(&e, [[0.1, 0.2], [0.4, 0.2]]), 0.);
        assert_eq!(sampson(&e, [[0.1, 0.2], [-0.9, 0.2]]), 0.);
        assert!((sampson(&e, [[0.1, 0.2], [0.4, 0.3]]) - 0.005).abs() < 1e-14);
    }
}
