//! Post-inference pose fitting and failure-inclusive scoring of a frozen RGB export.
use super::solver::{PoseFit, SolverConfig, estimate};
use crate::{
    camera,
    schema::{Capability, CapabilityStatus, Metric},
};
use anyhow::{Context, Result, ensure};
use burn_gekko_data::{
    real_views::{PairImages, PairLabels, original_pixel},
    sha256_file, write_json,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoseScoreConfig {
    pub images: PathBuf,
    pub images_sha256: String,
    pub labels: PathBuf,
    pub labels_sha256: String,
    pub predictions: PathBuf,
    pub predictions_sha256: String,
    pub provenance: PathBuf,
    pub provenance_sha256: String,
    pub checkpoint_sha256: String,
    pub methods: Vec<String>,
    pub threshold_original_pixels: f64,
    pub minimum_baseline_meters: f64,
    pub solver: SolverConfig,
    pub output: PathBuf,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Prediction {
    pub pair: String,
    pub method: String,
    pub grid: [usize; 2],
    pub indices: Vec<usize>,
    pub mutual: Vec<bool>,
    pub coordinates: Vec<[f64; 2]>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseRow {
    pub pair: String,
    pub sequence: String,
    pub interval: usize,
    pub method: String,
    pub mutual_matches: usize,
    pub fit: PoseFit,
    pub rotation_degrees: f64,
    pub translation_degrees: Option<f64>,
    pub pose_degrees: Option<f64>,
    pub baseline_meters: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PoseSummary {
    pub pairs: usize,
    pub pose_pairs: usize,
    pub excluded_low_baseline: usize,
    pub successes: usize,
    pub success_fraction: f64,
    pub mean_rotation_degrees: f64,
    pub mean_translation_degrees: Option<f64>,
    pub pose_auc_5: Option<f64>,
    pub pose_auc_10: Option<f64>,
    pub pose_auc_20: Option<f64>,
    pub pose_recall_5: Option<f64>,
    pub pose_recall_10: Option<f64>,
    pub pose_recall_20: Option<f64>,
    pub mean_mutual_matches: f64,
    pub mean_inliers: f64,
    pub failures: BTreeMap<String, usize>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodSummary {
    pub sequences: BTreeMap<String, PoseSummary>,
    pub macro_success_fraction: f64,
    pub macro_rotation_degrees: f64,
    pub macro_translation_degrees: Option<f64>,
    pub macro_pose_auc_5: Option<f64>,
    pub macro_pose_auc_10: Option<f64>,
    pub macro_pose_auc_20: Option<f64>,
    pub macro_pose_recall_10: Option<f64>,
}
#[derive(Debug, Serialize, Deserialize)]
pub struct PoseReport {
    pub schema: u32,
    #[serde(default = "burn_gekko_data::real_views::default_image_size")]
    pub image_size: usize,
    pub checkpoint_sha256: String,
    pub evaluation_use: String,
    pub protocol: String,
    pub config: PoseScoreConfig,
    pub inputs: BTreeMap<PathBuf, String>,
    pub rows: Vec<PoseRow>,
    pub methods: BTreeMap<String, MethodSummary>,
    pub contrasts: Vec<Value>,
    pub examples: Vec<Value>,
}
pub(crate) fn summary(rows: &[&PoseRow]) -> Result<PoseSummary> {
    ensure!(!rows.is_empty(), "no pose observations");
    let mut errors = Vec::new();
    let mut translations = Vec::new();
    let mut failures = BTreeMap::new();
    for r in rows {
        ensure!(
            r.rotation_degrees.is_finite() && (0. ..=180.).contains(&r.rotation_degrees),
            "invalid rotation error"
        );
        ensure!(
            r.fit.pose.is_some() == r.fit.failure.is_none(),
            "inconsistent pose outcome"
        );
        if let Some(t) = r.translation_degrees {
            ensure!(
                t.is_finite()
                    && (0. ..=180.).contains(&t)
                    && r.pose_degrees == Some(t.max(r.rotation_degrees)),
                "invalid pose error"
            );
            translations.push(t);
            errors.push(t.max(r.rotation_degrees));
        } else {
            ensure!(
                r.pose_degrees.is_none(),
                "pose without translation observation"
            );
        }
        if let Some(f) = &r.fit.failure {
            ensure!(
                r.rotation_degrees == 180. && r.translation_degrees.is_none_or(|v| v == 180.),
                "failed fits must remain in error metrics"
            );
            *failures.entry(f.clone()).or_insert(0) += 1;
        }
    }
    let n = rows.len() as f64;
    let successes = rows.iter().filter(|r| r.fit.pose.is_some()).count();
    let auc = |t| {
        if errors.is_empty() {
            Ok(None)
        } else {
            camera::pose_auc(&errors, t).map(Some)
        }
    };
    let recall = |t| {
        (!errors.is_empty())
            .then(|| errors.iter().filter(|&&e| e <= t).count() as f64 / errors.len() as f64)
    };
    Ok(PoseSummary {
        pairs: rows.len(),
        pose_pairs: errors.len(),
        excluded_low_baseline: rows.len() - errors.len(),
        successes,
        success_fraction: successes as f64 / n,
        mean_rotation_degrees: rows.iter().map(|r| r.rotation_degrees).sum::<f64>() / n,
        mean_translation_degrees: (!translations.is_empty())
            .then(|| translations.iter().sum::<f64>() / translations.len() as f64),
        pose_auc_5: auc(5.)?,
        pose_auc_10: auc(10.)?,
        pose_auc_20: auc(20.)?,
        pose_recall_5: recall(5.),
        pose_recall_10: recall(10.),
        pose_recall_20: recall(20.),
        mean_mutual_matches: rows.iter().map(|r| r.mutual_matches as f64).sum::<f64>() / n,
        mean_inliers: rows
            .iter()
            .map(|r| r.fit.inliers.iter().filter(|&&v| v).count() as f64)
            .sum::<f64>()
            / n,
        failures,
    })
}
pub fn summarize(rows: &[PoseRow], names: &[String]) -> Result<BTreeMap<String, MethodSummary>> {
    ensure!(
        !rows.is_empty()
            && !names.is_empty()
            && names.iter().collect::<BTreeSet<_>>().len() == names.len(),
        "invalid pose population"
    );
    let pairs = rows.iter().map(|r| &r.pair).collect::<BTreeSet<_>>();
    let mut result = BTreeMap::new();
    for name in names {
        let observations = rows
            .iter()
            .filter(|r| r.method == *name)
            .collect::<Vec<_>>();
        ensure!(
            observations.len() == pairs.len()
                && observations
                    .iter()
                    .map(|r| &r.pair)
                    .collect::<BTreeSet<_>>()
                    == pairs,
            "missing or duplicate method observations"
        );
        let mut groups: BTreeMap<String, Vec<&PoseRow>> = BTreeMap::new();
        for r in observations {
            groups.entry(r.sequence.clone()).or_default().push(r);
        }
        let sequences = groups
            .into_iter()
            .map(|(k, v)| Ok((k, summary(&v)?)))
            .collect::<Result<BTreeMap<_, _>>>()?;
        let mean = |f: fn(&PoseSummary) -> f64| {
            sequences.values().map(f).sum::<f64>() / sequences.len() as f64
        };
        let opt = |f: fn(&PoseSummary) -> Option<f64>| {
            sequences
                .values()
                .map(f)
                .collect::<Option<Vec<_>>>()
                .map(|v| v.iter().sum::<f64>() / v.len() as f64)
        };
        result.insert(
            name.clone(),
            MethodSummary {
                macro_success_fraction: mean(|s| s.success_fraction),
                macro_rotation_degrees: mean(|s| s.mean_rotation_degrees),
                macro_translation_degrees: opt(|s| s.mean_translation_degrees),
                macro_pose_auc_5: opt(|s| s.pose_auc_5),
                macro_pose_auc_10: opt(|s| s.pose_auc_10),
                macro_pose_auc_20: opt(|s| s.pose_auc_20),
                macro_pose_recall_10: opt(|s| s.pose_recall_10),
                sequences,
            },
        );
    }
    ensure!(
        rows.len() == names.len() * pairs.len(),
        "undeclared pose methods"
    );
    Ok(result)
}
pub fn capability(report: &PoseReport, method: &str) -> Result<Capability> {
    let s = report.methods.get(method).context("missing pose method")?;
    let n = s.sequences.values().map(|s| s.pairs).sum();
    let pose_n = s.sequences.values().map(|s| s.pose_pairs).sum();
    let mut metrics = Vec::new();
    for (id, label, value, unit, lower, samples) in [
        (
            "success",
            "Solver success",
            Some(s.macro_success_fraction),
            "fraction",
            false,
            n,
        ),
        (
            "rotation",
            "Mean rotation error",
            Some(s.macro_rotation_degrees),
            "degrees",
            true,
            n,
        ),
        (
            "translation",
            "Mean signed translation-direction error",
            s.macro_translation_degrees,
            "degrees",
            true,
            pose_n,
        ),
        (
            "auc5",
            "Pose AUC at 5 degrees",
            s.macro_pose_auc_5,
            "fraction",
            false,
            pose_n,
        ),
        (
            "auc10",
            "Pose AUC at 10 degrees",
            s.macro_pose_auc_10,
            "fraction",
            false,
            pose_n,
        ),
        (
            "auc20",
            "Pose AUC at 20 degrees",
            s.macro_pose_auc_20,
            "fraction",
            false,
            pose_n,
        ),
        (
            "recall10",
            "Poses within 10 degrees",
            s.macro_pose_recall_10,
            "fraction",
            false,
            pose_n,
        ),
    ] {
        if let Some(value) = value {
            metrics.push(Metric {id:id.into(),label:label.into(),value,unit:unit.into(),lower_is_better:lower,samples,aggregation:"equal pairs within each sequence, then equal sequences; failed fits count as 180 degrees".into()});
        }
    }
    Ok(Capability {id:format!("calibrated_pose.{method}"),label:format!("Calibrated camera-motion probe / {method}"),status:CapabilityStatus::Evaluated,protocol:format!("Evaluation role: {}. {}", report.evaluation_use, report.protocol),limitations:vec!["Known RGB intrinsics enter a CPU eight-point RANSAC solver after RGB-only model inference. This is not a trained camera head or predicted calibration.".into(),"Three sequences in one TUM environment; no official relative-pose benchmark parity or broad generalization claim. Failure counts and low-baseline exclusions remain explicit.".into()],metrics})
}
pub fn score(c: &PoseScoreConfig) -> Result<PoseReport> {
    ensure!(!c.output.exists(), "preserve existing pose evaluation");
    ensure!(
        c.threshold_original_pixels > 0.
            && c.threshold_original_pixels.is_finite()
            && c.minimum_baseline_meters >= 0.
            && c.minimum_baseline_meters.is_finite(),
        "invalid pose thresholds"
    );
    let mut inputs = BTreeMap::new();
    for (p, s) in [
        (&c.images, &c.images_sha256),
        (&c.labels, &c.labels_sha256),
        (&c.predictions, &c.predictions_sha256),
        (&c.provenance, &c.provenance_sha256),
    ] {
        ensure!(
            sha256_file(p)? == *s,
            "pose input hash mismatch: {}",
            p.display()
        );
        inputs.insert(p.clone(), s.clone());
    }
    let images: PairImages = serde_json::from_slice(&fs::read(&c.images)?)?;
    let grid = burn_gekko_data::real_views::descriptor_grid(images.image_size)?;
    let tokens = grid[0] * grid[1];
    let labels: PairLabels = serde_json::from_slice(&fs::read(&c.labels)?)?;
    let provenance: Value = serde_json::from_slice(&fs::read(&c.provenance)?)?;
    ensure!(
        images.schema == 1 && labels.schema == 1 && labels.image_manifest_sha256 == c.images_sha256,
        "pose dataset contract mismatch"
    );
    ensure!(
        provenance
            .get("image_size")
            .is_none_or(|v| *v == images.image_size),
        "pose input resolution mismatch"
    );
    ensure!(
        provenance["checkpoint_sha256"] == c.checkpoint_sha256
            && provenance["images_sha256"] == c.images_sha256
            && provenance["predictions_sha256"] == c.predictions_sha256
            && provenance["methods"] == json!(c.methods)
            && provenance["evaluation_use"] == images.evaluation_use,
        "pose export identity mismatch"
    );
    let [fx, fy, cx, cy] = labels.intrinsics;
    ensure!(
        labels.intrinsics.iter().all(|v| v.is_finite()) && fx > 0. && fy > 0.,
        "invalid known calibration"
    );
    ensure!(
        (c.solver.threshold - c.threshold_original_pixels / ((fx + fy) / 2.)).abs() < 1e-12,
        "pixel/normalized threshold mismatch"
    );
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
    ensure!(
        pairs.len() == images.pairs.len()
            && truth.len() == labels.rows.len()
            && pairs.keys().eq(truth.keys()),
        "camera labels and RGB pair set differ"
    );
    let predictions = fs::read_to_string(&c.predictions)?
        .lines()
        .map(|s| Ok(serde_json::from_str::<Prediction>(s)?))
        .collect::<Result<Vec<_>>>()?;
    let declared = c.methods.iter().cloned().collect::<BTreeSet<_>>();
    ensure!(
        declared.len() == c.methods.len()
            && !declared.is_empty()
            && predictions.len() == pairs.len() * declared.len(),
        "incomplete prediction population"
    );
    let mut seen = BTreeSet::new();
    let mut rows = Vec::new();
    let mut examples = Vec::new();
    let mut example_ids = BTreeSet::new();
    for sequence in pairs
        .values()
        .map(|p| p.sequence.clone())
        .collect::<BTreeSet<_>>()
    {
        let ids = pairs
            .values()
            .filter(|p| p.sequence == sequence)
            .map(|p| &p.id)
            .collect::<Vec<_>>();
        for i in [0, ids.len() / 2, ids.len() - 1] {
            example_ids.insert(ids[i].clone());
        }
    }
    for p in predictions {
        ensure!(
            declared.contains(&p.method) && seen.insert((p.pair.clone(), p.method.clone())),
            "duplicate or undeclared pose prediction"
        );
        let pair = pairs.get(&p.pair).context("unknown image pair")?;
        let gt = truth[&p.pair];
        ensure!(
            p.grid == grid
                && p.indices.len() == tokens
                && p.coordinates.len() == tokens
                && p.mutual.len() == tokens
                && p.indices.iter().all(|&v| v < tokens)
                && p.coordinates.iter().all(|xy| xy
                    .iter()
                    .all(|v| v.is_finite() && (0. ..=(grid[0] - 1) as f64).contains(v))),
            "invalid pose readout"
        );
        let a = images.images.get(&pair.target).context("target image")?;
        let b = images
            .images
            .get(&pair.reference)
            .context("reference image")?;
        let mut points = Vec::new();
        let mut pixels = Vec::new();
        for i in 0..tokens {
            if !p.mutual[i] {
                continue;
            }
            let x = original_pixel(
                [(i % grid[1]) as f64, (i / grid[1]) as f64],
                p.grid,
                a.original_hw,
            );
            let y = original_pixel(p.coordinates[i], p.grid, b.original_hw);
            points.push([
                [(x[0] - cx) / fx, (x[1] - cy) / fy],
                [(y[0] - cx) / fx, (y[1] - cy) / fy],
            ]);
            pixels.push([x, y]);
        }
        let fit = estimate(&points, &c.solver)?;
        // Validate label rotations even if the solver fails.
        camera::rotation_degrees(gt.rotation, gt.rotation)?;
        ensure!(
            gt.translation.iter().all(|v| v.is_finite()),
            "nonfinite pose truth"
        );
        let baseline = gt.translation.iter().map(|v| v * v).sum::<f64>().sqrt();
        let rotation = if let Some(p) = &fit.pose {
            camera::rotation_degrees(p.rotation, gt.rotation)?
        } else {
            180.
        };
        let translation = if baseline < c.minimum_baseline_meters || baseline < 1e-8 {
            None
        } else {
            Some(if let Some(p) = &fit.pose {
                camera::translation_degrees(p.translation, gt.translation)?.unwrap()
            } else {
                180.
            })
        };
        if p.method == "spatial_residual_conditional_local" && example_ids.contains(&p.pair) {
            let selected=(0..8.min(pixels.len())).map(|k|k*(pixels.len()-1)/7.min(pixels.len().saturating_sub(1)).max(1)).map(|i|json!({"target":pixels[i][0],"reference":pixels[i][1],"ransac_inlier":fit.inliers[i]})).collect::<Vec<_>>();
            examples.push(json!({"pair":pair,"images":{"target":a,"reference":b},"matches":selected,"rotation_degrees":rotation,"translation_degrees":translation,"success":fit.pose.is_some(),"failure":fit.failure,"selection":"first/middle/last lexicographic pair per sequence; up to eight uniformly spaced mutual matches, no quality filtering"}));
        }
        rows.push(PoseRow {
            pair: p.pair,
            sequence: pair.sequence.clone(),
            interval: pair.interval,
            method: p.method,
            mutual_matches: points.len(),
            fit,
            rotation_degrees: rotation,
            translation_degrees: translation,
            pose_degrees: translation.map(|t| t.max(rotation)),
            baseline_meters: baseline,
        });
    }
    let methods = summarize(&rows, &c.methods)?;
    let mut contrasts = Vec::new();
    let candidate = methods
        .get("spatial_residual_conditional_local")
        .context("missing pair control")?;
    for name in c
        .methods
        .iter()
        .filter(|n| n.as_str() != "spatial_residual_conditional_local")
    {
        let control = &methods[name];
        let gains = candidate
            .sequences
            .iter()
            .map(|(seq, s)| {
                Ok((
                    seq.clone(),
                    s.pose_auc_10.context("no pose eligible pairs")?
                        - control.sequences[seq]
                            .pose_auc_10
                            .context("no control pose pairs")?,
                ))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let passed = gains.values().all(|v| *v > 0.)
            && candidate.macro_success_fraction >= control.macro_success_fraction;
        contrasts.push(json!({"candidate":"spatial_residual_conditional_local","control":name,"sequence_auc10_gains":gains,"macro_auc10_gain":candidate.macro_pose_auc_10.unwrap()-control.macro_pose_auc_10.unwrap(),"passed":passed,"gate":"positive AUC@10 gain in every sequence and no macro solver-success regression"}));
    }
    let report = PoseReport {
        schema: 1,
        image_size: images.image_size,
        checkpoint_sha256: c.checkpoint_sha256.clone(),
        evaluation_use: images.evaluation_use,
        protocol: format!(
            "TUM Freiburg 3 calibrated relative-motion diagnostic. RGB-only {}px model, fixed local centroid and mutual matches; known intrinsics supplied only to deterministic CPU eight-point essential RANSAC. Equal pair weight within sequence, then equal sequences. Failed fits contribute 180 degrees; ground-truth baselines below {} cm excluded from translation/pose only. Not a learned camera head or official SLAM/relative-pose protocol.",
            images.image_size,
            100. * c.minimum_baseline_meters
        ),
        config: c.clone(),
        inputs,
        rows,
        methods,
        contrasts,
        examples,
    };
    write_json(&c.output, &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn solver_failures_count_in_pose_metrics_and_low_baselines_are_explicit() {
        let row = PoseRow {
            pair: "a".into(),
            sequence: "s".into(),
            interval: 15,
            method: "m".into(),
            mutual_matches: 0,
            fit: PoseFit {
                pose: None,
                failure: Some("insufficient_correspondences".into()),
                trials: 0,
                inliers: vec![],
                positive_depth_points: 0,
            },
            rotation_degrees: 180.,
            translation_degrees: Some(180.),
            pose_degrees: Some(180.),
            baseline_meters: 0.1,
        };
        let mut low = row.clone();
        low.pair = "b".into();
        low.baseline_meters = 0.;
        low.translation_degrees = None;
        low.pose_degrees = None;
        let out = summarize(&[row.clone(), low], &["m".into()]).unwrap();
        let s = &out["m"].sequences["s"];
        assert_eq!(s.pairs, 2);
        assert_eq!(s.pose_pairs, 1);
        assert_eq!(s.excluded_low_baseline, 1);
        assert_eq!(s.pose_auc_10, Some(0.));
        assert_eq!(s.mean_rotation_degrees, 180.);
        assert!(summarize(&[row.clone(), row], &["m".into()]).is_err());
    }
}
