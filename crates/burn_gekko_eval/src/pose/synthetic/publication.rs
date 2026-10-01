//! Single-checkpoint camera capability from the complete fixed solver-seed panel.
use super::Config;
use crate::{
    pose::{
        benchmark::summary,
        synthetic_comparison::{self, Arm, Population},
    },
    schema::{Capability, CapabilityStatus, Metric},
};
use anyhow::{Context, Result, ensure};
use burn_gekko_data::{sha256_file, write_json};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::PathBuf};

fn measurements(rows: &Population, seeds: &[u64]) -> Result<([f64; 3], [f64; 2])> {
    ensure!(!seeds.is_empty(), "missing solver seeds");
    let mut totals = [0.; 3];
    let mut range = [1f64, 0f64];
    for seed in seeds {
        let selected = rows
            .iter()
            .filter(|((s, _), _)| s == seed)
            .map(|(_, r)| r)
            .collect::<Vec<_>>();
        let measured = summary(&selected)?;
        let auc = measured.pose_auc_10.context("no eligible pose rooms")?;
        range[0] = range[0].min(auc);
        range[1] = range[1].max(auc);
        for (sum, value) in totals.iter_mut().zip([
            auc,
            measured.mean_rotation_degrees,
            measured
                .mean_translation_degrees
                .context("no eligible translations")?,
        ]) {
            *sum += value / seeds.len() as f64;
        }
    }
    Ok((totals, range))
}

pub fn publish(c: &Config) -> Result<Value> {
    let output = c.output.with_extension("head.json");
    ensure!(
        !output.exists(),
        "preserve existing synthetic camera capability"
    );
    let report_hash = sha256_file(&c.output)?;
    let report: Value = serde_json::from_slice(&fs::read(&c.output)?)?;
    ensure!(
        report["config"] == serde_json::to_value(c)?
            && report["checkpoint_sha256"] == c.checkpoint_sha256,
        "synthetic report/config/checkpoint mismatch"
    );
    let mut sources: BTreeMap<PathBuf, String> = serde_json::from_value(report["inputs"].clone())?;
    ensure!(
        sources.get(&c.predictions) == Some(&c.predictions_sha256),
        "missing RGB prediction pin"
    );
    sources.insert(c.output.clone(), report_hash.clone());
    let mut metrics = Vec::new();
    let mut limitations=vec![
        "Synthetic reused development rooms. All declared solver seeds contribute; repeated solver seeds are not independent images or training runs. Failed fits retain 180-degree errors. Low baselines are excluded from translation and joint pose only.".into(),
        "Known intrinsics enter a calibrated eight-point solver after RGB-only inference. These metrics do not evaluate predicted intrinsics or a learned calibration head. They do not establish real-view transfer or SotA.".into(),
    ];
    for method in &c.methods {
        let (_, rows) = synthetic_comparison::load(&Arm {
            report: c.output.clone(),
            sha256: report_hash.clone(),
            method: method.clone(),
        })?;
        let (values, range) = measurements(&rows, &c.seeds)?;
        ensure!(
            (values[0]
                - report["methods"][method]["mean_pose_auc_10"]
                    .as_f64()
                    .context("recorded pose AUC")?)
            .abs()
                < 1e-12,
            "pose aggregate differs from complete panel"
        );
        for (field, value) in ["minimum_pose_auc_10", "maximum_pose_auc_10"]
            .into_iter()
            .zip(range)
        {
            ensure!(
                (report["methods"][method][field]
                    .as_f64()
                    .context("seed AUC range")?
                    - value)
                    .abs()
                    < 1e-12,
                "seed range differs from complete panel"
            );
        }
        let label = match method.as_str() {
            "spatial_pair_local" => "Pair-conditioned",
            "spatial_self_local" => "Same-image",
            "spatial_encoder_local" => "Encoder",
            _ => method,
        };
        for ((field, title, unit, lower), value) in [
            ("auc10", "Pose AUC at 10 degrees", "fraction", false),
            ("rotation_degrees", "Rotation error", "degrees", true),
            (
                "translation_degrees",
                "Signed translation-direction error",
                "degrees",
                true,
            ),
        ]
        .into_iter()
        .zip(values)
        {
            metrics.push(Metric{id:format!("{method}_{field}"),label:format!("{label}: {title}"),value,unit:unit.into(),lower_is_better:lower,samples:c.rooms,aggregation:format!("equal rooms within each of {} solver seeds, then equal seeds; failures retained; count is unique rooms, not solver repeats",c.seeds.len())});
        }
        limitations.push(format!("{label} pose AUC at 10 degrees spans {:.2}% to {:.2}% over all {} declared solver seeds; the table reports their mean.",100.*range[0],100.*range[1],c.seeds.len()));
    }
    let capability = Capability {
        id: "synthetic_calibrated_camera".into(),
        label: "Synthetic calibrated camera recovery".into(),
        status: CapabilityStatus::Evaluated,
        protocol: report["protocol"]
            .as_str()
            .context("camera protocol")?
            .into(),
        limitations,
        metrics,
    };
    capability.validate()?;
    let result = json!({"schema":1,"checkpoint_sha256":c.checkpoint_sha256,"capability":capability,"sources":sources});
    write_json(&output, &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::{
        benchmark::PoseRow,
        solver::{PoseFit, RelativePose},
    };
    #[test]
    fn failed_fits_and_every_declared_solver_seed_contribute() {
        let row = |error| PoseRow {
            pair: "room".into(),
            sequence: "fixture".into(),
            interval: 1,
            method: "fixture".into(),
            mutual_matches: 0,
            fit: PoseFit {
                pose: (error == 0.).then_some(RelativePose {
                    rotation: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                    translation: [1., 0., 0.],
                }),
                failure: (error != 0.).then_some("fixture".into()),
                trials: 0,
                inliers: vec![],
                positive_depth_points: 0,
            },
            rotation_degrees: error,
            translation_degrees: Some(error),
            pose_degrees: Some(error),
            baseline_meters: 0.25,
        };
        let rows = BTreeMap::from([
            ((7, "room".into()), row(180.)),
            ((8, "room".into()), row(0.)),
        ]);
        assert_eq!(
            measurements(&rows, &[7, 8]).unwrap(),
            ([0.5, 90., 90.], [0., 1.])
        );
        assert!(measurements(&rows, &[7, 9]).is_err());
        assert!(measurements(&rows, &[]).is_err());
    }
}
