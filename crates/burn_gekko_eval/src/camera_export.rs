//! Turn post-inference camera prediction/label records into an extensible head report.
use crate::{
    camera::{CameraRecord, evaluate},
    schema::{Capability, CapabilityStatus, Metric},
};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeSet, fs, path::PathBuf};
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraScoreConfig {
    pub records: PathBuf,
    pub records_sha256: String,
    pub provenance: PathBuf,
    pub provenance_sha256: String,
    pub checkpoint_sha256: String,
    pub evaluation_use: String,
    pub input_contract: String,
    pub coordinate_frame: String,
    pub output: PathBuf,
}
pub fn score(c: &CameraScoreConfig) -> Result<()> {
    ensure!(!c.output.exists(), "preserve existing camera evaluation");
    ensure!(
        matches!(c.evaluation_use.as_str(), "development" | "held_out")
            && !c.input_contract.is_empty()
            && !c.coordinate_frame.is_empty(),
        "camera input/frame/evaluation contracts required"
    );
    ensure!(
        burn_gekko_data::sha256_file(&c.records)? == c.records_sha256
            && burn_gekko_data::sha256_file(&c.provenance)? == c.provenance_sha256,
        "camera artifact checksum mismatch"
    );
    let provenance: Value = serde_json::from_slice(&fs::read(&c.provenance)?)?;
    ensure!(
        provenance["checkpoint_sha256"] == c.checkpoint_sha256,
        "camera checkpoint mismatch"
    );
    ensure!(
        provenance["input_contract"] == c.input_contract
            && provenance["coordinate_frame"] == c.coordinate_frame,
        "camera protocol differs from inference provenance"
    );
    let rows = fs::read_to_string(&c.records)?
        .lines()
        .map(|line| Ok(serde_json::from_str::<CameraRecord>(line)?))
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        rows.iter()
            .map(|r| &r.sample)
            .collect::<BTreeSet<_>>()
            .len()
            == rows.len(),
        "duplicate camera observation"
    );
    let result = evaluate(&rows)?;
    let metric = |id: &str, label: &str, value: f64, unit: &str, lower: bool, n: usize| Metric {
        id: id.into(),
        label: label.into(),
        value,
        unit: unit.into(),
        lower_is_better: lower,
        samples: n,
        aggregation:
            "equal camera-pair weighting; zero truth baselines excluded from translation and pose"
                .into(),
    };
    let mut metrics = vec![
        metric(
            "rotation_degrees",
            "Relative rotation error",
            result.rotation_mean_degrees,
            "degrees",
            true,
            result.samples,
        ),
        metric(
            "focal_relative_error",
            "Normalized focal relative error",
            result.focal_mean_relative_error,
            "fraction",
            true,
            result.samples,
        ),
    ];
    if let Some(t) = result.translation_mean_degrees {
        metrics.push(metric(
            "translation_degrees",
            "Signed translation-direction error",
            t,
            "degrees",
            true,
            result.nonzero_baselines,
        ));
    }
    for (id, label, value) in [
        ("pose_auc_5", "Pose AUC @ 5 degrees", result.pose_auc_5),
        ("pose_auc_10", "Pose AUC @ 10 degrees", result.pose_auc_10),
        ("pose_auc_20", "Pose AUC @ 20 degrees", result.pose_auc_20),
    ] {
        if let Some(value) = value {
            metrics.push(metric(
                id,
                label,
                value,
                "AUC",
                false,
                result.nonzero_baselines,
            ));
        }
    }
    let capability = Capability {
        id: "camera".into(),
        label: "Camera intrinsics and relative pose".into(),
        status: CapabilityStatus::Evaluated,
        protocol: format!(
            "{}; {}; frame {}; fx/width, fy/height; centered principal point",
            c.evaluation_use, c.input_contract, c.coordinate_frame
        ),
        limitations: vec![format!(
            "{} / {} pairs have nonzero truth baselines. Translation scale is not evaluated; direction is signed.",
            result.nonzero_baselines, result.samples
        )],
        metrics,
    };
    capability.validate()?;
    if let Some(parent) = c.output.parent() {
        fs::create_dir_all(parent)?;
    }
    // Head schema is deliberately shared with future depth/semantic/motion decoders.
    burn_gekko_data::write_json(
        &c.output,
        &json!({"schema":1,"checkpoint_sha256":c.checkpoint_sha256,"capability":capability}),
    )?;
    let sidecar = c.output.with_extension("provenance.json");
    ensure!(!sidecar.exists(), "camera provenance output exists");
    burn_gekko_data::write_json(
        &sidecar,
        &json!({"config":c,"metrics":result,"prediction_provenance":provenance,"observations":rows.len(),"coordinate_frame":provenance["coordinate_frame"].as_str().context("camera frame")?}),
    )?;
    Ok(())
}
