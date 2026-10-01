//! Room-cluster uncertainty and training-label controls for audited output heads.
use super::{HeadRow, audit, camera_score};
use crate::{
    adaptation::Input,
    schema::{Capability, CapabilityStatus, Metric},
    statistics::{Interval, bootstrap_mean},
};
use anyhow::{Context, Result, ensure};
use burn_gekko_data::{Split, head_cache::HeadCache, sha256_file, write_json};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, path::PathBuf};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub evidence: Input,
    pub checkpoint_sha256: String,
    pub expected_updates: usize,
    pub expected_training_rooms: usize,
    pub expected_validation_rooms: usize,
    pub output: PathBuf,
}

fn interval(observations: &[(u64, f64)]) -> Result<Interval> {
    let mut rooms: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
    for &(room, value) in observations {
        rooms.entry(room).or_default().push(value);
    }
    ensure!(
        rooms.len() >= 2,
        "head uncertainty needs at least two rooms"
    );
    bootstrap_mean(
        &rooms
            .values()
            .map(|v| v.iter().sum::<f64>() / v.len() as f64)
            .collect::<Vec<_>>(),
        859,
    )
}

fn measurements(rows: &[HeadRow], cache: &HeadCache) -> Result<BTreeMap<String, Interval>> {
    // Fit the constant only on training labels; validation is scoring only.
    let constant = audit::constant_prediction(cache)?;
    let labels: BTreeMap<_, _> = cache
        .samples
        .iter()
        .filter(|s| s.split == Split::Validation)
        .map(|s| ((s.room_seed, s.target_view), &s.camera))
        .collect();
    let mut observations: BTreeMap<String, Vec<(u64, f64)>> = BTreeMap::new();
    for row in rows {
        let baseline = camera_score(
            &constant,
            labels
                .get(&(row.room_seed, row.target_view))
                .context("head room outside validation")?,
        )?;
        let mut add = |name: &str, value| {
            observations
                .entry(name.into())
                .or_default()
                .push((row.room_seed, value));
        };
        add("rotation_degrees", row.camera.rotation_degrees);
        add(
            "focal_error_percent",
            row.camera.focal_relative_error * 100.,
        );
        add(
            "rotation_reduction_vs_constant_degrees",
            baseline.rotation_degrees - row.camera.rotation_degrees,
        );
        add(
            "focal_reduction_vs_constant_percentage_points",
            (baseline.focal_relative_error - row.camera.focal_relative_error) * 100.,
        );
        if let (Some(actual), Some(control)) =
            (row.camera.translation_degrees, baseline.translation_degrees)
        {
            add("translation_degrees", actual);
            add(
                "translation_reduction_vs_constant_degrees",
                control - actual,
            );
        }
        // Perfect predictions have infinite PSNR represented as None. Do not
        // silently drop those views and change the uncertainty population.
        let rgb = row
            .rgb
            .psnr_db
            .context("finite RGB PSNR required for this diagnostic")?;
        let mono = row
            .monocular
            .psnr_db
            .context("finite monocular PSNR required for this diagnostic")?;
        add("rgb_psnr_db", rgb);
        add("monocular_psnr_db", mono);
        add("rgb_reference_gain_db", rgb - mono);
    }
    observations
        .into_iter()
        .map(|(name, rows)| Ok((name, interval(&rows)?)))
        .collect()
}

pub fn analyze(c: &Config) -> Result<Value> {
    ensure!(
        !c.output.exists() && !c.output.with_extension("head.json").exists(),
        "preserve head diagnostic outputs"
    );
    let mut sources = BTreeMap::new();
    let audited = audit::load(&c.evidence, &c.checkpoint_sha256, &mut sources)?;
    let train_rooms = audited
        .cache
        .samples
        .iter()
        .filter(|s| s.split == Split::Train)
        .map(|s| s.room_seed)
        .collect::<std::collections::BTreeSet<_>>();
    ensure!(
        audited.report["starting_step"] == 0
            && audited.report["stop_reason"] == "step_limit"
            && audited.steps.len() == c.expected_updates
            && train_rooms.len() == c.expected_training_rooms
            && audited.scores.rooms == c.expected_validation_rooms,
        "head study horizon or population differs from registration"
    );
    let intervals = measurements(&audited.scores.rows, &audited.cache)?;
    let mut paired_metrics = Vec::new();
    for (id, label, unit) in [
        ("rgb_reference_gain_db", "RGB benefit from references", "dB"),
        (
            "rotation_reduction_vs_constant_degrees",
            "Rotation error reduction vs training-label constant",
            "degrees",
        ),
        (
            "translation_reduction_vs_constant_degrees",
            "Signed direction error reduction vs training-label constant",
            "degrees",
        ),
        (
            "focal_reduction_vs_constant_percentage_points",
            "Focal error reduction vs training-label constant",
            "percentage points",
        ),
    ] {
        if let Some(value) = intervals.get(id) {
            for (suffix, label_suffix, number) in [
                ("", "", value.mean),
                ("_lower95", " (lower 95% bound)", value.low),
            ] {
                paired_metrics.push(Metric {
                    id: format!("{id}{suffix}"), label: format!("{label}{label_suffix}"),
                    value: number, unit: unit.into(), lower_is_better: false,
                    samples: value.clusters,
                    aggregation: "equal room means; paired views within each room; 10000 whole-room bootstrap replicates".into(),
                });
            }
        }
    }
    let result = json!({
        "schema":1, "checkpoint_sha256":c.checkpoint_sha256,
        "training_rooms":train_rooms.len(), "validation_rooms":audited.scores.rooms,
        "validation_targets":audited.scores.targets, "updates":audited.steps.len(),
        "stability_gates":audited.report["stability_gates"],
        "intervals":intervals, "sources":sources,
        "scope":"Fixed endpoint on reused development rooms. Means weight rooms equally and bootstrap entire rooms, retaining paired predictions and all camera failures. Positive reductions favor the learned head. Camera constants use training labels only. These intervals do not measure training-seed variation, independent transfer, image sharpness or SotA. Infinite PSNR is rejected rather than silently omitted."
    });
    write_json(&c.output, &result)?;
    sources.insert(c.output.clone(), sha256_file(&c.output)?);
    let capability = Capability {
        id:"head_generalization_diagnostics".into(), label:"Output-head paired development uncertainty".into(),
        status:CapabilityStatus::Evaluated,
        protocol:"All RGB/camera observations rescored from checksum-bound raw predictions. Positive gains favor reference use or learned calibration; lower 95% bounds use a room-cluster bootstrap.".into(),
        limitations:vec![format!("{} training and {} validation rooms; one fixed head-training seed. Validation is reused development. Numerical stability and positive control gains alone do not establish accurate absolute calibration or sharp RGB completion.",train_rooms.len(),audited.scores.rooms)],
        metrics:paired_metrics,
    };
    capability.validate()?;
    write_json(
        &c.output.with_extension("head.json"),
        &json!({"schema":1,"checkpoint_sha256":c.checkpoint_sha256,"capability":capability,"source":sources}),
    )?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uncertainty_resamples_rooms_instead_of_repeated_views() {
        let sparse = interval(&[(1, 0.), (2, 10.)]).unwrap();
        let repeated = interval(&[(1, 0.), (1, 0.), (1, 0.), (2, 10.)]).unwrap();
        assert_eq!(sparse.mean, 5.);
        assert_eq!(sparse.clusters, 2);
        assert_eq!(
            serde_json::to_value(sparse).unwrap(),
            serde_json::to_value(repeated).unwrap()
        );
        assert!(interval(&[(1, 0.), (1, 10.)]).is_err());
        assert!(interval(&[(1, f64::NAN), (2, 0.)]).is_err());
    }
}
