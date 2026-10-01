//! Publish one completed diagnostic without repeating pose fitting.
use super::Config;
use crate::schema::{Capability, CapabilityStatus, Metric};
use anyhow::{Context, Result, ensure};
use burn_gekko_data::{sha256_file, write_json};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::PathBuf};

pub fn publish(c: &Config) -> Result<Value> {
    let path = c.output.join("summary.json");
    let output = c.output.join("head.json");
    ensure!(!output.exists(), "preserve existing diagnostic capability");
    let report: Value = serde_json::from_slice(&fs::read(&path)?)?;
    ensure!(
        report["schema"] == 1
            && report["status"] == "localization_and_consensus_diagnostic"
            && report["config"] == serde_json::to_value(c)?,
        "diagnostic report/config mismatch"
    );
    let mut sources: BTreeMap<PathBuf, String> = serde_json::from_value(report["sources"].clone())?;
    ensure!(
        sources.get(&c.original) == Some(&c.original_sha256),
        "missing original report pin"
    );
    for (p, expected) in &sources {
        ensure!(
            sha256_file(p)? == *expected,
            "diagnostic evidence changed: {}",
            p.display()
        );
    }
    sources.insert(path.clone(), sha256_file(&path)?);
    let mut metrics = Vec::new();
    for (method, label) in [
        ("spatial_residual_conditional_local", "Pair-conditioned"),
        ("spatial_self_conditional_local", "Same-image"),
        ("student_l06_centered_conditional_local", "Encoder"),
    ] {
        for (field, title, unit, lower) in [
            ("auc10", "Pose AUC at 10 degrees", "fraction", false),
            ("rotation_degrees", "Rotation error", "degrees", true),
            (
                "translation_degrees",
                "Translation-direction error",
                "degrees",
                true,
            ),
        ] {
            let value = report["methods"][format!("recorded/{method}")][field]["mean"]
                .as_f64()
                .context("missing diagnostic metric")?;
            metrics.push(Metric { id:format!("{method}_{field}"), label:format!("{label}: {title}"), value, unit:unit.into(), lower_is_better:lower, samples:c.seeds.len(), aggregation:"equal pairs within each sequence, then equal sequences and all solver seeds; failures included".into() });
        }
    }
    let points = c.minimal_solver.sample_size();
    let mut limitations = report["limitations"]
        .as_array()
        .context("diagnostic limitations")?
        .iter()
        .map(|v| v.as_str().map(str::to_owned).context("invalid limitation"))
        .collect::<Result<Vec<_>>>()?;
    limitations.push("This table shows the registered subpixel readout. All hard-coordinate ablations, seed ranges and failures remain in the pinned source reports. This is calibrated post-inference fitting, not a learned camera head or predicted intrinsics. Model and attached RGB-head weights are unchanged.".into());
    let capability = Capability {
        id: format!("pose_localization_{points}_point"),
        label: format!("Calibrated solver diagnostic / {points}-point hypotheses"),
        status: CapabilityStatus::Evaluated,
        protocol: report["protocol"]
            .as_str()
            .context("diagnostic protocol")?
            .into(),
        limitations,
        metrics,
    };
    capability.validate()?;
    let result = json!({"schema":1,"checkpoint_sha256":report["checkpoint_sha256"],"capability":capability,"sources":sources});
    write_json(&output, &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn publishing_binds_config_and_rejects_changed_upstream_evidence() {
        let root = tempfile::tempdir().unwrap();
        let original = root.path().join("original.json");
        fs::write(&original, b"frozen upstream evidence").unwrap();
        let c = Config {
            original: original.clone(),
            original_sha256: sha256_file(&original).unwrap(),
            seeds: vec![781, 782],
            minimal_solver: crate::pose::solver::MinimalSolver::FivePoint,
            output: root.path().into(),
        };
        let mut methods = BTreeMap::new();
        for method in [
            "spatial_residual_conditional_local",
            "spatial_self_conditional_local",
            "student_l06_centered_conditional_local",
        ] {
            methods.insert(format!("recorded/{method}"),json!({"auc10":{"mean":0.1},"rotation_degrees":{"mean":5.},"translation_degrees":{"mean":30.}}));
        }
        let report = json!({"schema":1,"status":"localization_and_consensus_diagnostic","config":c,"checkpoint_sha256":"checkpoint","sources":{original.to_str().unwrap():c.original_sha256},"methods":methods,"protocol":"fixture","limitations":["synthetic fixture"]});
        write_json(&root.path().join("summary.json"), &report).unwrap();
        let output = publish(&c).unwrap();
        assert_eq!(output["capability"]["metrics"].as_array().unwrap().len(), 9);
        assert_eq!(output["sources"].as_object().unwrap().len(), 2);
        fs::remove_file(root.path().join("head.json")).unwrap();
        fs::write(&original, b"changed").unwrap();
        assert!(
            publish(&c)
                .unwrap_err()
                .to_string()
                .contains("evidence changed")
        );
        assert!(!root.path().join("head.json").exists());
    }
}
