//! Repeat an immutable pose export under declared solver seeds, without model inference.
use super::benchmark::{PoseReport, score};
use anyhow::{Context, Result, ensure};
use burn_gekko_data::{sha256_file, write_json};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StabilityConfig {
    pub original: PathBuf,
    pub original_sha256: String,
    pub seeds: Vec<u64>,
    pub output: PathBuf,
}

fn range(values: &[f64]) -> Result<Value> {
    ensure!(
        !values.is_empty() && values.iter().all(|x| x.is_finite()),
        "invalid stability population"
    );
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    Ok(
        json!({"mean":mean,"minimum":values.iter().copied().reduce(f64::min).unwrap(),"maximum":values.iter().copied().reduce(f64::max).unwrap(),"population_stddev":(values.iter().map(|x|(x-mean).powi(2)).sum::<f64>() / values.len() as f64).sqrt(),"solver_seeds":values.len()}),
    )
}

pub fn analyze(c: &StabilityConfig) -> Result<Value> {
    ensure!(!c.output.exists(), "preserve existing stability study");
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
        "stability output outside .data"
    );
    let original: PoseReport = serde_json::from_slice(&fs::read(&c.original)?)?;
    ensure!(
        original.evaluation_use == "development",
        "solver diagnosis requires development classification"
    );
    ensure!(
        c.seeds.contains(&original.config.solver.seed),
        "retain the original solver seed"
    );
    let expected_pairs = original
        .rows
        .iter()
        .map(|r| (&r.pair, &r.method))
        .collect::<BTreeSet<_>>();
    ensure!(
        expected_pairs.len() == original.rows.len(),
        "duplicate original pose identity"
    );
    let mut config_sources = BTreeMap::from([(c.original.clone(), c.original_sha256.clone())]);
    for (path, hash) in &original.inputs {
        ensure!(sha256_file(path)? == *hash, "original pose input changed");
        config_sources.insert(path.clone(), hash.clone());
    }
    fs::create_dir_all(&c.output)?;
    let mut reports = Vec::new();
    for &seed in &c.seeds {
        let mut config = original.config.clone();
        config.solver.seed = seed;
        config.output = c.output.join(format!("seed-{seed}.json"));
        let result = score(&config)?;
        ensure!(
            result
                .rows
                .iter()
                .map(|r| (&r.pair, &r.method))
                .collect::<BTreeSet<_>>()
                == expected_pairs,
            "solver seed changed pair population"
        );
        if seed == original.config.solver.seed {
            // Repeat the same seed before interpreting any variability. JSON
            // float round trips permit 1e-12, far below any reported precision.
            for (a, b) in original.rows.iter().zip(&result.rows) {
                ensure!(
                    a.pair == b.pair
                        && a.method == b.method
                        && a.fit.inliers == b.fit.inliers
                        && a.fit.trials == b.fit.trials
                        && (a.rotation_degrees - b.rotation_degrees).abs() < 1e-12
                        && match (a.translation_degrees, b.translation_degrees) {
                            (Some(a), Some(b)) => (a - b).abs() < 1e-12,
                            (None, None) => true,
                            _ => false,
                        },
                    "original solver replay differs"
                );
            }
        }
        config_sources.insert(config.output.clone(), sha256_file(&config.output)?);
        reports.push(result);
    }
    let mut methods = BTreeMap::new();
    for name in &original.config.methods {
        let observations = reports.iter().map(|r| &r.methods[name]).collect::<Vec<_>>();
        let mut sequences = BTreeMap::new();
        for sequence in original.methods[name].sequences.keys() {
            let values = observations
                .iter()
                .map(|m| {
                    m.sequences[sequence]
                        .pose_auc_10
                        .context("pose AUC missing")
                })
                .collect::<Result<Vec<_>>>()?;
            sequences.insert(sequence.clone(), range(&values)?);
        }
        let auc = observations
            .iter()
            .map(|m| m.macro_pose_auc_10.context("pose AUC missing"))
            .collect::<Result<Vec<_>>>()?;
        let recall = observations
            .iter()
            .map(|m| m.macro_pose_recall_10.context("pose recall missing"))
            .collect::<Result<Vec<_>>>()?;
        methods.insert(
            name.clone(),
            json!({"auc10":range(&auc)?,"recall10":range(&recall)?,"sequence_auc10":sequences}),
        );
    }
    let mut contrasts = Vec::new();
    for (i, contrast) in original.contrasts.iter().enumerate() {
        let observations = reports
            .iter()
            .map(|r| r.contrasts.get(i).context("pose contrast missing"))
            .collect::<Result<Vec<_>>>()?;
        ensure!(
            observations
                .iter()
                .all(|o| o["candidate"] == contrast["candidate"]
                    && o["control"] == contrast["control"]
                    && o["gate"] == contrast["gate"]),
            "pose contrast protocol changed"
        );
        let gains = observations
            .iter()
            .map(|v| v["macro_auc10_gain"].as_f64().context("pose gain missing"))
            .collect::<Result<Vec<_>>>()?;
        let mut sequence_gains = BTreeMap::new();
        for sequence in contrast["sequence_auc10_gains"]
            .as_object()
            .context("sequence gains")?
            .keys()
        {
            let values = observations
                .iter()
                .map(|v| {
                    v["sequence_auc10_gains"][sequence]
                        .as_f64()
                        .context("sequence gain missing")
                })
                .collect::<Result<Vec<_>>>()?;
            sequence_gains.insert(sequence.clone(), range(&values)?);
        }
        contrasts.push(json!({"candidate":contrast["candidate"],"control":contrast["control"],"gate":contrast["gate"],"original_passed":contrast["passed"],"passing_seeds":observations.iter().filter(|v|v["passed"]==true).count(),"total_seeds":c.seeds.len(),"macro_auc10_gain":range(&gains)?,"sequence_auc10_gains":sequence_gains}));
    }
    let report = json!({"schema":1,"checkpoint_sha256":original.checkpoint_sha256,"status":"solver_seed_sensitivity_diagnostic","protocol":"Frozen RGB matches, camera labels, solver policy and all pairs. Only the declared RANSAC RNG seeds change; controls receive every seed. Original-seed replay checked. No best-seed selection, model fitting or new generalization evidence. Population standard deviation describes solver randomness, not uncertainty across training seeds or scenes.","config":c,"methods":methods,"contrasts":contrasts,"sources":config_sources});
    write_json(&c.output.join("summary.json"), &report)?;
    let mut metrics = Vec::new();
    for name in &original.config.methods {
        for (id, label) in [
            ("mean", "Mean pose AUC at 10 degrees"),
            ("minimum", "Minimum pose AUC across solver seeds"),
            ("maximum", "Maximum pose AUC across solver seeds"),
        ] {
            metrics.push(crate::schema::Metric {id:format!("{name}_{id}"),label:format!("{name}: {label}"),value:report["methods"][name]["auc10"][id].as_f64().context("solver stability metric")?,unit:"fraction".into(),lower_is_better:false,samples:c.seeds.len(),aggregation:"equal solver seeds; each seed retains the full original pair/sequence macro protocol".into()});
        }
    }
    let mut limitations=vec!["Solver seeds are not independently trained models. The minimum/maximum is a sensitivity range, not a confidence interval. Original benchmark gates remain unchanged; no seed is selected for publication.".into()];
    for contrast in &report["contrasts"].as_array().unwrap()[..] {
        limitations.push(format!(
            "All-sequence transfer against {} passed {}/{} solver seeds.",
            contrast["control"].as_str().unwrap(),
            contrast["passing_seeds"],
            contrast["total_seeds"]
        ));
    }
    let capability = crate::schema::Capability {
        id: format!(
            "pose_solver_stability_{}",
            original.config.solver.max_trials
        ),
        label: format!(
            "Camera solver sensitivity / at most {} trials",
            original.config.solver.max_trials
        ),
        status: crate::schema::CapabilityStatus::Evaluated,
        protocol: report["protocol"].as_str().unwrap().into(),
        limitations,
        metrics,
    };
    capability.validate()?;
    write_json(
        &c.output.join("head.json"),
        &json!({"schema":1,"checkpoint_sha256":original.checkpoint_sha256,"capability":capability}),
    )?;
    Ok(report)
}
