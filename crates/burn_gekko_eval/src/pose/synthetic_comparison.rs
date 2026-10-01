//! Paired room-level analysis of fixed synthetic pose exports and solver panels.
use super::benchmark::PoseRow;
use crate::{camera, statistics::bootstrap_mean};
use anyhow::{Context, Result, ensure};
use burn_gekko_data::{sha256_file, write_json};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Arm {
    pub report: PathBuf,
    pub sha256: String,
    pub method: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub candidate: Arm,
    pub control: Arm,
    pub output: PathBuf,
}
pub(crate) type Population = BTreeMap<(u64, String), PoseRow>;

fn complete_methods(
    rows: &[PoseRow],
    methods: &[String],
    rooms: usize,
) -> Result<BTreeSet<String>> {
    let expected = methods.iter().map(String::as_str).collect::<BTreeSet<_>>();
    ensure!(
        rooms > 0 && !expected.is_empty() && expected.len() == methods.len(),
        "invalid declared method panel"
    );
    let mut populations: BTreeMap<&str, BTreeSet<String>> = BTreeMap::new();
    for row in rows {
        ensure!(
            expected.contains(row.method.as_str()),
            "undeclared pose readout"
        );
        ensure!(
            populations
                .entry(&row.method)
                .or_default()
                .insert(row.pair.clone()),
            "duplicate method/room in solver panel"
        );
    }
    ensure!(
        populations.len() == expected.len(),
        "missing declared control readout"
    );
    let population = populations.values().next().unwrap();
    ensure!(
        population.len() == rooms && populations.values().all(|v| v == population),
        "incomplete or mismatched control room population"
    );
    Ok(population.clone())
}

pub(crate) fn load(a: &Arm) -> Result<(Value, Population)> {
    ensure!(
        sha256_file(&a.report)? == a.sha256,
        "pose comparison hash differs"
    );
    let v: Value = serde_json::from_slice(&fs::read(&a.report)?)?;
    ensure!(
        v["schema"] == 1 && v["status"] == "development_diagnostic",
        "unsupported pose comparison"
    );
    for (path, hash) in v["inputs"].as_object().context("pose input closure")? {
        ensure!(
            sha256_file(&PathBuf::from(path))? == hash.as_str().context("source hash")?,
            "pose input changed"
        );
    }
    let mut rows = Population::new();
    let config: super::synthetic::Config = serde_json::from_value(v["config"].clone())?;
    ensure!(
        v["rooms"] == config.rooms && config.methods.contains(&a.method),
        "pose cohort declaration differs"
    );
    let mut seen_seeds = BTreeSet::new();
    let mut room_panel = None;
    for seed in v["seeds"].as_array().context("solver seeds")? {
        let id = seed["seed"].as_u64().context("seed identity")?;
        ensure!(seen_seeds.insert(id), "duplicate solver seed");
        let parsed: Vec<PoseRow> = serde_json::from_value(seed["rows"].clone())?;
        let population = complete_methods(&parsed, &config.methods, config.rooms)?;
        if let Some(expected) = &room_panel {
            ensure!(
                *expected == population,
                "room population changes across solver seeds"
            );
        } else {
            room_panel = Some(population);
        }
        let selected = parsed
            .into_iter()
            .filter(|r| r.method == a.method)
            .collect::<Vec<_>>();
        ensure!(
            selected.len() == config.rooms,
            "incomplete declared room population"
        );
        let recomputed = super::benchmark::summary(&selected.iter().collect::<Vec<_>>())?;
        let reported: super::benchmark::PoseSummary =
            serde_json::from_value(seed["methods"][&a.method].clone())?;
        // JSON floating-point round trips may differ by a final bit; counters and
        // eligibility are still exact, AUC is checked far below display precision.
        ensure!(
            recomputed.pairs == reported.pairs
                && recomputed.successes == reported.successes
                && recomputed.pose_pairs == reported.pose_pairs
                && recomputed
                    .pose_auc_10
                    .zip(reported.pose_auc_10)
                    .is_some_and(|(a, b)| (a - b).abs() < 1e-12),
            "pose summary disagrees with rows"
        );
        for row in selected {
            ensure!(
                rows.insert((id, row.pair.clone()), row).is_none(),
                "duplicate room/solver seed"
            );
        }
    }
    ensure!(
        !rows.is_empty()
            && seen_seeds == config.seeds.iter().copied().collect()
            && seen_seeds.len() == config.seeds.len(),
        "incomplete declared solver panel"
    );
    Ok((v, rows))
}

pub(crate) fn paired(candidate: &Population, control: &Population) -> Result<Value> {
    ensure!(
        candidate.keys().eq(control.keys()) && !candidate.is_empty(),
        "paired pose populations differ"
    );
    let mut rooms: BTreeMap<String, Vec<[f64; 3]>> = BTreeMap::new();
    let mut seeds: BTreeMap<u64, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for (key, a) in candidate {
        let b = &control[key];
        ensure!(
            a.baseline_meters == b.baseline_meters
                && a.translation_degrees.is_some() == b.translation_degrees.is_some(),
            "pose label/eligibility mismatch"
        );
        let (Some(ap), Some(bp)) = (a.pose_degrees, b.pose_degrees) else {
            continue;
        };
        ensure!(
            ap.is_finite() && bp.is_finite(),
            "nonfinite paired pose error"
        );
        rooms.entry(key.1.clone()).or_default().push([
            bp - ap,
            f64::from(ap <= 10.) - f64::from(bp <= 10.),
            b.translation_degrees.unwrap() - a.translation_degrees.unwrap(),
        ]);
        let pair = seeds.entry(key.0).or_default();
        pair.0.push(ap);
        pair.1.push(bp);
    }
    ensure!(
        !rooms.is_empty() && seeds.len() >= 2 && rooms.values().all(|v| v.len() == seeds.len()),
        "incomplete eligible room/seed panel"
    );
    let mut intervals = BTreeMap::new();
    for (i, name) in [
        "pose_error_reduction_degrees",
        "recall_at_10_gain",
        "translation_error_reduction_degrees",
    ]
    .into_iter()
    .enumerate()
    {
        let means = rooms
            .values()
            .map(|v| v.iter().map(|x| x[i]).sum::<f64>() / v.len() as f64)
            .collect::<Vec<_>>();
        intervals.insert(name, bootstrap_mean(&means, 887)?);
    }
    let mut rows = Vec::new();
    for (seed, (a, b)) in seeds {
        let ca = camera::pose_auc(&a, 10.)?;
        let co = camera::pose_auc(&b, 10.)?;
        rows.push(json!({"seed":seed,"candidate_auc10":ca,"control_auc10":co,"gain":ca-co}));
    }
    let mean =
        |key: &str| rows.iter().map(|r| r[key].as_f64().unwrap()).sum::<f64>() / rows.len() as f64;
    Ok(
        json!({"eligible_rooms":rooms.len(),"room_seed_observations":rooms.values().map(Vec::len).sum::<usize>(),
        "mean_candidate_auc10":mean("candidate_auc10"),"mean_control_auc10":mean("control_auc10"),"mean_auc10_gain":mean("gain"),
        "solver_seeds_with_positive_auc10_gain":rows.iter().filter(|r| r["gain"].as_f64().unwrap()>0.).count(),"solver_seeds":rows.len(),
        "paired_room_intervals":intervals,"per_seed":rows}),
    )
}

pub fn compare(c: &Config) -> Result<Value> {
    ensure!(!c.output.exists(), "preserve existing comparison");
    let (a, ar) = load(&c.candidate)?;
    let (b, br) = load(&c.control)?;
    let protocol = |v: &Value| -> Result<Value> {
        let mut p = v["config"].as_object().context("pose config")?.clone();
        for key in [
            "predictions",
            "predictions_sha256",
            "checkpoint_sha256",
            "output",
        ] {
            p.remove(key);
        }
        Ok(Value::Object(p))
    };
    ensure!(
        a["dataset_id"] == b["dataset_id"]
            && a["rooms"] == b["rooms"]
            && protocol(&a)? == protocol(&b)?,
        "pose protocols differ"
    );
    let mut result = paired(&ar, &br)?;
    let map = result.as_object_mut().unwrap();
    map.insert("schema".into(), json!(1));
    map.insert("candidate".into(),json!({"report":c.candidate.report,"sha256":c.candidate.sha256,"checkpoint_sha256":a["checkpoint_sha256"],"method":c.candidate.method}));
    map.insert("control".into(),json!({"report":c.control.report,"sha256":c.control.sha256,"checkpoint_sha256":b["checkpoint_sha256"],"method":c.control.method}));
    map.insert("scope".into(),json!("Retrospective development contrast. All fixed solver seeds retained; 95% paired bootstrap intervals resample rooms after averaging per-room gains across seeds. Repeated solver seeds are not independent rooms or training seeds. AUC differences describe this fixed panel; intervals here apply to mean angular error and recall, not AUC. No checkpoint selection or external-transfer claim."));
    write_json(&c.output, &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose::solver::PoseFit;
    fn population(error: f64) -> Population {
        (1..=2)
            .flat_map(|seed| {
                (0..3).map(move |room| {
                    (
                        (seed, room.to_string()),
                        PoseRow {
                            pair: room.to_string(),
                            sequence: "fixture".into(),
                            interval: 1,
                            method: "fixture".into(),
                            mutual_matches: 0,
                            fit: PoseFit {
                                pose: None,
                                failure: Some("fixture".into()),
                                trials: 0,
                                inliers: vec![],
                                positive_depth_points: 0,
                            },
                            rotation_degrees: error,
                            translation_degrees: Some(error),
                            pose_degrees: Some(error),
                            baseline_meters: 0.25,
                        },
                    )
                })
            })
            .collect()
    }
    #[test]
    fn uncertainty_clusters_rooms_not_repeated_solver_seeds() {
        let a = population(4.);
        let mut b = population(8.);
        let result = paired(&a, &b).unwrap();
        let ci = &result["paired_room_intervals"]["pose_error_reduction_degrees"];
        assert_eq!(ci["mean"], 4.);
        assert_eq!(ci["low"], 4.);
        assert_eq!(ci["clusters"], 3);
        assert_eq!(result["room_seed_observations"], 6);
        b.remove(&(2, "1".into()));
        assert!(paired(&a, &b).is_err());
    }

    #[test]
    fn complete_primary_readout_cannot_hide_a_missing_or_different_control() {
        let primary = population(4.)
            .into_iter()
            .filter(|((seed, _), _)| *seed == 1)
            .map(|(_, row)| row)
            .collect::<Vec<_>>();
        let mut rows = primary.clone();
        rows.extend(primary.into_iter().map(|mut r| {
            r.method = "control".into();
            r
        }));
        let methods = vec!["fixture".into(), "control".into()];
        assert_eq!(complete_methods(&rows, &methods, 3).unwrap().len(), 3);
        let mut changed = rows.clone();
        changed.pop();
        assert!(complete_methods(&changed, &methods, 3).is_err());
        changed = rows.clone();
        changed.last_mut().unwrap().pair = "foreign-room".into();
        assert!(complete_methods(&changed, &methods, 3).is_err());
        changed = rows.clone();
        changed.last_mut().unwrap().method = "undeclared".into();
        assert!(complete_methods(&changed, &methods, 3).is_err());
        changed = rows;
        changed.push(changed[0].clone());
        assert!(complete_methods(&changed, &methods, 3).is_err());
    }
}
