//! Fixed A-B-B-A runtime screen with numerical replay and unprofiled efficiency.
use crate::{adaptation::Input, replay};
use anyhow::{Context, Result, ensure};
use burn_gekko_data::{read_config, sha256_file, write_json};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub protocol: Input,
    pub original_prefix: Input,
    /// Baseline, candidate, candidate, baseline; both repetitions are required.
    pub summaries: [Input; 4],
    pub updates: usize,
    pub output: PathBuf,
}
fn number(v: &Value) -> Result<f64> {
    v.as_f64()
        .filter(|v| v.is_finite())
        .context("missing finite runtime measurement")
}
fn load(input: &Input, sources: &mut BTreeMap<PathBuf, String>) -> Result<Value> {
    ensure!(
        sha256_file(&input.path)? == input.sha256,
        "runtime input changed"
    );
    sources.insert(input.path.clone(), input.sha256.clone());
    Ok(serde_json::from_slice(&fs::read(&input.path)?)?)
}
fn rows(path: PathBuf) -> Result<Vec<Value>> {
    fs::read_to_string(path)?
        .lines()
        .map(|s| Ok(serde_json::from_str(s)?))
        .collect()
}
fn gates(e: &[Value]) -> Result<Value> {
    ensure!(e.len() == 4, "A-B-B-A requires four complete runs");
    let min_a = |key: &str| -> Result<f64> { Ok(number(&e[0][key])?.min(number(&e[3][key])?)) };
    let max_a = |key: &str| -> Result<f64> { Ok(number(&e[0][key])?.max(number(&e[3][key])?)) };
    let median = number(&e[0]["median_update_seconds_by_stage"]["2"])?
        .min(number(&e[3]["median_update_seconds_by_stage"]["2"])?);
    let p95 = number(&e[0]["p95_update_seconds_by_stage"]["2"])?
        .min(number(&e[3]["p95_update_seconds_by_stage"]["2"])?);
    let joules = min_a("observed_board_joules_per_target")?;
    let vram = max_a("peak_process_vram_mib")?;
    let rss = max_a("peak_process_rss_mib")?;
    let mut medians = true;
    let mut tails = true;
    let mut energy = true;
    let mut memory = true;
    for row in &e[1..3] {
        medians &= number(&row["median_update_seconds_by_stage"]["2"])? <= median * 0.95;
        tails &= number(&row["p95_update_seconds_by_stage"]["2"])? <= p95;
        energy &= number(&row["observed_board_joules_per_target"])? <= joules;
        memory &= number(&row["peak_process_vram_mib"])? <= vram * 1.05
            && number(&row["peak_process_rss_mib"])? <= rss * 1.05;
    }
    let coverage = e
        .iter()
        .map(|r| number(&r["telemetry_coverage"]))
        .collect::<Result<Vec<_>>>()?
        .iter()
        .all(|&v| v > 0.99);
    Ok(
        json!({"both_candidate_medians_at_least_5_percent_faster":medians,
        "both_candidate_p95_no_worse":tails,"both_candidate_board_energy_no_worse":energy,
        "memory_within_5_percent":memory,"all_telemetry_over_99_percent":coverage,
        "passed":medians && tails && energy && memory && coverage}),
    )
}

pub fn analyze(c: &Config) -> Result<Value> {
    ensure!(
        c.updates == 128 && !c.output.exists(),
        "unregistered runtime horizon or existing output"
    );
    let mut sources = BTreeMap::new();
    for input in [&c.protocol, &c.original_prefix] {
        ensure!(
            sha256_file(&input.path)? == input.sha256,
            "runtime protocol or original changed"
        );
        sources.insert(input.path.clone(), input.sha256.clone());
    }
    let original = rows(c.original_prefix.path.clone())?;
    ensure!(
        original.len() == 64,
        "expected complete registered 64-update original"
    );
    let mut summaries = Vec::new();
    let mut updates = Vec::new();
    let mut recipe = None;
    for input in &c.summaries {
        let summary = load(input, &mut sources)?;
        let upstream: BTreeMap<PathBuf, String> =
            serde_json::from_value(summary["sources"].clone())?;
        for (path, expected) in upstream {
            ensure!(sha256_file(&path)? == expected, "runtime source changed");
            sources.insert(path, expected);
        }
        let run = PathBuf::from(summary["run"].as_str().context("runtime run path")?);
        for file in [
            "config.toml",
            "metrics.jsonl",
            "report.json",
            "provenance.json",
        ] {
            ensure!(
                sources.contains_key(&run.join(file)),
                "incomplete runtime evidence closure"
            );
        }
        let settings: Value = read_config(&run.join("config.toml"))?;
        if let Some(previous) = &recipe {
            ensure!(*previous == settings, "runtime numerical recipe differs");
        } else {
            recipe = Some(settings);
        }
        ensure!(
            summary["stop_reason"] == "step_limit"
                && summary["coverage"]["updates"] == c.updates
                && summary["encoder_stages"]["2"]["updates"] == c.updates
                && summary["encoder_stages"]["2"]["min_gradient_tensors"] == 151
                && summary["encoder_stages"]["2"]["max_gradient_tensors"] == 151
                && summary["parameter_probes"]["teacher_qkv_max_abs_delta"] == 0.
                && summary["parameter_probes"]["preservation_anchor_qkv_max_abs_delta"]
                    == json!([0., 0.]),
            "runtime replay incomplete or fixed targets changed"
        );
        let row = rows(run.join("metrics.jsonl"))?;
        ensure!(
            row.len() == c.updates
                && row.iter().all(|r| r["view_geometry_nll"]
                    .as_f64()
                    .is_some_and(|v| v.is_finite() && v > 0.)),
            "runtime geometry objective inactive or trajectory incomplete"
        );
        updates.push(row);
        summaries.push(summary);
    }
    ensure!(
        summaries[0]["source_sha256"] == summaries[3]["source_sha256"]
            && summaries[1]["source_sha256"] == summaries[2]["source_sha256"]
            && summaries[0]["source_sha256"] != summaries[1]["source_sha256"],
        "runtime source identities do not form A-B-B-A"
    );
    let mut replays = vec![replay::compare(&original, &updates[0], 1e-6, 1e-5)?];
    for row in &updates[1..] {
        replays.push(replay::compare(&updates[0], row, 1e-6, 1e-5)?);
    }
    let numerical = replays.iter().all(|r| r["passed"] == true);
    let efficiency = summaries
        .iter()
        .map(|s| s["efficiency"].clone())
        .collect::<Vec<_>>();
    let performance = gates(&efficiency)?;
    let result = json!({"schema":1,"numerical_replay_passed":numerical,"replays":replays,
        "performance_gates":performance,"passed":numerical && performance["passed"] == true,
        "order":["baseline","candidate","candidate","baseline"],"efficiency":efficiency,
        "sources":sources,
        "scope":"Discarded 128-update A-B-B-A runtime screen. Numerical replay is necessary before performance interpretation. Warm timing excludes ten updates; board energy covers the entire command and authorized shared desktop load. Two repetitions per variant are a bounded engineering screen, not process-power attribution, causal occupancy measurement, long-run reliability or model quality. Production adoption is separate."});
    write_json(&c.output, &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fastest_baseline_and_both_candidates_must_pass_energy_and_memory() {
        let a = json!({"median_update_seconds_by_stage":{"2":1.},"p95_update_seconds_by_stage":{"2":1.2},"observed_board_joules_per_target":30.,"peak_process_vram_mib":100.,"peak_process_rss_mib":200.,"telemetry_coverage":0.999});
        let mut b = a.clone();
        b["median_update_seconds_by_stage"]["2"] = json!(0.9);
        let mut v = vec![a.clone(), b.clone(), b, a];
        assert_eq!(gates(&v).unwrap()["passed"], true);
        v[2]["observed_board_joules_per_target"] = json!(31.);
        assert_eq!(gates(&v).unwrap()["passed"], false);
        v[2]["observed_board_joules_per_target"] = json!(30.);
        v[3]["median_update_seconds_by_stage"]["2"] = json!(0.94);
        assert_eq!(gates(&v).unwrap()["passed"], false);
        v[3]["median_update_seconds_by_stage"]["2"] = json!(1.);
        v[1]["peak_process_vram_mib"] = json!(106.);
        assert_eq!(gates(&v).unwrap()["passed"], false);
    }
}
