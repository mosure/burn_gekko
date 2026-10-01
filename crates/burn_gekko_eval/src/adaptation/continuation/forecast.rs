//! Forecast a complete registered horizon from bounded, matched native preflights.
use crate::adaptation::{Input, number, pinned};
use anyhow::{Context, Result, ensure};
use burn_gekko_data::{read_config, sha256_file, write_json};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Arm {
    pub summary: Input,
    pub full_config: Input,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub arms: [Arm; 2],
    pub ledger: Input,
    pub updates: u64,
    pub preflight_updates: u64,
    pub update_margin: f64,
    pub periodic_margin_seconds: f64,
    pub evaluation_reserve_seconds: f64,
    pub output: PathBuf,
}

fn projection(p95: f64, overhead: f64, c: &Config) -> f64 {
    p95 * c.updates as f64 * c.update_margin + overhead + c.periodic_margin_seconds
}

pub fn analyze(c: &Config) -> Result<Value> {
    ensure!(!c.output.exists(), "preserve existing throughput forecast");
    ensure!(
        c.updates > c.preflight_updates
            && c.preflight_updates >= 32
            && c.update_margin.is_finite()
            && c.update_margin >= 1.25
            && c.periodic_margin_seconds.is_finite()
            && c.periodic_margin_seconds >= 600.
            && c.evaluation_reserve_seconds.is_finite()
            && c.evaluation_reserve_seconds >= 7200.,
        "insufficient registered forecast margin"
    );
    let mut sources = BTreeMap::new();
    let ledger = pinned(&c.ledger, &mut sources)?;
    let ceiling = number(&ledger["ceiling_seconds"])?;
    let used = number(&ledger["command_seconds"])?;
    ensure!(
        ceiling > 0.
            && ceiling <= 43200.
            && used >= 0.
            && used < ceiling
            && ledger.get("running").is_none(),
        "invalid or active GPU budget"
    );
    let mut forecasts = Vec::new();
    let mut common = None;
    for arm in &c.arms {
        let summary = pinned(&arm.summary, &mut sources)?;
        for (path, expected) in summary["sources"]
            .as_object()
            .context("preflight sources")?
        {
            let path = PathBuf::from(path);
            let expected = expected.as_str().context("source checksum")?;
            ensure!(sha256_file(&path)? == expected, "preflight source changed");
            sources.insert(path, expected.to_string());
        }
        let run = PathBuf::from(summary["run"].as_str().context("preflight run")?);
        ensure!(
            sources.contains_key(&run.join("config.toml"))
                && sources.contains_key(&run.join("metrics.jsonl"))
                && sources.contains_key(&run.join("report.json")),
            "incomplete preflight source closure"
        );
        let mut preflight: Value = read_config(&run.join("config.toml"))?;
        let full: Value = read_config(&arm.full_config.path)?;
        ensure!(
            sha256_file(&arm.full_config.path)? == arm.full_config.sha256,
            "full recipe checksum differs"
        );
        sources.insert(arm.full_config.path.clone(), arm.full_config.sha256.clone());
        ensure!(
            summary["stop_reason"] == "step_limit"
                && summary["coverage"]["updates"] == c.preflight_updates
                && preflight["steps"] == c.preflight_updates
                && full["steps"] == c.updates
                && full["decay_steps"] == c.updates
                && summary["encoder_stages"]["2"]["updates"] == c.preflight_updates
                && summary["encoder_stages"]["2"]["min_gradient_tensors"] == 151
                && summary["encoder_stages"]["2"]["max_gradient_tensors"] == 151
                && summary["parameter_probes"]["teacher_qkv_max_abs_delta"] == 0.
                && summary["parameter_probes"]["preservation_anchor_qkv_max_abs_delta"]
                    == json!([0., 0.]),
            "preflight did not complete with the registered encoder and fixed targets"
        );
        let mut comparable_full = full.clone();
        for key in ["steps", "max_seconds", "checkpoint_every"] {
            preflight
                .as_object_mut()
                .context("preflight recipe")?
                .remove(key);
            comparable_full
                .as_object_mut()
                .context("full recipe")?
                .remove(key);
        }
        // The trainer serializes default fields into its run config. Compare each
        // explicitly registered field, while the matched runs below check all defaults.
        ensure!(
            comparable_full
                .as_object()
                .unwrap()
                .iter()
                .all(|(k, v)| preflight.get(k) == Some(v)),
            "preflight changes the full scientific recipe"
        );
        let mut identity = preflight;
        identity.as_object_mut().unwrap().remove("view_geometry");
        let identity = json!({"recipe":identity,"source":summary["source_sha256"],"teacher":summary["teacher_id"]});
        if let Some(common) = &common {
            ensure!(*common == identity, "preflight arms differ beyond geometry");
        } else {
            common = Some(identity);
        }
        let rows = fs::read_to_string(run.join("metrics.jsonl"))?
            .lines()
            .map(serde_json::from_str::<Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        ensure!(
            rows.len() as u64 == c.preflight_updates,
            "incomplete preflight trajectory"
        );
        let seconds = rows
            .iter()
            .map(|r| number(&r["seconds"]))
            .collect::<Result<Vec<_>>>()?;
        for row in &rows {
            for key in [
                "total",
                "cross",
                "monocular",
                "gradient_norm",
                "encoder_preservation_mse",
            ] {
                ensure!(number(&row[key])? >= 0., "invalid preflight scalar");
            }
            if full.get("view_geometry").is_some() {
                ensure!(
                    number(&row["view_geometry_nll"])? > 0.
                        && number(&row["view_geometry_valid_fraction"])? > 0.,
                    "inactive geometry preflight"
                );
            }
        }
        let efficiency = &summary["efficiency"];
        let p95 = number(&efficiency["p95_update_seconds_by_stage"]["2"])?;
        let wall = number(&efficiency["command_seconds"])?;
        let overhead = wall - seconds.iter().sum::<f64>();
        ensure!(p95 > 0. && overhead >= 0., "invalid preflight timing");
        let projected = projection(p95, overhead, c);
        let cap = number(&full["max_seconds"])?;
        forecasts.push(
            json!({"full_config":arm.full_config.path,"preflight_run":run,
            "warm_update_p95_seconds":p95,"observed_nonupdate_seconds":overhead,
            "forecast_command_seconds":projected,"trainer_cap_seconds":cap,
            "fits_individual_cap":projected < cap,"preflight_efficiency":efficiency}),
        );
    }
    let total = forecasts
        .iter()
        .map(|v| v["forecast_command_seconds"].as_f64().unwrap())
        .sum::<f64>();
    let fits = forecasts.iter().all(|v| v["fits_individual_cap"] == true)
        && total + c.evaluation_reserve_seconds <= ceiling - used;
    let result = json!({"schema":1,"complete_horizon_forecast_passed":fits,"updates_per_arm":c.updates,
        "remaining_command_seconds":ceiling-used,"forecast_training_seconds":total,
        "evaluation_reserve_seconds":c.evaluation_reserve_seconds,"arms":forecasts,"sources":sources,
        "scope":"Engineering admission check, not checkpoint selection. Warm update p95 excludes ten updates; multiplicative margin and explicit periodic-evaluation allowance are retained. Shared-GPU changes can still invalidate a forecast. No quality threshold is tuned here."});
    write_json(&c.output, &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn forecast_reserves_overhead_and_evaluation_without_shortening_updates() {
        let input = || Input {
            path: PathBuf::from("fixture"),
            sha256: String::new(),
        };
        let c = Config {
            arms: std::array::from_fn(|_| Arm {
                summary: input(),
                full_config: input(),
            }),
            ledger: input(),
            updates: 4096,
            preflight_updates: 64,
            update_margin: 1.25,
            periodic_margin_seconds: 600.,
            evaluation_reserve_seconds: 7200.,
            output: PathBuf::new(),
        };
        assert_eq!(projection(1.2, 200., &c), 6944.);
        assert_eq!(projection(2., 200., &c), 11040.);
        // A slow arm exceeds the registered cap; unused global budget must not
        // implicitly change that cap or the fixed update endpoint.
        assert!(projection(2., 200., &c) > 10000.);
    }
}
