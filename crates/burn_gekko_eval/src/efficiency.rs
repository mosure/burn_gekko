//! Board telemetry includes the desktop/shared GPU load. No per-process energy attribution.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::BTreeMap;
#[derive(Debug, Serialize, Deserialize)]
pub struct Efficiency {
    pub command_seconds: f64,
    pub observed_seconds: f64,
    pub telemetry_coverage: f64,
    pub observed_board_energy_wh: f64,
    pub observed_mean_board_power_w: f64,
    pub median_device_gpu_percent: Option<f64>,
    pub peak_process_vram_mib: Option<f64>,
    pub peak_process_rss_mib: Option<f64>,
    pub logged_target_exposures: usize,
    pub observed_board_joules_per_target: f64,
    pub median_update_seconds_by_stage: BTreeMap<String, f64>,
    pub p95_update_seconds_by_stage: BTreeMap<String, f64>,
    pub scope: String,
}
pub fn evaluate(steps: &[Value], telemetry: &[Value], command_seconds: f64) -> Result<Efficiency> {
    ensure!(
        !steps.is_empty()
            && telemetry.len() > 1
            && command_seconds.is_finite()
            && command_seconds > 0.,
        "insufficient efficiency observations"
    );
    let mut energy = 0.;
    let mut observed = 0.;
    let mut last = None;
    for row in telemetry {
        let t = row["elapsed_seconds"]
            .as_f64()
            .filter(|x| x.is_finite() && *x >= 0.);
        let p = row["device_power_w"]
            .as_f64()
            .filter(|x| x.is_finite() && *x >= 0.);
        if let (Some(t), Some(p)) = (t, p) {
            if let Some((a, b)) = last {
                ensure!(
                    t > a && t <= command_seconds + 2.,
                    "non-monotonic/out-of-command telemetry"
                );
                // Missing telemetry is never silently filled across long gaps.
                if t - a <= 5. {
                    observed += t - a;
                    energy += (t - a) * (p + b) / 2.;
                }
            }
            last = Some((t, p));
        } else {
            last = None;
        }
    }
    ensure!(observed > 0., "no valid telemetry intervals");
    let mut stages = BTreeMap::<String, Vec<f64>>::new();
    let mut targets = 0;
    for row in steps {
        let time = row["seconds"]
            .as_f64()
            .filter(|x| x.is_finite() && *x > 0.)
            .ok_or_else(|| anyhow::anyhow!("invalid update timing"))?;
        targets += row["samples"]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("missing update target identities"))?
            .len();
        stages
            .entry(row["stage"].to_string())
            .or_default()
            .push(time);
    }
    ensure!(targets > 0, "no logged target exposures");
    let mut median = BTreeMap::new();
    let mut p95 = BTreeMap::new();
    for (stage, mut values) in stages {
        // Drop first ten updates of each stage for steady timing, retain all for energy.
        if values.len() > 20 {
            values.drain(..10);
        }
        values.sort_by(f64::total_cmp);
        let q = |p: f64| {
            let x = p * (values.len() - 1) as f64;
            let l = x.floor() as usize;
            let h = x.ceil() as usize;
            values[l] + (values[h] - values[l]) * x.fract()
        };
        median.insert(stage.clone(), q(0.5));
        p95.insert(stage, q(0.95));
    }
    let observations = |key: &str| {
        telemetry
            .iter()
            .filter_map(|v| v[key].as_f64())
            .filter(|v| v.is_finite() && *v >= 0.)
            .collect::<Vec<_>>()
    };
    let mut utilization = observations("device_gpu_percent");
    utilization.retain(|v| *v <= 100.);
    utilization.sort_by(f64::total_cmp);
    let median_utilization = if utilization.is_empty() {
        None
    } else {
        Some((utilization[(utilization.len() - 1) / 2] + utilization[utilization.len() / 2]) / 2.)
    };
    let peak = |key| observations(key).into_iter().max_by(f64::total_cmp);
    Ok(Efficiency {command_seconds,observed_seconds:observed,telemetry_coverage:observed/command_seconds,observed_board_energy_wh:energy/3600.,observed_mean_board_power_w:energy/observed,median_device_gpu_percent:median_utilization,peak_process_vram_mib:peak("process_vram_mib"),peak_process_rss_mib:peak("process_rss_mib"),logged_target_exposures:targets,observed_board_joules_per_target:energy/targets as f64,median_update_seconds_by_stage:median,p95_update_seconds_by_stage:p95,scope:"Device-wide board power/utilization, including shared desktop load; observed intervals only, gaps over 5 seconds excluded. Process memory is separately attributed. Stage timing excludes first 10 updates when at least 21 exist; energy retains command preparation and evaluation. Utilization is sample-median, not SM occupancy.".into()})
}
#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn energy_integrates_only_observed_intervals() {
        let steps = vec![json!({"seconds":0.5,"samples":[[1,0],[2,0]],"stage":0})];
        let telemetry = vec![
            json!({"elapsed_seconds":0.,"device_power_w":100.}),
            json!({"elapsed_seconds":1.,"device_power_w":200.}),
            json!({"elapsed_seconds":10.,"device_power_w":100.}),
        ];
        let e = evaluate(&steps, &telemetry, 10.).unwrap();
        assert_eq!(e.observed_seconds, 1.);
        assert_eq!(e.observed_board_joules_per_target, 75.);
        assert_eq!(e.telemetry_coverage, 0.1);
    }
}
