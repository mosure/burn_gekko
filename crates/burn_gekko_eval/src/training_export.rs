//! Native closeout of a completed training command, without GPU inference.
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingSummaryConfig {
    pub run: PathBuf,
    pub ledger: PathBuf,
    pub telemetry: PathBuf,
    pub command: String,
    pub output: PathBuf,
}

pub fn summarize(c: &TrainingSummaryConfig) -> Result<Value> {
    let mut sources = BTreeMap::new();
    let mut load = |p: PathBuf, lines: bool| -> Result<Value> {
        sources.insert(p.clone(), burn_gekko_data::sha256_file(&p)?);
        let text = fs::read_to_string(&p)?;
        if lines {
            Ok(Value::Array(
                text.lines()
                    .map(serde_json::from_str)
                    .collect::<std::result::Result<Vec<_>, _>>()?,
            ))
        } else if p.extension().is_some_and(|s| s == "toml") {
            burn_gekko_data::read_config(&p)
        } else {
            Ok(serde_json::from_str(&text)?)
        }
    };
    let config = load(c.run.join("config.toml"), false)?;
    let report = load(c.run.join("report.json"), false)?;
    let metadata = load(c.run.join("final/metadata.json"), false)?;
    let steps = load(c.run.join("metrics.jsonl"), true)?;
    let ledger = load(c.ledger.clone(), false)?;
    let telemetry = load(c.telemetry.clone(), true)?;
    let start = report["starting_step"].as_u64().context("starting step")?;
    let end = metadata["completed_steps"].as_u64().context("final step")?;
    ensure!(
        report["completed_steps"] == end,
        "final report step mismatch"
    );
    let commands = ledger["commands"]
        .as_array()
        .context("command ledger")?
        .iter()
        .filter(|v| v["name"] == c.command)
        .collect::<Vec<_>>();
    ensure!(commands.len() == 1, "missing or ambiguous training command");
    let command = commands[0];
    ensure!(
        command["status"] == "complete",
        "training command incomplete"
    );
    let run = command["argv"]
        .as_array()
        .context("training argv")?
        .windows(2)
        .find(|w| w[0] == "--run")
        .and_then(|w| w[1].as_str())
        .context("training command has no run")?;
    ensure!(
        fs::canonicalize(run)? == fs::canonicalize(&c.run)?,
        "telemetry command belongs to another run"
    );
    let pid = command["pid"].as_u64().context("training PID")?;
    let telemetry = telemetry.as_array().unwrap();
    ensure!(
        telemetry.iter().all(|v| v["pid"] == pid),
        "telemetry PID differs from training command"
    );
    let checkpoint = c.run.join("final/model.mpk");
    let sha = burn_gekko_data::sha256_file(&checkpoint)?;
    ensure!(
        metadata["model_sha256"] == sha,
        "checkpoint checksum mismatch"
    );
    sources.insert(checkpoint, sha.clone());
    let steps = steps.as_array().unwrap();
    let coverage = crate::training::coverage(
        steps,
        start,
        end,
        config["batch_size"].as_u64().context("batch size")? as usize,
        config["train_rooms"].as_u64().context("training rooms")? as usize,
    )?;
    let result = json!({
        "schema":1, "run":c.run, "checkpoint_sha256":sha,
        "stop_reason":report["stop_reason"], "coverage":coverage,
        "encoder_stages":crate::training::encoder_stages(steps,start,end)?,
        "scalar_windows":crate::training::scalar_windows(steps,start,end,
            &["cross","monocular","warp_pair_nll","warp_self_nll","augmentation_seconds","gradient_norm"],32)?,
        "efficiency":crate::efficiency::evaluate(steps,telemetry,
            command["elapsed_seconds"].as_f64().context("command duration")?)?,
        "parameter_probes":{
            "teacher_qkv_max_abs_delta":report["teacher_max_abs_delta"],
            "first_encoder_qkv_max_abs_delta":report["first_encoder_max_abs_delta"],
            "last_encoder_qkv_max_abs_delta":report["last_encoder_max_abs_delta"],
            "prediction_head_max_abs_delta":report["prediction_head_max_abs_delta"]
        },
        "sources":sources,
        "scope":"One completed training command. Minibatch trends do not establish generalization or convergence. Parameter probes measure named QKV/head tensors, not a whole-model norm."
    });
    ensure!(!c.output.exists(), "training summary already exists");
    burn_gekko_data::write_json(&c.output, &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn summary_checks_command_pid_checkpoint_and_actual_encoder_updates() {
        let root = tempfile::tempdir().unwrap();
        let run = root.path().join("run");
        fs::create_dir_all(run.join("final")).unwrap();
        let write = |p: PathBuf, v: Value| burn_gekko_data::write_json(&p, &v).unwrap();
        fs::write(run.join("final/model.mpk"), b"model").unwrap();
        let sha = burn_gekko_data::sha256_file(&run.join("final/model.mpk")).unwrap();
        write(
            run.join("final/metadata.json"),
            json!({"completed_steps":2,"model_sha256":sha}),
        );
        write(
            run.join("report.json"),
            json!({"starting_step":0,"completed_steps":2,"stop_reason":"step_limit"}),
        );
        fs::write(run.join("config.toml"), "batch_size = 1\ntrain_rooms = 1\n").unwrap();
        let rows = [
            json!({"step":1,"samples":[[1,0]],"stage":0,"encoder_gradient_tensors":0,"seconds":0.5,"cross":0.2}),
            json!({"step":2,"samples":[[1,1]],"stage":1,"encoder_gradient_tensors":28,"seconds":0.6,"cross":0.1}),
        ];
        fs::write(
            run.join("metrics.jsonl"),
            rows.iter()
                .map(Value::to_string)
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .unwrap();
        let c = TrainingSummaryConfig {
            run: run.clone(),
            ledger: root.path().join("ledger.json"),
            telemetry: root.path().join("gpu.jsonl"),
            command: "train".into(),
            output: root.path().join("summary.json"),
        };
        let ledger = json!({"commands":[{"name":"train","status":"complete","pid":7,"elapsed_seconds":2.,"argv":["binary","--run",run]}]});
        write(c.ledger.clone(), ledger.clone());
        let telemetry = [
            json!({"pid":7,"elapsed_seconds":0.,"device_power_w":100.}),
            json!({"pid":7,"elapsed_seconds":2.,"device_power_w":100.}),
        ];
        fs::write(
            &c.telemetry,
            telemetry
                .iter()
                .map(Value::to_string)
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .unwrap();
        let result = summarize(&c).unwrap();
        assert_eq!(result["coverage"]["unique_room_views"], 2);
        assert_eq!(result["encoder_stages"]["1"]["first_step"], 2);
        assert_eq!(
            result["efficiency"]["observed_board_joules_per_target"],
            100.
        );
        fs::remove_file(&c.output).unwrap();
        let mut wrong = ledger.clone();
        wrong["commands"][0]["pid"] = json!(8);
        write(c.ledger.clone(), wrong);
        assert!(summarize(&c).unwrap_err().to_string().contains("PID"));
        let mut wrong = ledger.clone();
        wrong["commands"][0]["argv"][2] = json!(root.path());
        write(c.ledger.clone(), wrong);
        assert!(
            summarize(&c)
                .unwrap_err()
                .to_string()
                .contains("another run")
        );
        write(c.ledger.clone(), ledger);
        fs::write(run.join("final/model.mpk"), b"tampered").unwrap();
        assert!(
            summarize(&c)
                .unwrap_err()
                .to_string()
                .contains("checksum mismatch")
        );
    }
}
