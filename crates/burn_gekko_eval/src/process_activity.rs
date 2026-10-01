//! Descriptive NVIDIA pmon counters, without attributing board energy to processes.
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActivityConfig {
    pub input: PathBuf,
    pub output: PathBuf,
}

#[derive(Debug, Serialize)]
pub struct ProcessActivity {
    pub gpu_index: u64,
    pub pid: u64,
    pub command: String,
    pub observed_rows: usize,
    pub numeric_sm_rows: usize,
    pub unavailable_sm_rows: usize,
    pub mean_reported_sm_percent: Option<f64>,
    pub peak_reported_fb_mb: Option<u64>,
}

/// Supports named pmon columns with or without `-o DT` timestamps.
/// Missing counters (`-`) remain missing; they are never converted to zero.
pub fn parse(text: &str) -> Result<Vec<ProcessActivity>> {
    let mut header: Option<Vec<&str>> = None;
    let mut processes = BTreeMap::<(u64, u64, String), (ProcessActivity, f64)>::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let columns = line.split_whitespace().collect::<Vec<_>>();
        if line.trim_start().starts_with('#') {
            if columns.contains(&"pid") && columns.contains(&"command") {
                // Undated output has a standalone comment marker; dated output
                // attaches it to the first column name (`#Date`).
                header = Some(if columns.first() == Some(&"#") {
                    columns[1..].to_vec()
                } else {
                    columns
                });
            }
            continue;
        }
        let h = header.as_ref().context("pmon data before named header")?;
        ensure!(
            h.last() == Some(&"command"),
            "pmon command column must be last"
        );
        ensure!(columns.len() >= h.len(), "incomplete pmon observation");
        let field = |name: &str| -> Result<&str> {
            Ok(columns[h
                .iter()
                .position(|x| *x == name)
                .context("missing pmon column")?])
        };
        if field("pid")? == "-" {
            continue;
        }
        let pid = field("pid")?
            .parse::<u64>()
            .context("invalid process PID")?;
        // NVIDIA may include a truncated argument suffix (e.g. chrome --type=g).
        // Treat that final text column as opaque, preserving its internal spaces.
        let command = columns[h.len() - 1..].join(" ");
        let gpu_index = field("gpu")?.parse::<u64>().context("invalid GPU index")?;
        let (entry, sum) = processes
            .entry((gpu_index, pid, command.clone()))
            .or_insert((
                ProcessActivity {
                    gpu_index,
                    pid,
                    command,
                    observed_rows: 0,
                    numeric_sm_rows: 0,
                    unavailable_sm_rows: 0,
                    mean_reported_sm_percent: None,
                    peak_reported_fb_mb: None,
                },
                0.,
            ));
        entry.observed_rows += 1;
        let sm = field("sm")?;
        if sm == "-" {
            entry.unavailable_sm_rows += 1;
        } else {
            let value = sm.parse::<f64>().context("invalid SM counter")?;
            ensure!((0.0..=100.0).contains(&value), "invalid SM counter range");
            *sum += value;
            entry.numeric_sm_rows += 1;
            entry.mean_reported_sm_percent = Some(*sum / entry.numeric_sm_rows as f64);
        }
        let memory = field("fb")?;
        if memory != "-" {
            let memory = memory
                .parse::<u64>()
                .context("invalid process framebuffer counter")?;
            entry.peak_reported_fb_mb = Some(entry.peak_reported_fb_mb.unwrap_or(0).max(memory));
        }
    }
    ensure!(header.is_some(), "missing pmon header");
    Ok(processes.into_values().map(|(entry, _)| entry).collect())
}

pub fn summarize(c: &ActivityConfig) -> Result<()> {
    ensure!(!c.output.exists(), "activity summary already exists");
    let bytes = fs::read_to_string(&c.input)?;
    let result = serde_json::json!({
        "schema":1,"input":c.input,"input_sha256":burn_gekko_data::sha256_file(&c.input)?,
        "processes":parse(&bytes)?,
        "scope":"Descriptive counters over this logged observation window, not occupancy or process-attributed energy. Process percentages are not additive. Missing counters remain unavailable. The observation window may include training, idle intervals and evaluation."
    });
    burn_gekko_data::write_json(&c.output, &result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn missing_counters_are_distinct_from_idle_and_truncated_rows_fail() {
        let header = "#Date Time gpu pid type sm mem enc dec jpg ofa fb ccpm command\n";
        let rows = "20260930 02:00:00 0 7 G - - - - - - 100 0 desktop\n20260930 02:00:10 0 7 G 0 0 - - - - 110 0 desktop\n20260930 02:00:20 0 7 G 40 2 - - - - 105 0 desktop\n20260930 02:00:20 0 8 G - - - - - - 90 0 remote\n";
        let p = parse(&format!("{header}{rows}")).unwrap();
        assert_eq!(p[0].observed_rows, 3);
        assert_eq!(p[0].numeric_sm_rows, 2);
        assert_eq!(p[0].unavailable_sm_rows, 1);
        assert_eq!(p[0].mean_reported_sm_percent, Some(20.));
        assert_eq!(p[0].peak_reported_fb_mb, Some(110));
        assert_eq!(p[1].mean_reported_sm_percent, None);
        let extra = "20260930 02:00:20 1 7 G 50 2 - - - - 200 0 chrome --type=g\n";
        let p = parse(&format!("{header}{rows}{extra}")).unwrap();
        assert_eq!(p.len(), 3);
        assert_eq!(p[2].gpu_index, 1);
        assert_eq!(p[2].command, "chrome --type=g");
        assert_eq!(p[2].mean_reported_sm_percent, Some(50.));
        assert!(parse(&format!("{header}20260930 02:00:00 0 7\n")).is_err());
        assert!(parse(&format!("{header}{}", rows.replace("40 2", "101 2"))).is_err());
    }

    #[test]
    fn undated_header_does_not_shift_gpu_and_pid_columns() {
        let rows = "# gpu pid type sm mem enc dec jpg ofa fb ccpm command\n# Idx # C/G % % % % % % MB MB name\n0 42 C 37 5 - - - - 27380 0 gekko\n";
        let p = parse(rows).unwrap();
        assert_eq!(p[0].gpu_index, 0);
        assert_eq!(p[0].pid, 42);
        assert_eq!(p[0].mean_reported_sm_percent, Some(37.));
        assert_eq!(p[0].peak_reported_fb_mb, Some(27380));
    }
}
