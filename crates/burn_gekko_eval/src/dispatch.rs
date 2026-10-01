//! CPU analysis of pinned Nsight GPU/API traces, without an occupancy inference.
use anyhow::{Context, Result, ensure};
use burn_gekko_data::{sha256_file, write_json};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};
pub mod comparison;
pub mod warm;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DispatchConfig {
    pub gpu_trace: Input,
    pub api_trace: Input,
    pub output: PathBuf,
    pub description: String,
    pub warm_ranges: Option<warm::Config>,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Span {
    start: u64,
    end: u64,
}
impl Span {
    fn read(row: &Value) -> Result<Self> {
        let start = row["Start (ns)"]
            .as_u64()
            .context("trace requires integer Start (ns)")?;
        let duration = row["Duration (ns)"]
            .as_u64()
            .context("trace requires integer Duration (ns)")?;
        let end = start
            .checked_add(duration)
            .context("trace timestamp overflow")?;
        Ok(Self { start, end })
    }
    fn duration(self) -> u64 {
        self.end - self.start
    }
    fn clipped(self, window: Self) -> Option<Self> {
        let start = self.start.max(window.start);
        let end = self.end.min(window.end);
        (start < end).then_some(Self { start, end })
    }
}
fn merged(mut spans: Vec<Span>) -> Vec<Span> {
    spans.sort_by_key(|v| (v.start, v.end));
    let mut union = Vec::<Span>::new();
    for span in spans {
        if span.start == span.end {
            continue;
        }
        if let Some(last) = union.last_mut().filter(|v| span.start <= v.end) {
            last.end = last.end.max(span.end);
        } else {
            union.push(span);
        }
    }
    union
}
fn load(input: &Input) -> Result<Vec<Value>> {
    ensure!(
        sha256_file(&input.path)? == input.sha256,
        "dispatch input changed"
    );
    let rows: Vec<Value> = serde_json::from_slice(&fs::read(&input.path)?)?;
    ensure!(!rows.is_empty(), "empty dispatch trace");
    Ok(rows)
}
#[derive(Default, Serialize)]
struct Group {
    events: u64,
    summed_duration_ns: u64,
    events_at_most_10us: u64,
}
fn add(groups: &mut BTreeMap<String, Group>, name: &str, duration: u64) -> Result<()> {
    let group = groups.entry(name.into()).or_default();
    group.events += 1;
    group.summed_duration_ns = group
        .summed_duration_ns
        .checked_add(duration)
        .context("duration sum overflow")?;
    group.events_at_most_10us += u64::from(duration <= 10_000);
    Ok(())
}
fn name(row: &Value) -> Result<&str> {
    row["Name"]
        .as_str()
        .filter(|v| !v.is_empty())
        .context("missing trace name")
}
fn analyze(gpu: &[Value], api: &[Value]) -> Result<Value> {
    ensure!(!gpu.is_empty() && !api.is_empty(), "empty dispatch trace");
    let mut spans = Vec::new();
    let mut devices = BTreeSet::new();
    let mut kernels = BTreeMap::new();
    let mut memory = BTreeMap::new();
    let mut kernel_spans = Vec::new();
    for row in gpu {
        let span = Span::read(row)?;
        devices.insert(
            row["Device"]
                .as_str()
                .context("missing GPU trace device")?
                .to_owned(),
        );
        let grid = &row["GrdX"];
        if grid.as_u64().is_some_and(|x| x > 0) {
            add(&mut kernels, name(row)?, span.duration())?;
            kernel_spans.push(span);
        } else {
            ensure!(grid.is_null() || grid == "", "unknown GPU trace event kind");
            ensure!(
                row["Bytes (B)"].as_u64().is_some(),
                "memory event lacks byte count"
            );
            add(&mut memory, name(row)?, span.duration())?;
        }
        spans.push(span);
    }
    ensure!(
        devices.len() == 1 && !kernels.is_empty(),
        "expected one traced device with kernels"
    );
    let union = merged(spans);
    let window = Span {
        start: union.first().context("zero-length GPU trace")?.start,
        end: union.last().unwrap().end,
    };
    let active: u64 = union.iter().map(|v| v.duration()).sum();
    let kernel_union: u64 = merged(kernel_spans).iter().map(|v| v.duration()).sum();
    let gaps = union
        .windows(2)
        .map(|v| v[1].start - v[0].end)
        .collect::<Vec<_>>();
    let mut pids = BTreeSet::new();
    let mut calls = BTreeMap::new();
    let mut api_spans = Vec::new();
    let mut synchronization = Vec::new();
    for row in api {
        let span = Span::read(row)?;
        pids.insert(row["Pid"].as_u64().context("missing API trace PID")?);
        if let Some(span) = span.clipped(window) {
            let function = name(row)?;
            add(&mut calls, function, span.duration())?;
            api_spans.push(span);
            if function.contains("Synchronize") {
                synchronization.push(span);
            }
        }
    }
    ensure!(
        pids.len() == 1 && !calls.is_empty(),
        "expected one API process with events in the GPU window"
    );
    let api_union: u64 = merged(api_spans).iter().map(|v| v.duration()).sum();
    let sync_union: u64 = merged(synchronization).iter().map(|v| v.duration()).sum();
    let launches: u64 = kernels.values().map(|v| v.events).sum();
    let short: u64 = kernels.values().map(|v| v.events_at_most_10us).sum();
    Ok(
        json!({"schema":1, "device":devices.first(), "api_pid":pids.first(),
        "window_start_ns":window.start, "window_end_ns":window.end, "window_seconds":window.duration() as f64 / 1e9,
        "gpu_event_union_seconds":active as f64 / 1e9, "gpu_event_coverage_fraction":active as f64 / window.duration() as f64,
        "kernel_event_union_seconds":kernel_union as f64 / 1e9,
        "kernel_events":launches, "kernel_events_at_most_10us":short, "small_kernel_fraction":short as f64 / launches as f64,
        "uncovered_interval_seconds":(window.duration()-active) as f64 / 1e9,
        "largest_uncovered_interval_ms":gaps.iter().max().copied().unwrap_or(0) as f64 / 1e6,
        "uncovered_intervals_at_least_100us":gaps.iter().filter(|&&v| v >= 100_000).count(),
        "api_event_union_seconds":api_union as f64 / 1e9, "synchronization_api_union_seconds":sync_union as f64 / 1e9,
        "kernels":kernels, "memory_operations":memory, "apis":calls,
        "scope":"First-to-last traced GPU event of one profiled process. Overlapping GPU events are merged. API durations are clipped to that window and merged separately. Preparation, JIT and final evaluation may be present; this is not an isolated steady-state training measurement. Uncovered intervals are not whole-device idle time or SM occupancy, and synchronization may overlap productive GPU work. Per-name duration sums may overlap and must not be added to infer elapsed time. Profiling overhead prevents comparison with unprofiled throughput; no causal dispatch-bottleneck claim follows from small-kernel counts alone."}),
    )
}
pub fn summarize(c: &DispatchConfig) -> Result<Value> {
    ensure!(!c.output.exists(), "preserve existing dispatch receipt");
    ensure!(
        !c.description.trim().is_empty(),
        "dispatch scope description required"
    );
    let gpu = load(&c.gpu_trace)?;
    let api = load(&c.api_trace)?;
    let mut result = analyze(&gpu, &api)?;
    result["description"] = json!(c.description);
    result["sources"] = json!({c.gpu_trace.path.display().to_string():c.gpu_trace.sha256,c.api_trace.path.display().to_string():c.api_trace.sha256});
    if let Some(config) = &c.warm_ranges {
        result["warm_ranges"] =
            warm::analyze(&gpu, &api, &load(&config.trace)?, config.expected_updates)?;
        result["sources"][config.trace.path.display().to_string()] = json!(config.trace.sha256);
    }
    write_json(&c.output, &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn overlapping_kernels_and_api_calls_do_not_invent_occupancy() {
        let gpu = vec![
            json!({"Start (ns)":100,"Duration (ns)":100,"GrdX":1,"Name":"a","Device":"GPU0"}),
            json!({"Start (ns)":150,"Duration (ns)":100,"GrdX":1,"Name":"b","Device":"GPU0"}),
            json!({"Start (ns)":300,"Duration (ns)":50,"GrdX":"","Bytes (B)":4,"Name":"copy","Device":"GPU0"}),
        ];
        let api = vec![
            json!({"Start (ns)":0,"Duration (ns)":200,"Name":"cuStreamSynchronize","Pid":42}),
            json!({"Start (ns)":120,"Duration (ns)":100,"Name":"cuLaunchKernel","Pid":42}),
        ];
        let r = analyze(&gpu, &api).unwrap();
        assert_eq!(r["gpu_event_coverage_fraction"], 0.8);
        assert_eq!(r["kernel_event_union_seconds"], 150e-9);
        assert_eq!(r["api_event_union_seconds"], 120e-9);
        assert_eq!(r["synchronization_api_union_seconds"], 100e-9);
        assert_eq!(r["uncovered_interval_seconds"], 50e-9);
        let mut multiple = api;
        multiple[1]["Pid"] = json!(43);
        assert!(analyze(&gpu, &multiple).is_err());
    }
    #[test]
    fn nanosecond_contract_and_timestamp_overflow_are_checked() {
        assert!(Span::read(&json!({"Start (us)":0,"Duration (ns)":10})).is_err());
        assert!(Span::read(&json!({"Start (ns)":u64::MAX,"Duration (ns)":1})).is_err());
        assert_eq!(
            merged(vec![
                Span { start: 0, end: 10 },
                Span { start: 10, end: 20 }
            ]),
            vec![Span { start: 0, end: 20 }]
        );
    }
}
