//! Overlap-aware event coverage inside validated warm-update host ranges.
use super::{Input, Span, merged, name};
use anyhow::{Context, Result, ensure};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub trace: Input,
    pub expected_updates: usize,
}

const PHASES: [&str; 5] = [
    "data",
    "frozen-targets",
    "forward-and-loss-readback",
    "backward-and-clip",
    "optimizer-and-sync",
];

fn coverage(gpu: &[Value], api: &[Value], windows: &[Span]) -> Result<Value> {
    let windows = merged(windows.to_vec());
    let wall: u64 = windows.iter().map(|s| s.duration()).sum();
    ensure!(wall > 0, "empty profiling window");
    let clips = |span: Span| {
        windows
            .iter()
            .filter_map(|&w| span.clipped(w))
            .collect::<Vec<_>>()
    };
    let mut gpu_spans = Vec::new();
    let mut kernels = Vec::new();
    let mut kernel_count = 0;
    let mut small = 0;
    let mut copies = 0;
    for row in gpu {
        let span = Span::read(row)?;
        let selected = clips(span);
        if selected.is_empty() {
            continue;
        }
        if row["GrdX"].as_u64().is_some_and(|n| n > 0) {
            kernel_count += 1;
            small += usize::from(span.duration() <= 10_000);
            kernels.extend_from_slice(&selected);
        } else {
            ensure!(row["Bytes (B)"].as_u64().is_some(), "unknown GPU event");
            copies += 1;
        }
        gpu_spans.extend(selected);
    }
    let mut calls = Vec::new();
    let mut sync = Vec::new();
    for row in api {
        let selected = clips(Span::read(row)?);
        if name(row)?.contains("Synchronize") {
            sync.extend_from_slice(&selected);
        }
        calls.extend(selected);
    }
    let duration = |spans| merged(spans).iter().map(|s| s.duration()).sum::<u64>();
    let active = duration(gpu_spans);
    Ok(json!({
        "host_range_seconds":wall as f64/1e9,
        "gpu_event_union_seconds":active as f64/1e9,
        "gpu_event_coverage_fraction":active as f64/wall as f64,
        "uncovered_interval_seconds":(wall-active) as f64/1e9,
        "kernel_event_union_seconds":duration(kernels) as f64/1e9,
        "api_event_union_seconds":duration(calls) as f64/1e9,
        "synchronization_api_union_seconds":duration(sync) as f64/1e9,
        "kernel_events_intersecting_ranges":kernel_count,
        "kernels_at_most_10us_intersecting_ranges":small,
        "memory_events_intersecting_ranges":copies
    }))
}

pub(super) fn analyze(
    gpu: &[Value],
    api: &[Value],
    rows: &[Value],
    expected: usize,
) -> Result<Value> {
    ensure!(expected > 0, "missing registered warm-update count");
    let mut ranges = BTreeMap::new();
    let mut threads = BTreeSet::new();
    for row in rows {
        let Some(label) = name(row)?.strip_prefix("burn_gekko:") else {
            continue;
        };
        ensure!(
            label == "warm-training" || label == "update" || PHASES.contains(&label),
            "unknown training range"
        );
        let span = Span::read(row)?;
        ensure!(
            row["End (ns)"].as_u64() == Some(span.end) && span.duration() > 0,
            "incomplete NVTX range"
        );
        let id = row["RangeId"].as_u64().context("range identity")?;
        ensure!(
            ranges.insert(id, (label, span, &row["ParentId"])).is_none(),
            "duplicate range identity"
        );
        threads.insert((
            row["PID"].as_u64().context("range PID")?,
            row["TID"].as_u64().context("range TID")?,
        ));
    }
    ensure!(threads.len() == 1, "expected one annotated training thread");
    let pid = threads.first().unwrap().0;
    ensure!(
        api.iter().all(|r| r["Pid"].as_u64() == Some(pid)),
        "NVTX/API process mismatch"
    );
    let roots: Vec<_> = ranges
        .iter()
        .filter(|(_, (n, _, _))| *n == "warm-training")
        .collect();
    ensure!(
        roots.len() == 1,
        "expected one complete warm-training range"
    );
    let (&root, (_, warm, parent)) = roots[0];
    // Nsight's JSON formatter represents SQL NULL as an empty string.
    ensure!(
        parent.is_null() || *parent == "",
        "warm range must be outermost"
    );
    let updates: Vec<_> = ranges
        .iter()
        .filter(|(_, (n, _, _))| *n == "update")
        .collect();
    ensure!(
        updates.len() == expected && ranges.len() == 1 + expected * (1 + PHASES.len()),
        "incomplete warm phase panel"
    );
    let mut grouped: BTreeMap<&str, Vec<Span>> = BTreeMap::new();
    let contained = |inner: Span, outer: Span| inner.start >= outer.start && inner.end <= outer.end;
    for (id, (_, update, parent)) in updates.iter().copied() {
        ensure!(
            parent.as_u64() == Some(root) && contained(*update, *warm),
            "update outside warm range"
        );
        for phase in PHASES {
            let matching: Vec<_> = ranges
                .values()
                .filter(|(label, _, p)| *label == phase && p.as_u64() == Some(*id))
                .collect();
            ensure!(
                matching.len() == 1 && contained(matching[0].1, *update),
                "missing or misplaced training phase"
            );
            grouped.entry(phase).or_default().push(matching[0].1);
        }
    }
    let disjoint = |spans: Vec<Span>| {
        merged(spans.clone())
            .iter()
            .map(|s| s.duration())
            .sum::<u64>()
            == spans.iter().map(|s| s.duration()).sum::<u64>()
    };
    let update_spans: Vec<_> = updates.iter().map(|(_, (_, span, _))| *span).collect();
    ensure!(disjoint(update_spans.clone()), "overlapping update ranges");
    ensure!(
        disjoint(grouped.values().flatten().copied().collect()),
        "overlapping training phases"
    );
    let mut phases = BTreeMap::new();
    for (name, windows) in grouped {
        phases.insert(name, coverage(gpu, api, &windows)?);
    }
    Ok(
        json!({"updates":expected,"warm_training":coverage(gpu,api,&[*warm])?,
        "update_ranges":coverage(gpu,api,&update_spans)?,"phases":phases,
        "scope":"GPU/API intervals are clipped to complete host NVTX ranges and merged within each measurement. Phase values describe temporal overlap, not kernel ownership: asynchronous work can cross host phase boundaries. Event counts can overlap between phases. Warm capture excludes the first ten updates; the registered diagnostic must disable periodic evaluation/checkpoints. Update ranges end after the trainer's existing device synchronization. Uncovered time is not whole-device idle time or SM occupancy. Profiler overhead is not normal training throughput or process power."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn empty_phases_and_async_boundary_crossings_are_counted_without_double_duration() {
        let gpu = vec![
            json!({"Start (ns)":5,"Duration (ns)":30,"GrdX":1}),
            json!({"Start (ns)":7,"Duration (ns)":4,"GrdX":1}),
        ];
        let api = vec![json!({"Start (ns)":0,"Duration (ns)":40,"Name":"cuEventSynchronize"})];
        let result = coverage(
            &gpu,
            &api,
            &[Span { start: 0, end: 10 }, Span { start: 20, end: 30 }],
        )
        .unwrap();
        assert_eq!(result["gpu_event_union_seconds"], 15e-9);
        assert_eq!(result["host_range_seconds"], 20e-9);
        assert_eq!(result["kernel_events_intersecting_ranges"], 2);
        assert_eq!(result["synchronization_api_union_seconds"], 20e-9);
        let empty = coverage(&gpu, &api, &[Span { start: 50, end: 60 }]).unwrap();
        assert_eq!(empty["gpu_event_coverage_fraction"], 0.);
        assert_eq!(empty["kernel_events_intersecting_ranges"], 0);
    }
    #[test]
    fn incomplete_warm_panels_cannot_be_reported_as_steady_training() {
        let range = |id, parent, name, start, duration| json!({"Start (ns)":start,"End (ns)":start+duration,"Duration (ns)":duration,"Name":format!("burn_gekko:{name}"),"RangeId":id,"ParentId":parent,"PID":42,"TID":42});
        let mut rows = vec![
            range(1, None, "warm-training", 0, 100),
            range(2, Some(1), "update", 5, 90),
        ];
        for (i, p) in PHASES.iter().enumerate() {
            rows.push(range(i + 3, Some(2), p, 10 + i as u64 * 10, 8));
        }
        let api =
            vec![json!({"Start (ns)":0,"Duration (ns)":100,"Name":"cuLaunchKernel","Pid":42})];
        let gpu = vec![json!({"Start (ns)":20,"Duration (ns)":2,"GrdX":1})];
        assert!(analyze(&gpu, &api, &rows, 1).is_ok());
        rows[0]["ParentId"] = json!("");
        assert!(analyze(&gpu, &api, &rows, 1).is_ok());
        rows[0]["ParentId"] = json!(999);
        assert!(analyze(&gpu, &api, &rows, 1).is_err());
        rows[0]["ParentId"] = json!("");
        assert!(analyze(&gpu, &api, &rows, 2).is_err());
        rows[2]["ParentId"] = json!(1);
        assert!(analyze(&gpu, &api, &rows, 1).is_err());
        rows[2]["ParentId"] = json!(2);
        rows.pop();
        assert!(analyze(&gpu, &api, &rows, 1).is_err());
    }
}
