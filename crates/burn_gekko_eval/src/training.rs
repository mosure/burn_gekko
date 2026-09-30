//! Exposure counts from the actual update log, restricted to the selected endpoint.
use anyhow::{Context, Result, ensure};
use serde::Serialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Serialize)]
pub struct EncoderStage {
    pub updates: usize,
    pub first_step: u64,
    pub last_step: u64,
    pub min_gradient_tensors: u64,
    pub max_gradient_tensors: u64,
}

/// What actually received encoder gradients, restricted to the selected phase.
/// Legacy logs without either field return no evidence; partial or contradictory
/// evidence fails rather than being interpreted as a frozen encoder.
pub fn encoder_stages(
    rows: &[Value],
    starting_step: u64,
    selected_step: u64,
) -> Result<BTreeMap<u64, EncoderStage>> {
    let rows = rows
        .iter()
        .filter(|r| {
            r["step"]
                .as_u64()
                .is_some_and(|s| s > starting_step && s <= selected_step)
        })
        .collect::<Vec<_>>();
    let mut stages = BTreeMap::<u64, EncoderStage>::new();
    if rows
        .iter()
        .all(|r| r.get("stage").is_none() && r.get("encoder_gradient_tensors").is_none())
    {
        return Ok(stages);
    }
    let mut previous = None;
    for row in rows {
        let step = row["step"]
            .as_u64()
            .context("missing encoder update index")?;
        let stage = row["stage"].as_u64().context("missing encoder stage")?;
        let gradients = row["encoder_gradient_tensors"]
            .as_u64()
            .context("missing encoder gradient count")?;
        ensure!(stage <= 2, "invalid logged encoder stage");
        ensure!(
            (stage == 0) == (gradients == 0),
            "encoder gradients contradict the logged stage"
        );
        if let Some((last_step, last_stage)) = previous {
            ensure!(step == last_step + 1, "noncontiguous encoder trajectory");
            ensure!(
                stage >= last_stage && stage <= last_stage + 1,
                "invalid progressive encoder transition"
            );
        } else {
            ensure!(step == starting_step + 1, "encoder trajectory starts late");
        }
        previous = Some((step, stage));
        let entry = stages.entry(stage).or_insert(EncoderStage {
            updates: 0,
            first_step: step,
            last_step: step,
            min_gradient_tensors: gradients,
            max_gradient_tensors: gradients,
        });
        entry.updates += 1;
        entry.last_step = step;
        entry.min_gradient_tensors = entry.min_gradient_tensors.min(gradients);
        entry.max_gradient_tensors = entry.max_gradient_tensors.max(gradients);
    }
    ensure!(
        previous.is_some_and(|(step, _)| step == selected_step),
        "incomplete encoder trajectory"
    );
    Ok(stages)
}

#[derive(Debug, Serialize)]
pub struct ScalarWindows {
    pub window_updates: usize,
    pub first_mean: f64,
    pub last_mean: f64,
}
/// Descriptive first/last nonoverlapping windows, restricted to the selected
/// phase. These minibatch summaries are not held-out improvement estimates.
pub fn scalar_windows(
    rows: &[Value],
    starting_step: u64,
    selected_step: u64,
    keys: &[&str],
    window: usize,
) -> Result<BTreeMap<String, ScalarWindows>> {
    ensure!(window > 0, "empty training window");
    let rows = rows
        .iter()
        .filter(|r| {
            r["step"]
                .as_u64()
                .is_some_and(|s| s > starting_step && s <= selected_step)
        })
        .collect::<Vec<_>>();
    let count = window.min(rows.len() / 2);
    let mut result = BTreeMap::new();
    if count == 0 {
        return Ok(result);
    }
    for key in keys {
        if rows.iter().all(|r| r.get(key).is_none()) {
            continue;
        }
        let numbers = rows
            .iter()
            .map(|r| {
                r[key]
                    .as_f64()
                    .filter(|v| v.is_finite())
                    .context("missing/nonfinite logged scalar")
            })
            .collect::<Result<Vec<_>>>()?;
        let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
        result.insert(
            (*key).into(),
            ScalarWindows {
                window_updates: count,
                first_mean: mean(&numbers[..count]),
                last_mean: mean(&numbers[numbers.len() - count..]),
            },
        );
    }
    Ok(result)
}

#[derive(Debug, Serialize)]
pub struct Coverage {
    pub updates: usize,
    pub target_exposures: usize,
    pub unique_rooms: usize,
    pub unique_room_views: usize,
    pub configured_rooms: usize,
    pub room_coverage: f64,
    pub mean_exposures_per_configured_room: f64,
}

pub fn coverage(
    rows: &[Value],
    starting_step: u64,
    selected_step: u64,
    batch_size: usize,
    configured_rooms: usize,
) -> Result<Coverage> {
    ensure!(
        selected_step > starting_step && batch_size > 0 && configured_rooms > 0,
        "invalid training coverage contract"
    );
    let mut updates = 0;
    let mut exposures = 0;
    let mut rooms = BTreeSet::new();
    let mut views = BTreeSet::new();
    for row in rows {
        let step = row["step"].as_u64().context("missing update number")?;
        if step <= starting_step || step > selected_step {
            continue;
        }
        ensure!(
            step == starting_step + updates as u64 + 1,
            "missing, duplicate or unordered update"
        );
        let samples = row["samples"]
            .as_array()
            .context("missing target identities")?;
        ensure!(
            samples.len() == batch_size,
            "update batch size differs from config"
        );
        for sample in samples {
            ensure!(
                sample.as_array().is_some_and(|v| v.len() == 2),
                "invalid room/view identity"
            );
            let room = sample[0].as_u64().context("invalid room seed")?;
            let view = sample[1].as_u64().context("invalid view index")?;
            rooms.insert(room);
            views.insert((room, view));
        }
        exposures += samples.len();
        updates += 1;
    }
    ensure!(
        updates as u64 == selected_step - starting_step,
        "incomplete selected trajectory"
    );
    ensure!(
        rooms.len() <= configured_rooms,
        "more logged rooms than configured"
    );
    Ok(Coverage {
        updates,
        target_exposures: exposures,
        unique_rooms: rooms.len(),
        unique_room_views: views.len(),
        configured_rooms,
        room_coverage: rooms.len() as f64 / configured_rooms as f64,
        mean_exposures_per_configured_room: exposures as f64 / configured_rooms as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    #[test]
    fn encoder_evidence_respects_resume_and_rejects_frozen_gradient_leaks() {
        let mut rows = vec![
            json!({"step":1,"stage":0,"encoder_gradient_tensors":0}),
            json!({"step":2,"stage":1,"encoder_gradient_tensors":28}),
            json!({"step":3,"stage":1,"encoder_gradient_tensors":28}),
            json!({"step":4,"stage":2,"encoder_gradient_tensors":180}),
        ];
        let stages = encoder_stages(&rows, 1, 3).unwrap();
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[&1].updates, 2);
        assert_eq!(stages[&1].first_step, 2);
        assert_eq!(stages[&1].last_step, 3);
        assert_eq!(stages[&1].min_gradient_tensors, 28);
        assert!(
            encoder_stages(&[json!({"step":1})], 0, 1)
                .unwrap()
                .is_empty()
        );
        rows[0]["encoder_gradient_tensors"] = json!(1);
        assert!(encoder_stages(&rows, 0, 3).is_err());
        rows[0]["encoder_gradient_tensors"] = json!(0);
        rows[2]["stage"] = json!(0);
        rows[2]["encoder_gradient_tensors"] = json!(0);
        assert!(encoder_stages(&rows, 0, 3).is_err());
        assert!(encoder_stages(&[json!({"step":1,"stage":0})], 0, 1).is_err());
    }
    #[test]
    fn scalar_windows_exclude_parent_and_postselected_updates() {
        let rows = (0..8)
            .map(|step| json!({"step":step,"loss":step as f64}))
            .collect::<Vec<_>>();
        let report = scalar_windows(&rows, 1, 6, &["loss", "absent"], 2).unwrap();
        assert_eq!(report.len(), 1);
        assert_eq!(report["loss"].first_mean, 2.5);
        assert_eq!(report["loss"].last_mean, 5.5);
        assert_eq!(report["loss"].window_updates, 2);
        assert!(
            scalar_windows(
                &[json!({"step":1,"loss":1.}), json!({"step":2})],
                0,
                2,
                &["loss"],
                1
            )
            .is_err()
        );
    }
    #[test]
    fn repeated_exposures_and_endpoint_selection_are_distinct_from_coverage() {
        let mut rows = vec![
            json!({"step":1,"samples":[[10,0],[10,0]]}),
            json!({"step":2,"samples":[[10,1],[11,0]]}),
            json!({"step":3,"samples":[[99,0],[99,1]]}),
        ];
        let c = coverage(&rows, 0, 2, 2, 4).unwrap();
        assert_eq!(c.target_exposures, 4);
        assert_eq!(c.unique_room_views, 3);
        assert_eq!(c.unique_rooms, 2);
        assert_eq!(c.room_coverage, 0.5);
        assert_eq!(coverage(&rows, 1, 2, 2, 4).unwrap().target_exposures, 2);
        assert!(coverage(&rows, 0, 2, 3, 4).is_err());
        rows[1]["step"] = json!(1);
        assert!(coverage(&rows, 0, 2, 2, 4).is_err());
    }
}
