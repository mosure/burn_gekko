//! Sustained continuation with completion, spatial-detail and camera retention.
use super::*;
use crate::pose::synthetic_comparison;
use crate::statistics::bootstrap_mean;

pub mod completion;
pub mod forecast;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Evaluation {
    pub geometry: Input,
    pub pose: Input,
    pub completion: completion::InputSet,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingArm {
    pub registered_config: Input,
    pub summary: Input,
    pub evaluation: Evaluation,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub parent_checkpoint_sha256: String,
    pub parent: Evaluation,
    pub control: TrainingArm,
    pub candidate: TrainingArm,
    pub updates: u64,
    pub completion_rooms: usize,
    pub expected_weight: f64,
    pub protocol: Input,
    pub output: PathBuf,
}

#[derive(Clone, Serialize)]
pub struct Measurements {
    #[serde(flatten)]
    pub completion: completion::Measurements,
    pub view_aepe_pixels: f64,
    pub view_pck8: f64,
    pub mean_pose_auc10: f64,
}

pub fn gates(parent: &Measurements, control: &Measurements, candidate: &Measurements) -> Value {
    let c = &candidate.completion;
    let b = &control.completion;
    let checks = [
        (
            "latent_mse_within_one_percent_of_better_baseline",
            c.cross_mse <= parent.completion.cross_mse.min(b.cross_mse) * 1.01,
        ),
        ("references_help", c.cross_mse < c.monocular_mse),
        (
            "at_least_five_percent_lower_view_error_than_both",
            candidate.view_aepe_pixels
                <= parent.view_aepe_pixels.min(control.view_aepe_pixels) * 0.95,
        ),
        (
            "nondecreasing_view_pck8_against_both",
            candidate.view_pck8 >= parent.view_pck8.max(control.view_pck8),
        ),
        (
            "centered_mse_within_one_percent",
            c.centered_mse <= b.centered_mse * 1.01,
        ),
        (
            "spatial_correlation_retained",
            c.spatial_correlation >= b.spatial_correlation - 0.005,
        ),
        (
            "adjacent_correlation_retained",
            c.adjacent_correlation >= b.adjacent_correlation - 0.005,
        ),
        (
            "mean_pose_auc10_retained_against_both",
            candidate.mean_pose_auc10 >= parent.mean_pose_auc10.max(control.mean_pose_auc10),
        ),
    ];
    let mut map: serde_json::Map<String, Value> = checks
        .iter()
        .map(|(k, v)| (k.to_string(), json!(v)))
        .collect();
    map.insert("passed".into(), json!(checks.iter().all(|(_, v)| *v)));
    Value::Object(map)
}

fn training(
    a: &TrainingArm,
    updates: u64,
    sources: &mut BTreeMap<PathBuf, String>,
) -> Result<(Evidence, Value)> {
    let evidence = arm(
        &Arm {
            summary: Input {
                path: a.summary.path.clone(),
                sha256: a.summary.sha256.clone(),
            },
            warp: Input {
                path: a.evaluation.geometry.path.clone(),
                sha256: a.evaluation.geometry.sha256.clone(),
            },
        },
        2,
        updates,
        sources,
    )?;
    ensure!(
        sha256_file(&a.registered_config.path)? == a.registered_config.sha256,
        "registered training recipe changed"
    );
    sources.insert(
        a.registered_config.path.clone(),
        a.registered_config.sha256.clone(),
    );
    let mut registered: Value = read_config(&a.registered_config.path)?;
    for key in ["initial_encoder_stage", "encoder_stage_cap"] {
        ensure!(registered[key] == 2, "registered adaptation stage differs");
        registered
            .as_object_mut()
            .context("registered recipe")?
            .remove(key);
    }
    ensure!(
        registered
            .as_object()
            .context("registered recipe")?
            .iter()
            .all(|(k, v)| evidence.recipe.get(k) == Some(v)),
        "executed training recipe differs from registration"
    );
    let s = pinned(&a.summary, sources)?;
    ensure!(
        s["coverage"]["configured_rooms"] == 8192
            && s["coverage"]["unique_rooms"] == 8192
            && s["coverage"]["target_exposures"] == updates * 16
            && s["parameter_probes"]["preservation_anchor_qkv_max_abs_delta"] == json!([0., 0.]),
        "incomplete full-cohort coverage or changed preservation anchor"
    );
    for line in fs::read_to_string(evidence.run.join("metrics.jsonl"))?.lines() {
        let row: Value = serde_json::from_str(line)?;
        for key in [
            "total",
            "cross",
            "monocular",
            "visible",
            "ri",
            "warp_pair_nll",
            "warp_self_nll",
            "encoder_preservation_mse",
            "view_geometry_nll",
            "gradient_norm",
            "learning_rate",
        ] {
            ensure!(
                number(&row[key])? >= 0.,
                "invalid continuation training scalar"
            );
        }
        if evidence.recipe.get("view_geometry").is_some() {
            ensure!(
                number(&row["view_geometry_nll"])? > 0.
                    && number(&row["view_geometry_valid_fraction"])? > 0.,
                "inactive geometry update"
            );
        }
    }
    Ok((evidence, s))
}

struct EvaluationEvidence {
    measured: Measurements,
    completion: completion::Evidence,
    geometry: Value,
    pose: Value,
    pose_rows: synthetic_comparison::Population,
}
fn evaluation(
    c: &Evaluation,
    checkpoint: &str,
    dataset: &Value,
    rooms: usize,
    sources: &mut BTreeMap<PathBuf, String>,
) -> Result<EvaluationEvidence> {
    let geometry = pinned(&c.geometry, sources)?;
    ensure!(
        geometry["task"] == "renderer_viewpoint_correspondence"
            && geometry["checkpoint"]["model_sha256"] == checkpoint,
        "view evaluation identity differs"
    );
    let (aepe, pck, _) = warp(&geometry)?;
    let completion = completion::load(&c.completion, checkpoint, rooms, sources)?;
    ensure!(
        completion.assessment_config["dataset"] == *dataset,
        "completion dataset differs"
    );
    let (pose, pose_rows) = synthetic_comparison::load(&synthetic_comparison::Arm {
        report: c.pose.path.clone(),
        sha256: c.pose.sha256.clone(),
        method: "spatial_pair_local".into(),
    })?;
    sources.insert(c.pose.path.clone(), c.pose.sha256.clone());
    for (p, h) in pose["inputs"].as_object().context("pose source closure")? {
        sources.insert(
            PathBuf::from(p),
            h.as_str().context("pose input checksum")?.into(),
        );
    }
    let p = &pose["config"];
    ensure!(
        pose["checkpoint_sha256"] == checkpoint
            && p["dataset"] == *dataset
            && p["rooms"] == 32
            && p["seeds"] == json!([871, 872, 873, 874, 875, 876, 877, 878])
            && p["methods"]
                == json!([
                    "spatial_pair_local",
                    "spatial_self_local",
                    "spatial_encoder_local"
                ])
            && p["max_trials"] == 2048
            && p["min_trials"] == 64
            && p["confidence"] == 0.999
            && p["min_inliers"] == 12
            && p["threshold_pixels"] == 1.5
            && p["minimum_baseline_meters"] == 0.01,
        "synthetic pose protocol differs from registration"
    );
    let mut aucs = Vec::new();
    for &seed in &[871, 872, 873, 874, 875, 876, 877, 878] {
        let rows = pose_rows
            .iter()
            .filter(|((s, _), _)| *s == seed)
            .map(|(_, v)| v)
            .collect::<Vec<_>>();
        aucs.push(
            crate::pose::benchmark::summary(&rows)?
                .pose_auc_10
                .context("undefined pose AUC")?,
        );
    }
    let auc = aucs.iter().sum::<f64>() / aucs.len() as f64;
    ensure!(
        (auc - number(&pose["methods"]["spatial_pair_local"]["mean_pose_auc_10"])?).abs() < 1e-12,
        "pose aggregate differs from complete solver panel"
    );
    Ok(EvaluationEvidence {
        measured: Measurements {
            completion: completion.mean.clone(),
            view_aepe_pixels: aepe,
            view_pck8: pck,
            mean_pose_auc10: auc,
        },
        completion,
        geometry,
        pose,
        pose_rows,
    })
}

fn compare(a: &EvaluationEvidence, b: &EvaluationEvidence) -> Result<Value> {
    ensure!(
        warp(&a.geometry)?.2 == warp(&b.geometry)?.2
            && a.geometry["cache_manifest_sha256"] == b.geometry["cache_manifest_sha256"],
        "view correspondence cohorts or labels differ"
    );
    let errors = |g: &Value| -> Result<BTreeMap<u64, Vec<f64>>> {
        let mut rooms: BTreeMap<u64, Vec<f64>> = BTreeMap::new();
        for row in g["records"]
            .as_array()
            .context("view records")?
            .iter()
            .filter(|r| r["method"] == "spatial_pair_local")
        {
            rooms
                .entry(row["room_seed"].as_u64().context("view room seed")?)
                .or_default()
                .push(number(&row["score"]["mean_epe"])?);
        }
        Ok(rooms)
    };
    let ar = errors(&a.geometry)?;
    let br = errors(&b.geometry)?;
    let gains = ar
        .iter()
        .map(|(k, v)| {
            br[k].iter().sum::<f64>() / br[k].len() as f64 - v.iter().sum::<f64>() / v.len() as f64
        })
        .collect::<Vec<_>>();
    Ok(
        json!({"completion":completion::compare(&a.completion,&b.completion)?,
        "view_aepe_reduction_pixels":bootstrap_mean(&gains,853)?,
        "camera":synthetic_comparison::paired(&a.pose_rows,&b.pose_rows)?}),
    )
}

pub fn select(c: &Config) -> Result<Value> {
    ensure!(
        !c.output.exists()
            && c.updates == 4096
            && c.completion_rooms == 64
            && c.expected_weight.is_finite()
            && c.expected_weight > 0.
            && c.expected_weight <= 1.,
        "unregistered continuation or existing selection"
    );
    ensure!(
        sha256_file(&c.protocol.path)? == c.protocol.sha256,
        "registered protocol changed"
    );
    let mut sources = BTreeMap::from([(c.protocol.path.clone(), c.protocol.sha256.clone())]);
    let (control, cs) = training(&c.control, c.updates, &mut sources)?;
    let (mut candidate, gs) = training(&c.candidate, c.updates, &mut sources)?;
    ensure!(
        cs["source_sha256"].is_string()
            && cs["source_sha256"] == gs["source_sha256"]
            && cs["teacher_id"] == gs["teacher_id"],
        "training source or teacher differs"
    );
    ensure!(
        control.recipe.get("view_geometry").is_none(),
        "control uses geometry auxiliary"
    );
    let objective = candidate
        .recipe
        .as_object_mut()
        .unwrap()
        .remove("view_geometry")
        .context("candidate geometry objective")?;
    ensure!(
        number(&objective["weight"])? == c.expected_weight
            && number(&objective["temperature"])? == 0.07
            && gs["view_geometry"]["config"] == objective,
        "geometry recipe differs"
    );
    matched(&control, &candidate)?;
    ensure!(
        control.recipe["warm_start"]["model_sha256"] == c.parent_checkpoint_sha256
            && control.recipe["seed"] == 853
            && control.recipe["warmup_steps"] == 128
            && control.recipe["references"] == 2,
        "continuation initialization or schedule differs"
    );
    let dataset = &control.recipe["dataset"];
    let parent = evaluation(
        &c.parent,
        &c.parent_checkpoint_sha256,
        dataset,
        c.completion_rooms,
        &mut sources,
    )?;
    let control_eval = evaluation(
        &c.control.evaluation,
        &control.checkpoint,
        dataset,
        c.completion_rooms,
        &mut sources,
    )?;
    let candidate_eval = evaluation(
        &c.candidate.evaluation,
        &candidate.checkpoint,
        dataset,
        c.completion_rooms,
        &mut sources,
    )?;
    ensure!(
        candidate_eval.geometry["cache_manifest_sha256"]
            == gs["view_geometry"]["cache_manifest_sha256"],
        "geometry training/evaluation cache differs"
    );
    let against_parent = compare(&candidate_eval, &parent)?;
    let against_control = compare(&candidate_eval, &control_eval)?;
    let decision = gates(
        &parent.measured,
        &control_eval.measured,
        &candidate_eval.measured,
    );
    let passed = decision["passed"] == true;
    let result = json!({"schema":1,"updates_per_arm":c.updates,
        "selected_arm":if passed {"geometry"}else{"published_parent"},
        "selected_checkpoint_sha256":if passed {&candidate.checkpoint}else{&c.parent_checkpoint_sha256},
        "parent":parent.measured,"control":control_eval.measured,"candidate":candidate_eval.measured,
        "gates":decision,"against_parent":against_parent,"against_control":against_control,
        "pose_seed_ranges":{"parent":parent.pose["methods"],"control":control_eval.pose["methods"],"candidate":candidate_eval.pose["methods"]},
        "sources":sources,"scope":"Registered synthetic development retention checks, all conjunctive. Failure retains the published parent; it does not promote the control by default. Full fixed horizons and encoder gradients verified. Shared standalone completion masks, teacher arrays, view labels and all eight pose seeds retained. Reused external benchmarks cannot select the arm. Not independent qualification or SotA evidence."});
    write_json(&c.output, &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn baseline() -> Measurements {
        Measurements {
            completion: completion::Measurements {
                cross_mse: 0.2,
                monocular_mse: 0.24,
                centered_mse: 0.1,
                spatial_correlation: 0.4,
                adjacent_correlation: 0.38,
                adjacent_power_ratio: 0.17,
            },
            view_aepe_pixels: 12.,
            view_pck8: 0.6,
            mean_pose_auc10: 0.21,
        }
    }
    #[test]
    fn matching_gain_cannot_hide_detail_or_camera_regression() {
        let b = baseline();
        let mut a = b.clone();
        a.view_aepe_pixels = 10.;
        a.view_pck8 = 0.65;
        assert_eq!(gates(&b, &b, &a)["passed"], true);
        let good = a.clone();
        a.completion.adjacent_power_ratio = 1.;
        a.completion.adjacent_correlation = 0.37;
        assert_eq!(gates(&b, &b, &a)["passed"], false);
        a = good.clone();
        a.mean_pose_auc10 = 0.20;
        assert_eq!(gates(&b, &b, &a)["passed"], false);
        a = good;
        a.completion.centered_mse = 0.102;
        assert_eq!(gates(&b, &b, &a)["passed"], false);
    }
    #[test]
    fn retention_uses_better_parent_or_control_and_requires_reference_benefit() {
        let p = baseline();
        let mut b = p.clone();
        b.mean_pose_auc10 = 0.25;
        let mut a = p.clone();
        a.view_aepe_pixels = 10.;
        assert_eq!(gates(&p, &b, &a)["passed"], false);
        a.mean_pose_auc10 = 0.25;
        a.completion.monocular_mse = a.completion.cross_mse;
        assert_eq!(gates(&p, &b, &a)["passed"], false);
        a.completion.monocular_mse = 0.24;
        assert_eq!(gates(&p, &b, &a)["passed"], true);
    }
}
