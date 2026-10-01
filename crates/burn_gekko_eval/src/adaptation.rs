//! Selection of matched tail/full encoder screens using validation data only.
use anyhow::{Context, Result, ensure};
use burn_gekko_data::{read_config, sha256_file, write_json};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::PathBuf};

pub mod preservation;
pub mod view_geometry;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Input {
    pub path: PathBuf,
    pub sha256: String,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Arm {
    pub summary: Input,
    pub warp: Input,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionConfig {
    pub tail: Arm,
    pub full: Arm,
    #[serde(default = "screen_updates")]
    pub updates: u64,
    /// A longer matched continuation must verify both original screen parents.
    #[serde(default)]
    pub parents: Option<MatchedParents>,
    pub output: PathBuf,
}
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MatchedParents {
    pub tail: Arm,
    pub full: Arm,
}
fn screen_updates() -> u64 {
    2048
}
#[derive(Debug, Serialize)]
pub struct Measurements {
    pub cross_mse: f64,
    pub monocular_mse: f64,
    pub warp_mean_pixel_error: f64,
    pub warp_pck8: f64,
}
fn number(v: &Value) -> Result<f64> {
    v.as_f64()
        .filter(|v| v.is_finite())
        .context("missing finite selection metric")
}
fn pinned(input: &Input, sources: &mut BTreeMap<PathBuf, String>) -> Result<Value> {
    ensure!(
        sha256_file(&input.path)? == input.sha256,
        "selection input changed"
    );
    sources.insert(input.path.clone(), input.sha256.clone());
    Ok(serde_json::from_slice(&fs::read(&input.path)?)?)
}
type Cohort = BTreeMap<(u64, u64, u64), Vec<(u64, [f64; 2])>>;

struct Evidence {
    metrics: Measurements,
    recipe: Value,
    warp_population: Cohort,
    validation_population: Value,
    sampled_targets: Vec<Value>,
    checkpoint: String,
    run: PathBuf,
}

/// Recompute the selected readout from every recorded point, retaining room and direction weight.
fn warp(v: &Value) -> Result<(f64, f64, Cohort)> {
    ensure!(
        v["width"] == 256 && v["height"] == 256,
        "screen pixel scale changed"
    );
    let mut cohort = Cohort::new();
    let mut error = 0.;
    let mut pck = 0.;
    for r in v["records"]
        .as_array()
        .context("warp records")?
        .iter()
        .filter(|r| r["method"] == "spatial_pair_local")
    {
        let key = (
            r["room_seed"].as_u64().context("room seed")?,
            r["sample"].as_u64().context("sample")?,
            r["direction"].as_u64().context("direction")?,
        );
        ensure!(key.2 < 2 && key.1 < 32, "invalid screen warp population");
        let points = r["score"]["points"].as_array().context("warp points")?;
        ensure!(!points.is_empty(), "empty warp population");
        let mut identities = Vec::new();
        let mut sum = 0.;
        let mut hits = 0;
        for p in points {
            let truth = [number(&p[2][0])?, number(&p[2][1])?];
            let distance = (number(&p[1][0])? - truth[0]).hypot(number(&p[1][1])? - truth[1]);
            ensure!(
                (distance - number(&p[3])?).abs() < 1e-9,
                "warp point error mismatch"
            );
            identities.push((p[0].as_u64().context("query index")?, truth));
            sum += distance;
            hits += usize::from(distance <= 8.);
        }
        let mean = sum / points.len() as f64;
        let fraction = hits as f64 / points.len() as f64;
        ensure!(
            (mean - number(&r["score"]["mean_epe"])?).abs() < 1e-9
                && (fraction - number(&r["score"]["pck_half_patch"])?).abs() < 1e-9,
            "warp row summary mismatch"
        );
        ensure!(
            cohort.insert(key, identities).is_none(),
            "duplicate warp population"
        );
        error += mean;
        pck += fraction;
    }
    ensure!(
        cohort.len() == 64 && v["rooms"] == 32,
        "screen requires all 32 rooms and both directions"
    );
    for &(room, sample, _) in cohort.keys() {
        ensure!(
            cohort.contains_key(&(room, sample, 0)) && cohort.contains_key(&(room, sample, 1)),
            "missing warp direction"
        );
    }
    error /= cohort.len() as f64;
    pck /= cohort.len() as f64;
    ensure!(
        (error - number(&v["methods"]["spatial_pair_local"]["mean_epe"])?).abs() < 1e-9
            && (pck - number(&v["methods"]["spatial_pair_local"]["pck8"])?).abs() < 1e-9,
        "warp aggregate mismatch"
    );
    Ok((error, pck, cohort))
}

fn arm(
    a: &Arm,
    stage: u64,
    updates: u64,
    sources: &mut BTreeMap<PathBuf, String>,
) -> Result<Evidence> {
    let s = pinned(&a.summary, sources)?;
    for (path, hash) in s["sources"]
        .as_object()
        .context("training source closure")?
    {
        let path = PathBuf::from(path);
        let hash = hash.as_str().context("source checksum")?;
        ensure!(sha256_file(&path)? == hash, "training source changed");
        sources.insert(path, hash.into());
    }
    let run = PathBuf::from(s["run"].as_str().context("training run")?);
    for file in [
        "config.toml",
        "report.json",
        "metrics.jsonl",
        "final/model.mpk",
    ] {
        ensure!(
            sources.contains_key(&run.join(file)),
            "incomplete training source closure"
        );
    }
    ensure!(
        s["stop_reason"] == "step_limit" && s["coverage"]["updates"] == updates,
        "incomplete screen"
    );
    let stages = s["encoder_stages"].as_object().context("encoder stages")?;
    let evidence = &s["encoder_stages"][stage.to_string()];
    ensure!(
        stages.len() == 1
            && evidence["updates"] == updates
            && evidence["min_gradient_tensors"] == if stage == 1 { 28 } else { 151 }
            && evidence["max_gradient_tensors"] == evidence["min_gradient_tensors"],
        "unverified adaptation stage"
    );
    let p = &s["parameter_probes"];
    ensure!(
        number(&p["teacher_qkv_max_abs_delta"])? == 0.
            && number(&p["last_encoder_qkv_max_abs_delta"])? > 0.
            && number(&p["prediction_head_max_abs_delta"])? > 0.,
        "teacher or trainable parameter probe failed"
    );
    let first = number(&p["first_encoder_qkv_max_abs_delta"])?;
    ensure!(
        if stage == 1 { first == 0. } else { first > 0. },
        "first encoder probe contradicts stage"
    );
    let mut config: Value = read_config(&run.join("config.toml"))?;
    ensure!(
        config["initial_encoder_stage"] == stage
            && config["encoder_stage_cap"] == stage
            && config["steps"] == updates
            && config["decay_steps"] == updates,
        "configured stage mismatch"
    );
    config
        .as_object_mut()
        .unwrap()
        .remove("initial_encoder_stage");
    config.as_object_mut().unwrap().remove("encoder_stage_cap");
    let report: Value = serde_json::from_slice(&fs::read(run.join("report.json"))?)?;
    let validation = &report["final_validation"];
    let rows = validation["rows"].as_array().context("validation rows")?;
    ensure!(!rows.is_empty(), "empty validation");
    let mean = |key: &str| -> Result<f64> {
        Ok(rows
            .iter()
            .map(|r| number(&r[key]))
            .collect::<Result<Vec<_>>>()?
            .iter()
            .sum::<f64>()
            / rows.len() as f64)
    };
    let cross_mse = mean("cross_mse")?;
    let monocular_mse = mean("monocular_mse")?;
    ensure!(
        (cross_mse - number(&validation["mean_cross_mse"])?).abs() < 1e-9
            && (monocular_mse - number(&validation["mean_monocular_mse"])?).abs() < 1e-9,
        "validation aggregate mismatch"
    );
    let w = pinned(&a.warp, sources)?;
    ensure!(
        w["checkpoint"]["model_sha256"] == s["checkpoint_sha256"],
        "mixed screen checkpoint"
    );
    let (warp_mean_pixel_error, warp_pck8, cohort) = warp(&w)?;
    let sampled_targets = fs::read_to_string(run.join("metrics.jsonl"))?
        .lines()
        .map(|line| {
            let row: Value = serde_json::from_str(line)?;
            ensure!(
                row["samples"].is_array(),
                "missing training sample identities"
            );
            Ok(row["samples"].clone())
        })
        .collect::<Result<Vec<_>>>()?;
    ensure!(
        sampled_targets.len() as u64 == updates,
        "incomplete training sample order"
    );
    let identities = rows
        .iter()
        .map(|r| json!([r["room_seed"], r["target_view"]]))
        .collect::<Vec<_>>();
    Ok(Evidence {
        metrics: Measurements {
            cross_mse,
            monocular_mse,
            warp_mean_pixel_error,
            warp_pck8,
        },
        recipe: config,
        warp_population: cohort,
        validation_population: json!({"targets":identities,"hidden_tokens":validation["hidden_tokens"]}),
        sampled_targets,
        checkpoint: s["checkpoint_sha256"]
            .as_str()
            .context("checkpoint hash")?
            .into(),
        run,
    })
}

pub fn gates(tail: &Measurements, full: &Measurements) -> Value {
    let mse = full.cross_mse <= tail.cross_mse * 1.01;
    let references = full.cross_mse < full.monocular_mse;
    let error = full.warp_mean_pixel_error < tail.warp_mean_pixel_error;
    let pck = full.warp_pck8 >= tail.warp_pck8;
    json!({"latent_mse_within_one_percent":mse,"references_help":references,"lower_warp_pixel_error":error,"nondecreasing_warp_pck8":pck,"passed":mse && references && error && pck})
}
fn matched(tail: &Evidence, full: &Evidence) -> Result<()> {
    ensure!(
        tail.recipe == full.recipe,
        "screens differ beyond encoder stage"
    );
    ensure!(
        tail.warp_population == full.warp_population,
        "warp sample/query/transform populations differ"
    );
    ensure!(
        tail.validation_population == full.validation_population,
        "validation targets or hidden mask differ"
    );
    ensure!(
        tail.sampled_targets == full.sampled_targets,
        "training sample order differs"
    );
    Ok(())
}
fn link_parent(child: &mut Evidence, parent: &Evidence) -> Result<()> {
    let start = &child.recipe["warm_start"];
    ensure!(
        start["model_sha256"] == parent.checkpoint,
        "continuation parent checksum differs"
    );
    let path = start["checkpoint"]
        .as_str()
        .context("continuation parent path")?;
    ensure!(
        fs::canonicalize(path)? == fs::canonicalize(parent.run.join("final"))?,
        "continuation parent path differs"
    );
    child.recipe.as_object_mut().unwrap().remove("warm_start");
    Ok(())
}
pub fn select(c: &SelectionConfig) -> Result<Value> {
    ensure!(!c.output.exists(), "preserve existing screen selection");
    ensure!(
        c.updates == 2048 || (c.updates == 12000 && c.parents.is_some()),
        "selection requires a registered screen or linked recovery contract"
    );
    ensure!(
        c.updates != 2048 || c.parents.is_none(),
        "screen cannot override its common parent"
    );
    let mut sources = BTreeMap::new();
    let mut tail = arm(&c.tail, 1, c.updates, &mut sources)?;
    let mut full = arm(&c.full, 2, c.updates, &mut sources)?;
    if let Some(parents) = &c.parents {
        let parent_tail = arm(&parents.tail, 1, 2048, &mut sources)?;
        let parent_full = arm(&parents.full, 2, 2048, &mut sources)?;
        matched(&parent_tail, &parent_full)?;
        link_parent(&mut tail, &parent_tail)?;
        link_parent(&mut full, &parent_full)?;
    }
    matched(&tail, &full)?;
    let gates = gates(&tail.metrics, &full.metrics);
    let accepted = gates["passed"] == true;
    let out = json!({"schema":1,"updates_per_phase":c.updates,"linked_screen_parents":c.parents.is_some(),"selected_arm":if accepted {"full"}else{"tail"},"selected_checkpoint_sha256":if accepted {full.checkpoint}else{tail.checkpoint},"tail":tail.metrics,"full":full.metrics,"gates":gates,"sources":sources,"scope":"Registered matched validation-only adaptation selection. All stages, teacher probes, configured horizons and sample populations verified; continuation contracts also verify both matched screen parents. External benchmarks do not select the arm. This is a development selection, not independent generalization evidence."});
    write_json(&c.output, &out)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn continuation_cannot_substitute_a_different_screen_parent() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("final")).unwrap();
        let evidence = || Evidence {
            metrics: Measurements {
                cross_mse: 0.2,
                monocular_mse: 0.3,
                warp_mean_pixel_error: 4.,
                warp_pck8: 0.8,
            },
            recipe: json!({}),
            warp_population: Cohort::new(),
            validation_population: json!([]),
            sampled_targets: vec![],
            checkpoint: "parent-checksum".into(),
            run: root.path().into(),
        };
        let parent = evidence();
        let mut child = evidence();
        child.recipe =
            json!({"warm_start":{"checkpoint":root.path().join("final"),"model_sha256":"wrong"}});
        assert!(link_parent(&mut child, &parent).is_err());
        child.recipe["warm_start"]["model_sha256"] = json!(parent.checkpoint);
        child.recipe["warm_start"]["checkpoint"] = json!(root.path());
        assert!(link_parent(&mut child, &parent).is_err());
        child.recipe["warm_start"]["checkpoint"] = json!(root.path().join("final"));
        link_parent(&mut child, &parent).unwrap();
        assert!(child.recipe.get("warm_start").is_none());
    }
    #[test]
    fn improved_warp_cannot_hide_completion_regression_or_missing_reference_benefit() {
        let tail = Measurements {
            cross_mse: 0.2,
            monocular_mse: 0.22,
            warp_mean_pixel_error: 8.,
            warp_pck8: 0.7,
        };
        let mut full = Measurements {
            cross_mse: 0.205,
            monocular_mse: 0.22,
            warp_mean_pixel_error: 7.,
            warp_pck8: 0.71,
        };
        assert_eq!(gates(&tail, &full)["passed"], false);
        full.cross_mse = 0.201;
        assert_eq!(gates(&tail, &full)["passed"], true);
        full.monocular_mse = 0.2;
        assert_eq!(gates(&tail, &full)["passed"], false);
    }
    #[test]
    fn missing_directions_and_changed_error_records_are_rejected() {
        let row = json!({"method":"spatial_pair_local","room_seed":7,"sample":0,"direction":0,"score":{"points":[[0,[8.,8.],[10.,8.],2.]],"mean_epe":2.,"pck_half_patch":1.}});
        let mut v = json!({"width":256,"height":256,"rooms":32,"records":[row]});
        assert!(warp(&v).is_err());
        v["records"][0]["score"]["points"][0][3] = json!(0.);
        assert!(warp(&v).unwrap_err().to_string().contains("point error"));
    }
}
