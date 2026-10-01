//! Shared verification of one frozen-foundation output-head training run.
use super::{HeadRow, HeadScores, RgbScore, camera_score, rgb_score, summarize};
use crate::adaptation::Input;
use anyhow::{Context, Result, ensure};
use burn_gekko_data::{Split, head_cache::HeadCache, sha256_file};
use serde_json::Value;
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

pub struct AuditedHeads {
    pub cache: HeadCache,
    pub report: Value,
    pub steps: Vec<Value>,
    pub scores: HeadScores,
}
fn record(path: &Path, sources: &mut BTreeMap<PathBuf, String>) -> Result<String> {
    let hash = sha256_file(path)?;
    sources.insert(path.into(), hash.clone());
    Ok(hash)
}
fn pinned(input: &Input, sources: &mut BTreeMap<PathBuf, String>) -> Result<Value> {
    ensure!(
        record(&input.path, sources)? == input.sha256,
        "head input changed"
    );
    Ok(serde_json::from_slice(&fs::read(&input.path)?)?)
}
fn number(v: &Value, key: &str) -> Result<f64> {
    v[key]
        .as_f64()
        .filter(|v| v.is_finite())
        .context("nonfinite head scalar")
}
fn floats(path: &Path, sources: &mut BTreeMap<PathBuf, String>) -> Result<Vec<f32>> {
    record(path, sources)?;
    let bytes = fs::read(path)?;
    ensure!(bytes.len().is_multiple_of(4), "invalid head float array");
    let values = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|x| f32::from_le_bytes(*x))
        .collect::<Vec<_>>();
    ensure!(
        values.iter().all(|x| x.is_finite()),
        "nonfinite head float array"
    );
    Ok(values)
}
fn file(v: &Value) -> Result<Input> {
    Ok(serde_json::from_value(v.clone())?)
}
pub(super) fn constant_prediction(cache: &HeadCache) -> Result<Vec<f32>> {
    let train = cache
        .samples
        .iter()
        .filter(|s| s.split == Split::Train)
        .collect::<Vec<_>>();
    ensure!(!train.is_empty(), "missing training-label camera control");
    Ok((0..11)
        .map(|i| {
            train
                .iter()
                .map(|s| s.camera.regression()[i] as f64)
                .sum::<f64>() as f32
                / train.len() as f32
        })
        .collect())
}
fn json_file(v: &Value, sources: &mut BTreeMap<PathBuf, String>) -> Result<Value> {
    pinned(&file(v)?, sources)
}
fn bound(input: &Input, sources: &mut BTreeMap<PathBuf, String>) -> Result<Value> {
    let v = pinned(input, sources)?;
    ensure!(v["schema"] == 1, "invalid output-head evidence");
    for value in v["files"]
        .as_object()
        .context("head evidence sources")?
        .values()
    {
        let p = file(value)?;
        ensure!(
            record(&p.path, sources)? == p.sha256,
            "head source checksum mismatch"
        );
    }
    Ok(v)
}
pub(super) fn close(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(a), Value::Number(b)) => a
            .as_f64()
            .zip(b.as_f64())
            .is_some_and(|(a, b)| (a - b).abs() < 1e-9),
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len() && a.iter().all(|(k, v)| b.get(k).is_some_and(|x| close(v, x)))
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(a, b)| close(a, b))
        }
        _ => a == b,
    }
}
pub fn load(
    input: &Input,
    checkpoint: &str,
    sources: &mut BTreeMap<PathBuf, String>,
) -> Result<AuditedHeads> {
    let v = bound(input, sources)?;
    ensure!(
        v["checkpoint_sha256"] == checkpoint,
        "output heads belong to another foundation checkpoint"
    );
    let files = &v["files"];
    let report = json_file(&files["report"], sources)?;
    let meta = json_file(&files["metadata"], sources)?;
    let provenance = json_file(&files["provenance"], sources)?;
    let predictions = json_file(&files["predictions"], sources)?;
    let recipe: Value = burn_gekko_data::read_config(&file(&files["config"])?.path)?;
    let cache: HeadCache = serde_json::from_value(json_file(&files["cache_manifest"], sources)?)?;
    cache.validate()?;
    for obj in [&report, &meta, &provenance, &predictions] {
        ensure!(
            obj["checkpoint_sha256"] == checkpoint,
            "mixed output-head checkpoint"
        );
    }
    ensure!(
        cache.checkpoint_sha256 == checkpoint
            && meta["identity"] == provenance["identity"]
            && meta["completed_steps"] == report["completed_steps"]
            && meta["files"]["camera"] == files["camera_weights"]["sha256"]
            && meta["files"]["rgb"] == files["rgb_weights"]["sha256"],
        "head weights/training identity mismatch"
    );
    ensure!(
        provenance["cache_sha256"] == files["cache_manifest"]["sha256"]
            && recipe["cache_sha256"] == provenance["cache_sha256"]
            && recipe["checkpoint_sha256"] == checkpoint,
        "head training cache differs"
    );
    let cache_config =
        PathBuf::from(recipe["cache"].as_str().context("head cache path")?).join("config.toml");
    ensure!(
        record(&cache_config, sources)? == cache.config_sha256,
        "head cache recipe changed"
    );
    ensure!(
        provenance["frozen_foundation"] == true
            && provenance["encoder_gradients"] == false
            && provenance["noncommercial_weight_dependencies"]
                .as_array()
                .is_some_and(Vec::is_empty),
        "invalid frozen-foundation head provenance"
    );
    let steps = fs::read_to_string(file(&files["steps"])?.path)?
        .lines()
        .map(|line| Ok(serde_json::from_str::<Value>(line)?))
        .collect::<Result<Vec<_>>>()?;
    let start = report["starting_step"].as_u64().context("head start")?;
    ensure!(
        steps.len() as u64 + start == report["completed_steps"].as_u64().context("head end")?,
        "head update count differs"
    );
    for (i, row) in steps.iter().enumerate() {
        ensure!(
            row["step"] == start + i as u64 + 1,
            "head step sequence differs"
        );
        for key in [
            "camera_loss",
            "rgb_loss",
            "camera_gradient_norm",
            "rgb_gradient_norm",
            "learning_rate",
        ] {
            ensure!(number(row, key)? >= 0., "invalid head optimization value");
        }
    }
    let expected: BTreeMap<_, _> = cache
        .samples
        .iter()
        .filter(|s| s.split == Split::Validation)
        .map(|s| ((s.room_seed, s.target_view), s))
        .collect();
    let mut seen = BTreeSet::new();
    let mut rows = Vec::new();
    for sample in predictions["samples"].as_array().context("head examples")? {
        let id = (
            sample["room_seed"].as_u64().context("room seed")?,
            sample["target_view"].as_u64().context("target view")? as usize,
        );
        let target = expected
            .get(&id)
            .context("head target outside validation split")?;
        ensure!(seen.insert(id), "duplicate output-head target");
        let hidden: Vec<usize> = serde_json::from_value(sample["hidden_tokens"].clone())?;
        ensure!(
            hidden == target.hidden_tokens
                && close(
                    &sample["camera_target"],
                    &serde_json::to_value(&target.camera)?
                ),
            "head labels/mask changed"
        );
        let mut arrays = Vec::new();
        for key in ["prediction", "monocular", "target"] {
            let p = file(&sample["files"][key])?;
            ensure!(
                record(&p.path, sources)? == p.sha256,
                "head prediction changed"
            );
            if key == "target" {
                ensure!(
                    p.sha256 == target.rgb.sha256,
                    "head target RGB differs from cache"
                );
            }
            arrays.push(floats(&p.path, sources)?);
        }
        let cp: Vec<f32> = serde_json::from_value(sample["metrics"]["camera_prediction"].clone())?;
        let row = HeadRow {
            room_seed: id.0,
            target_view: id.1,
            reference_view: target.reference_view,
            camera: camera_score(&cp, &target.camera)?,
            rgb: rgb_score(&arrays[0], &arrays[2], &hidden)?,
            monocular: rgb_score(&arrays[1], &arrays[2], &hidden)?,
            camera_prediction: cp,
        };
        ensure!(
            close(&serde_json::to_value(&row)?, &sample["metrics"]),
            "head metrics differ from raw predictions"
        );
        rows.push(row);
    }
    ensure!(
        seen.len() == expected.len(),
        "incomplete output-head evaluation"
    );
    let scores = summarize(rows)?;
    ensure!(
        close(&serde_json::to_value(&scores)?, &report["validation"])
            && close(&serde_json::to_value(&scores)?, &predictions["scores"]),
        "head aggregate differs from observations"
    );
    let constant = constant_prediction(&cache)?;
    let control = summarize(
        cache
            .samples
            .iter()
            .filter(|s| s.split == Split::Validation)
            .map(|s| {
                Ok(HeadRow {
                    room_seed: s.room_seed,
                    target_view: s.target_view,
                    reference_view: s.reference_view,
                    camera: camera_score(&constant, &s.camera)?,
                    camera_prediction: constant.clone(),
                    rgb: RgbScore {
                        mse: 0.,
                        psnr_db: None,
                    },
                    monocular: RgbScore {
                        mse: 0.,
                        psnr_db: None,
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?,
    )?;
    ensure!(
        close(
            &serde_json::to_value(control)?,
            &report["constant_camera_validation"]
        ),
        "camera constant differs from training-only labels"
    );
    Ok(AuditedHeads {
        cache,
        report,
        steps,
        scores,
    })
}
