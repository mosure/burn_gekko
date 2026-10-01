//! Verify a fixed assessment prefix after exporter-only changes.
use anyhow::{Context, Result, ensure};
use burn_gekko_data::{read_config, sha256_file, write_json};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Export {
    pub directory: PathBuf,
    pub metrics_sha256: String,
    pub provenance_sha256: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub original: Export,
    pub replay: Export,
    pub checkpoint_sha256: String,
    pub expected_target_views: usize,
    pub latent_absolute_tolerance: f64,
    pub metric_absolute_tolerance: f64,
    pub output: PathBuf,
}
type Identity = (u64, u64);
fn load(e: &Export, checkpoint: &str) -> Result<(BTreeMap<Identity, Value>, Value)> {
    ensure!(
        sha256_file(&e.directory.join("metrics.json"))? == e.metrics_sha256
            && sha256_file(&e.directory.join("provenance.json"))? == e.provenance_sha256,
        "latent replay input changed"
    );
    let m: Value = serde_json::from_slice(&fs::read(e.directory.join("metrics.json"))?)?;
    let p: Value = serde_json::from_slice(&fs::read(e.directory.join("provenance.json"))?)?;
    ensure!(
        p["checkpoint"]["model_sha256"] == checkpoint,
        "latent replay checkpoint differs"
    );
    let config = e
        .directory
        .parent()
        .context("assessment directory")?
        .join("config.toml");
    ensure!(
        p["assessment_config_sha256"] == sha256_file(&config)?,
        "latent replay config changed"
    );
    let config: Value = read_config(&config)?;
    let signature = json!([
        p["dataset_id"],
        p["teacher_id"],
        p["training_mean_sha256"],
        config["seed"],
        config["references"],
        config["mask_ratio"],
        config["mask_pattern"],
        config["stable_attention"],
        m["hidden_tokens"],
        m["visible_tokens"],
        m["split"]
    ]);
    let mut rows = BTreeMap::new();
    for r in m["rows"].as_array().context("latent replay rows")? {
        let key = (
            r["room_seed"].as_u64().context("room seed")?,
            r["target_view"].as_u64().context("target view")?,
        );
        ensure!(
            rows.insert(key, r.clone()).is_none(),
            "duplicate latent replay target"
        );
    }
    ensure!(
        !rows.is_empty() && m["target_views"] == rows.len(),
        "incomplete latent replay population"
    );
    Ok((rows, signature))
}

pub fn verify(c: &Config) -> Result<Value> {
    ensure!(!c.output.exists(), "preserve existing latent replay");
    ensure!(
        c.expected_target_views > 0
            && c.latent_absolute_tolerance.is_finite()
            && c.latent_absolute_tolerance >= 0.
            && c.metric_absolute_tolerance.is_finite()
            && c.metric_absolute_tolerance >= 0.,
        "invalid latent replay limits"
    );
    let (a, sa) = load(&c.original, &c.checkpoint_sha256)?;
    let (b, sb) = load(&c.replay, &c.checkpoint_sha256)?;
    ensure!(
        sa == sb && b.len() == c.expected_target_views && b.keys().all(|k| a.contains_key(k)),
        "latent replay changes inputs/population"
    );
    let mut sources = BTreeMap::new();
    let mut max_array = 0_f64;
    let mut max_metric = 0_f64;
    for (key, b) in &b {
        let a = &a[key];
        for k in [
            "cross_mse",
            "monocular_mse",
            "cross_cosine",
            "monocular_cosine",
            "teacher_signal_power",
            "feature_snr_db",
            "spatial_variance_ratio",
        ] {
            let (a, b) = (
                a[k].as_f64().context("original metric")?,
                b[k].as_f64().context("replayed metric")?,
            );
            ensure!(a.is_finite() && b.is_finite(), "nonfinite replay metric");
            max_metric = max_metric.max((a - b).abs());
        }
        let suffix = format!("room-{}-view-{}", key.0, key.1);
        let dirs = [
            c.original.directory.join(&suffix),
            c.replay.directory.join(&suffix),
        ];
        let mut meta = Vec::new();
        for dir in &dirs {
            let path = dir.join("metadata.json");
            sources.insert(path.clone(), sha256_file(&path)?);
            meta.push(serde_json::from_slice::<Value>(&fs::read(path)?)?);
        }
        ensure!(
            meta[0] == meta[1],
            "latent replay metadata or hidden positions changed"
        );
        let n = meta[0]["latent_shape"][0].as_u64().context("tokens")? as usize;
        let d = meta[0]["latent_shape"][1].as_u64().context("channels")? as usize;
        ensure!(
            n > 0 && d > 0 && meta[0]["room_seed"] == key.0 && meta[0]["target_view"] == key.1,
            "invalid latent replay shape/identity"
        );
        let bytes = n
            .checked_mul(d)
            .and_then(|x| x.checked_mul(4))
            .context("latent shape overflow")?;
        for name in [
            "target-latent.f32",
            "cross-latent.f32",
            "monocular-latent.f32",
        ] {
            let mut arrays = Vec::new();
            for dir in &dirs {
                let path = dir.join(name);
                sources.insert(path.clone(), sha256_file(&path)?);
                let raw = fs::read(path)?;
                ensure!(raw.len() == bytes, "latent replay array shape mismatch");
                arrays.push(raw);
            }
            for (a, b) in arrays[0]
                .as_chunks::<4>()
                .0
                .iter()
                .zip(arrays[1].as_chunks::<4>().0)
            {
                let (a, b) = (f32::from_le_bytes(*a) as f64, f32::from_le_bytes(*b) as f64);
                ensure!(
                    a.is_finite() && b.is_finite(),
                    "nonfinite latent replay array"
                );
                max_array = max_array.max((a - b).abs());
            }
        }
    }
    let result = json!({"schema":1,"passed":max_array<=c.latent_absolute_tolerance && max_metric<=c.metric_absolute_tolerance,"checkpoint_sha256":c.checkpoint_sha256,"compared_target_views":b.len(),"maximum_latent_difference":max_array,"maximum_metric_difference":max_metric,"config":c,"sources":sources,"scope":"Same model, teacher, dataset, target identities, masks, references and baseline. Replay is a declared prefix, not a new accuracy result. Training versus evaluation provenance text may be corrected without modifying numeric outputs."});
    if let Some(parent) = c.output.parent() {
        fs::create_dir_all(parent)?;
    }
    write_json(&c.output, &result)?;
    Ok(result)
}
