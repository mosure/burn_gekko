//! Information-set sensitivity of one checkpoint, on exactly the same masked targets.
use crate::schema::{Capability, CapabilityStatus, Metric};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::PathBuf,
};
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Assessment {
    pub references: usize,
    pub directory: PathBuf,
    pub metrics_sha256: String,
    pub provenance_sha256: String,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub checkpoint_sha256: String,
    pub evaluations: Vec<Assessment>,
    pub output: PathBuf,
}
pub fn score(c: &Config) -> Result<()> {
    ensure!(
        !c.output.exists() && !c.output.with_extension("provenance.json").exists(),
        "preserve existing reference-count results"
    );
    ensure!(
        c.evaluations.len() >= 2
            && c.evaluations
                .iter()
                .map(|e| e.references)
                .collect::<BTreeSet<_>>()
                .len()
                == c.evaluations.len(),
        "need distinct reference counts"
    );
    let mut tables = Vec::new();
    let mut signature = None;
    for e in &c.evaluations {
        ensure!(e.references > 0, "reference count must be positive");
        ensure!(
            burn_gekko_data::sha256_file(&e.directory.join("metrics.json"))? == e.metrics_sha256
                && burn_gekko_data::sha256_file(&e.directory.join("provenance.json"))?
                    == e.provenance_sha256,
            "reference assessment checksum mismatch"
        );
        let metrics: Value = serde_json::from_slice(&fs::read(e.directory.join("metrics.json"))?)?;
        let proof: Value = serde_json::from_slice(&fs::read(e.directory.join("provenance.json"))?)?;
        ensure!(
            proof["checkpoint"]["model_sha256"] == c.checkpoint_sha256,
            "reference sweep mixes checkpoints"
        );
        let config_path = e
            .directory
            .parent()
            .context("assessment parent")?
            .join("config.toml");
        ensure!(
            burn_gekko_data::sha256_file(&config_path)?
                == proof["assessment_config_sha256"]
                    .as_str()
                    .context("assessment config hash")?,
            "reference config changed"
        );
        let config: Value = burn_gekko_data::read_config(&config_path)?;
        ensure!(
            config["references"].as_u64() == Some(e.references as u64),
            "reference count label differs from actual inference"
        );
        let current = json!([
            proof["dataset_id"],
            metrics["visible_tokens"],
            metrics["hidden_tokens"],
            metrics["split"],
            config["seed"],
            config["mask_pattern"]
        ]);
        if let Some(prior) = &signature {
            ensure!(
                prior == &current,
                "reference sweep changes masks, data or split"
            );
        } else {
            signature = Some(current);
        }
        let mut rows = BTreeMap::new();
        for row in metrics["rows"].as_array().context("assessment rows")? {
            let key = (
                row["room_seed"].as_u64().context("room seed")?,
                row["target_view"].as_u64().context("target view")?,
            );
            let cross = row["cross_mse"]
                .as_f64()
                .filter(|x| x.is_finite())
                .context("cross MSE")?;
            let mono = row["monocular_mse"]
                .as_f64()
                .filter(|x| x.is_finite())
                .context("monocular MSE")?;
            ensure!(
                rows.insert(key, (cross, mono)).is_none(),
                "duplicate target view"
            );
        }
        tables.push((e.references, rows));
    }
    let mut common = tables[0].1.keys().copied().collect::<BTreeSet<_>>();
    for (_, rows) in &tables {
        common.retain(|key| rows.contains_key(key));
    }
    ensure!(!common.is_empty(), "reference sweep has no shared targets");
    let mut metrics = Vec::new();
    for (references, rows) in &tables {
        let mut sum = 0.;
        for key in &common {
            let (cross, mono) = rows[key];
            ensure!(
                (mono - tables[0].1[key].1).abs() < 1e-7,
                "reference count affected isolated monocular prediction"
            );
            sum += cross;
        }
        metrics.push(Metric {
            id: format!("references_{references}_mse"),
            label: format!("{references} reference view(s) · hidden MSE"),
            value: sum / common.len() as f64,
            unit: "squared normalized feature units".into(),
            lower_is_better: true,
            samples: common.len(),
            aggregation:
                "equal target-view weighting on the exact intersection of room/view identities"
                    .into(),
        });
    }
    metrics.sort_by(|a, b| a.id.cmp(&b.id));
    let capability=Capability {id:"reference_count".into(),label:"Multi-view information sensitivity".into(),status:CapabilityStatus::Evaluated,protocol:"One checkpoint, same data/mask/target identities; only reference count changes. Monocular isolation verified at 1e-7.".into(),limitations:vec!["These are within-model information-set controls, not separately trained model versions.".into()],metrics};
    capability.validate()?;
    if let Some(parent) = c.output.parent() {
        fs::create_dir_all(parent)?;
    }
    burn_gekko_data::write_json(
        &c.output,
        &json!({"schema":1,"checkpoint_sha256":c.checkpoint_sha256,"capability":capability}),
    )?;
    burn_gekko_data::write_json(
        &c.output.with_extension("provenance.json"),
        &json!({"config":c,"shared_targets":common,"input_signature":signature}),
    )?;
    Ok(())
}
