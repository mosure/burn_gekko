//! Common-mask completion and spatial-detail evidence, independent of training probes.
use crate::{
    adaptation::{Input, number, pinned},
    statistics::bootstrap_mean,
};
use anyhow::{Context, Result, ensure};
use burn_gekko_data::{read_config, sha256_file};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputSet {
    pub detail: Input,
}

#[derive(Debug, Clone, Serialize)]
pub struct Measurements {
    pub cross_mse: f64,
    pub monocular_mse: f64,
    pub centered_mse: f64,
    pub spatial_correlation: f64,
    pub adjacent_correlation: f64,
    pub adjacent_power_ratio: f64,
}

type Population = BTreeMap<(u64, u64), Value>;
pub struct Evidence {
    pub mean: Measurements,
    pub population: Population,
    pub assessment_config: Value,
    pub rows: BTreeMap<(u64, u64), Measurements>,
}

fn closure(v: &Value, sources: &mut BTreeMap<PathBuf, String>) -> Result<()> {
    for (path, hash) in v["sources"]
        .as_object()
        .context("completion source closure")?
    {
        let path = PathBuf::from(path);
        let hash = hash.as_str().context("completion source checksum")?;
        ensure!(sha256_file(&path)? == hash, "completion input changed");
        sources.insert(path, hash.into());
    }
    Ok(())
}

pub fn load(
    input: &InputSet,
    checkpoint: &str,
    rooms: usize,
    sources: &mut BTreeMap<PathBuf, String>,
) -> Result<Evidence> {
    let detail = pinned(&input.detail, sources)?;
    ensure!(
        detail["checkpoint_sha256"] == checkpoint && detail["rooms"] == rooms,
        "completion checkpoint or cohort differs"
    );
    closure(&detail, sources)?;
    let dir = PathBuf::from(
        detail["config"]["directory"]
            .as_str()
            .context("completion directory")?,
    );
    let read = |name: &str| -> Result<Value> {
        let path = dir.join(name);
        ensure!(
            sources.contains_key(&path),
            "incomplete completion source closure"
        );
        Ok(serde_json::from_slice(&fs::read(path)?)?)
    };
    let metrics = read("metrics.json")?;
    let provenance = read("provenance.json")?;
    ensure!(
        metrics["task"] == "fixed_vjepa21_latent_prediction"
            && provenance["checkpoint"]["model_sha256"] == checkpoint,
        "not common-checkpoint latent completion"
    );
    let config_path = dir
        .parent()
        .context("assessment parent")?
        .join("config.toml");
    let config_hash = sha256_file(&config_path)?;
    ensure!(
        provenance["assessment_config_sha256"] == config_hash,
        "assessment recipe changed"
    );
    sources.insert(config_path.clone(), config_hash);
    let config: Value = read_config(&config_path)?;
    ensure!(
        config["rooms"] == rooms
            && config["export_rooms"] == rooms
            && config["split"] == "validation"
            && config["mask_pattern"] == "random"
            && config["mask_ratio"] == 0.9
            && config["seed"] == 853
            && config["references"] == 2
            && config["stable_attention"] == false,
        "completion protocol differs from registration"
    );
    let name = dir
        .file_name()
        .context("assessment model name")?
        .to_str()
        .context("model name encoding")?;
    let model = config["models"]
        .as_array()
        .context("assessment models")?
        .iter()
        .filter(|v| v["name"] == name)
        .collect::<Vec<_>>();
    ensure!(
        model.len() == 1
            && model[0]["model_sha256"] == checkpoint
            && model[0]["checkpoint"] == provenance["checkpoint"]["checkpoint"],
        "assessment model identity differs"
    );
    let mut rows = BTreeMap::new();
    let mut population = Population::new();
    for row in detail["rows"]
        .as_array()
        .context("completion detail rows")?
    {
        let seed = row["room_seed"].as_u64().context("completion seed")?;
        let view = row["target_view"]
            .as_u64()
            .context("completion target view")?;
        ensure!(view < 3, "unexpected completion view");
        let path = dir.join(format!("room-{seed}-view-{view}"));
        let meta_path = path.join("metadata.json");
        for file in [
            "metadata.json",
            "target-latent.f32",
            "cross-latent.f32",
            "monocular-latent.f32",
        ] {
            ensure!(
                sources.contains_key(&path.join(file)),
                "incomplete completion array closure"
            );
        }
        let meta: Value = serde_json::from_slice(&fs::read(meta_path)?)?;
        ensure!(
            meta["room_seed"] == seed && meta["target_view"] == view,
            "completion identity mismatch"
        );
        let identity = json!({"grid":meta["grid"],"latent_shape":meta["latent_shape"],
            "hidden_tokens":meta["hidden_tokens"],"truth_sha256":sources[&path.join("target-latent.f32")]});
        ensure!(
            population.insert((seed, view), identity).is_none(),
            "duplicate completion target"
        );
        let cross = &row["cross"];
        let value = Measurements {
            cross_mse: number(&cross["mse"])?,
            monocular_mse: number(&row["monocular"]["mse"])?,
            centered_mse: number(&cross["centered_mse"])?,
            spatial_correlation: number(&cross["spatial_correlation"])?,
            adjacent_correlation: number(&cross["adjacent_correlation"])?,
            adjacent_power_ratio: number(&cross["adjacent_power_ratio"])?,
        };
        ensure!(
            value.cross_mse >= 0.
                && value.monocular_mse >= 0.
                && value.centered_mse >= 0.
                && (-1. ..=1.).contains(&value.spatial_correlation)
                && (-1. ..=1.).contains(&value.adjacent_correlation),
            "invalid completion measures"
        );
        rows.insert((seed, view), value);
    }
    ensure!(
        rows.len() == rooms * 3 && detail["target_views"] == rows.len(),
        "incomplete completion cohort"
    );
    for &(seed, _) in rows.keys() {
        ensure!(
            (0..3).all(|v| rows.contains_key(&(seed, v))),
            "missing completion view"
        );
    }
    let mean = |f: fn(&Measurements) -> f64| rows.values().map(f).sum::<f64>() / rows.len() as f64;
    let mean = Measurements {
        cross_mse: mean(|v| v.cross_mse),
        monocular_mse: mean(|v| v.monocular_mse),
        centered_mse: mean(|v| v.centered_mse),
        spatial_correlation: mean(|v| v.spatial_correlation),
        adjacent_correlation: mean(|v| v.adjacent_correlation),
        adjacent_power_ratio: mean(|v| v.adjacent_power_ratio),
    };
    ensure!(
        (mean.cross_mse - number(&metrics["mean_cross_mse"])?).abs() < 2e-6
            && (mean.monocular_mse - number(&metrics["mean_monocular_mse"])?).abs() < 2e-6,
        "completion aggregate differs from exported arrays"
    );
    Ok(Evidence {
        mean,
        population,
        assessment_config: config,
        rows,
    })
}

pub fn compare(candidate: &Evidence, control: &Evidence) -> Result<Value> {
    // Each model/checkpoint binding is verified in load(). A bundle's roster is
    // not an assessment setting; allow pinned controls from an earlier bundle.
    let protocol = |value: &Value| -> Result<Value> {
        let mut settings = value.as_object().context("assessment settings")?.clone();
        settings.remove("models");
        Ok(Value::Object(settings))
    };
    ensure!(
        candidate.population == control.population
            && protocol(&candidate.assessment_config)? == protocol(&control.assessment_config)?,
        "completion targets, teacher arrays, masks or assessment recipe differ"
    );
    let mut rooms: BTreeMap<u64, Vec<[f64; 4]>> = BTreeMap::new();
    for (key, a) in &candidate.rows {
        let b = &control.rows[key];
        rooms.entry(key.0).or_default().push([
            b.cross_mse - a.cross_mse,
            b.centered_mse - a.centered_mse,
            a.spatial_correlation - b.spatial_correlation,
            a.adjacent_correlation - b.adjacent_correlation,
        ]);
    }
    let mut intervals = BTreeMap::new();
    for (i, name) in [
        "cross_mse_reduction",
        "centered_mse_reduction",
        "spatial_correlation_gain",
        "adjacent_correlation_gain",
    ]
    .iter()
    .enumerate()
    {
        let values = rooms
            .values()
            .map(|v| v.iter().map(|x| x[i]).sum::<f64>() / v.len() as f64)
            .collect::<Vec<_>>();
        intervals.insert(*name, bootstrap_mean(&values, 853 + i as u64)?);
    }
    Ok(
        json!({"rooms":rooms.len(),"paired_room_intervals":intervals,
        "scope":"95% paired room bootstrap; three views are clustered within each room. Reused synthetic development data."}),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture() -> Evidence {
        let measure = Measurements {
            cross_mse: 4.,
            monocular_mse: 5.,
            centered_mse: 2.,
            spatial_correlation: 0.4,
            adjacent_correlation: 0.3,
            adjacent_power_ratio: 0.2,
        };
        let rows = (1..=2)
            .flat_map(|room| (0..3).map(move |view| (room, view)))
            .map(|id| (id, measure.clone()))
            .collect::<BTreeMap<_, _>>();
        Evidence {
            population: rows
                .keys()
                .map(|&id| (id, json!({"hidden_tokens":[1,2],"truth_sha256":"teacher"})))
                .collect(),
            assessment_config: json!({"fixture":"common"}),
            mean: measure,
            rows,
        }
    }
    #[test]
    fn uncertainty_clusters_views_and_rejects_changed_teacher_or_mask() {
        let baseline = fixture();
        let mut candidate = fixture();
        for ((room, _), row) in &mut candidate.rows {
            row.cross_mse -= if *room == 1 { 1. } else { 3. };
        }
        let result = compare(&candidate, &baseline).unwrap();
        let interval = &result["paired_room_intervals"]["cross_mse_reduction"];
        assert_eq!(interval["clusters"], 2);
        assert_eq!(interval["mean"], 2.);
        assert_eq!(interval["low"], 1.);
        assert_eq!(interval["high"], 3.);
        candidate.population.get_mut(&(1, 0)).unwrap()["hidden_tokens"] = json!([0, 2]);
        assert!(compare(&candidate, &baseline).is_err());
        candidate.population = baseline.population.clone();
        candidate.population.get_mut(&(1, 0)).unwrap()["truth_sha256"] = json!("different-teacher");
        assert!(compare(&candidate, &baseline).is_err());
    }

    #[test]
    fn reusing_control_exports_changes_only_the_roster_not_assessment_settings() {
        let mut baseline = fixture();
        let mut candidate = fixture();
        baseline.assessment_config["models"] = json!(["previous roster"]);
        candidate.assessment_config["models"] = json!(["new candidate"]);
        assert!(compare(&candidate, &baseline).is_ok());
        candidate.assessment_config["seed"] = json!(999);
        assert!(compare(&candidate, &baseline).is_err());
    }
}
