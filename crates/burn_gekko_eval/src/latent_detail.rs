//! CPU-only decomposition of existing completion exports; no fitting or inference.
use crate::schema::{Capability, CapabilityStatus, Metric};
use crate::{metrics::detail, statistics::bootstrap_mean};
use anyhow::{Context, Result, ensure};
use burn_gekko_data::{sha256_file, write_json};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DetailConfig {
    pub directory: PathBuf,
    pub metrics_sha256: String,
    pub provenance_sha256: String,
    pub checkpoint_sha256: String,
    pub output: PathBuf,
}

#[derive(Debug, Serialize)]
struct Row {
    room_seed: u64,
    target_view: u64,
    cross: detail::DetailMetrics,
    monocular: detail::DetailMetrics,
}

fn record(path: &Path, sources: &mut BTreeMap<PathBuf, String>) -> Result<()> {
    sources.insert(path.to_path_buf(), sha256_file(path)?);
    Ok(())
}
fn floats(path: &Path, sources: &mut BTreeMap<PathBuf, String>) -> Result<Vec<f32>> {
    record(path, sources)?;
    let bytes = fs::read(path)?;
    ensure!(bytes.len().is_multiple_of(4), "malformed float export");
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}

/// Report per-target values and room-clustered paired error reductions.
pub fn analyze(c: &DetailConfig) -> Result<Value> {
    ensure!(!c.output.exists(), "preserve existing detail analysis");
    ensure!(
        !c.output.with_extension("head.json").exists(),
        "preserve existing detail capability"
    );
    ensure!(
        fs::canonicalize(
            c.output
                .ancestors()
                .skip(1)
                .find(|p| p.exists())
                .context("output parent")?
        )?
        .starts_with(fs::canonicalize(".data")?),
        "detail output outside .data"
    );
    let mut sources = BTreeMap::new();
    for (name, expected) in [
        ("metrics.json", &c.metrics_sha256),
        ("provenance.json", &c.provenance_sha256),
    ] {
        let p = c.directory.join(name);
        record(&p, &mut sources)?;
        ensure!(sources[&p] == *expected, "detail input checksum mismatch");
    }
    let metrics: Value = serde_json::from_slice(&fs::read(c.directory.join("metrics.json"))?)?;
    let provenance: Value =
        serde_json::from_slice(&fs::read(c.directory.join("provenance.json"))?)?;
    ensure!(
        provenance["checkpoint"]["model_sha256"] == c.checkpoint_sha256,
        "detail checkpoint mismatch"
    );
    ensure!(
        metrics["task"] == "fixed_vjepa21_latent_prediction",
        "unsupported detail task"
    );
    let exported = metrics["rows"].as_array().context("completion rows")?;
    ensure!(
        !exported.is_empty() && metrics["target_views"] == exported.len(),
        "incomplete completion rows"
    );
    let mut seen = BTreeSet::new();
    let mut rows = Vec::new();
    for row in exported {
        let seed = row["room_seed"].as_u64().context("room seed")?;
        let view = row["target_view"].as_u64().context("target view")?;
        ensure!(seen.insert((seed, view)), "duplicate completion target");
        let dir = c.directory.join(format!("room-{seed}-view-{view}"));
        record(&dir.join("metadata.json"), &mut sources)?;
        let meta: Value = serde_json::from_slice(&fs::read(dir.join("metadata.json"))?)?;
        ensure!(
            meta["room_seed"] == seed
                && meta["target_view"] == view
                && meta["float_encoding"] == "little_endian_f32",
            "completion metadata mismatch"
        );
        let grid: [usize; 2] = serde_json::from_value(meta["grid"].clone())?;
        let shape: [usize; 2] = serde_json::from_value(meta["latent_shape"].clone())?;
        ensure!(
            grid[0].checked_mul(grid[1]) == Some(shape[0]),
            "grid/latent shape mismatch"
        );
        let hidden: Vec<usize> = serde_json::from_value(meta["hidden_tokens"].clone())?;
        let truth = floats(&dir.join("target-latent.f32"), &mut sources)?;
        let cross = detail::measure(
            &floats(&dir.join("cross-latent.f32"), &mut sources)?,
            &truth,
            grid,
            shape[1],
            &hidden,
        )?;
        let monocular = detail::measure(
            &floats(&dir.join("monocular-latent.f32"), &mut sources)?,
            &truth,
            grid,
            shape[1],
            &hidden,
        )?;
        for (key, actual) in [
            ("cross_mse", cross.mse),
            ("monocular_mse", monocular.mse),
            (
                "prediction_spatial_variance",
                cross.prediction_spatial_power,
            ),
            ("teacher_spatial_variance", cross.teacher_spatial_power),
        ] {
            ensure!(
                (row[key].as_f64().context("exported metric")? - actual).abs() < 2e-6,
                "detail arrays disagree with {key}"
            );
        }
        rows.push(Row {
            room_seed: seed,
            target_view: view,
            cross,
            monocular,
        });
    }
    let mut methods = BTreeMap::new();
    for (method, cross) in [("cross", true), ("monocular", false)] {
        let observations = rows
            .iter()
            .map(|r| serde_json::to_value(if cross { &r.cross } else { &r.monocular }))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut measures = BTreeMap::new();
        for key in observations[0]
            .as_object()
            .unwrap()
            .keys()
            .filter(|k| !matches!(k.as_str(), "tokens" | "channels" | "adjacent_pairs"))
        {
            let values = observations
                .iter()
                .filter_map(|o| o[key].as_f64())
                .collect::<Vec<_>>();
            measures.insert(key.clone(), json!({"mean":(!values.is_empty()).then(||values.iter().sum::<f64>() / values.len() as f64), "defined_views":values.len()}));
        }
        let mean_mse = measures["mse"]["mean"].as_f64().unwrap();
        let key = if cross {
            "mean_cross_mse"
        } else {
            "mean_monocular_mse"
        };
        ensure!(
            (metrics[key].as_f64().context("population MSE")? - mean_mse).abs() < 2e-6,
            "detail population MSE differs"
        );
        methods.insert(method, measures);
    }
    let mut room_gains: BTreeMap<u64, Vec<[f64; 3]>> = BTreeMap::new();
    for r in &rows {
        room_gains.entry(r.room_seed).or_default().push([
            r.monocular.mse - r.cross.mse,
            r.monocular.mean_bias_mse - r.cross.mean_bias_mse,
            r.monocular.centered_mse - r.cross.centered_mse,
        ]);
    }
    let mut gains = BTreeMap::new();
    for (i, name) in ["mse", "mean_bias_mse", "centered_mse"]
        .into_iter()
        .enumerate()
    {
        let clusters = room_gains
            .values()
            .map(|v| v.iter().map(|x| x[i]).sum::<f64>() / v.len() as f64)
            .collect::<Vec<_>>();
        gains.insert(name, bootstrap_mean(&clusters, 0x44455441494c + i as u64)?);
    }
    let report = json!({
        "schema":1,"checkpoint_sha256":c.checkpoint_sha256,"status":"posthoc_development_diagnostic",
        "protocol":"Each channel is centered across hidden positions separately in prediction and teacher. Total MSE equals channel-mean bias MSE plus centered MSE. Methods share target identities and hidden masks. Adjacent differences use horizontal/vertical neighbors only when both endpoints are hidden. Means weight target views equally; uncertainty resamples room means.",
        "limitations":["The per-view oracle gain and variance-matching calculations use teacher values; they are diagnostics, not legal inference transforms or improved model scores.","A high variance ratio is not evidence of correct spatial structure. Centered correlation and adjacent-difference error expose wrong detail.","This analysis reuses existing exports as development data. It does not create a fresh test or change the published evaluation's historical role.","Feature power and MSE are normalized latent units, never RGB PSNR. Zero-power correlations and gains are undefined; defined-view counts remain explicit."],
        "target_views":rows.len(),"rooms":room_gains.len(),"methods":methods,"paired_reference_gains":gains,"rows":rows,"sources":sources,"config":c
    });
    if let Some(parent) = c.output.parent() {
        fs::create_dir_all(parent)?;
    }
    write_json(&c.output, &report)?;
    let mut measures = Vec::new();
    for (id, label, unit, lower) in [
        (
            "mean_bias_mse",
            "Channel-mean bias error",
            "squared normalized feature units",
            true,
        ),
        (
            "centered_mse",
            "Spatial structure error after channel centering",
            "squared normalized feature units",
            true,
        ),
        (
            "spatial_correlation",
            "Centered spatial agreement",
            "cosine",
            false,
        ),
        (
            "adjacent_power_ratio",
            "Neighbor-difference power retained",
            "fraction",
            false,
        ),
        (
            "adjacent_correlation",
            "Neighbor-difference agreement",
            "cosine",
            false,
        ),
        (
            "oracle_gain_mse",
            "Teacher-assisted amplitude oracle MSE (diagnostic)",
            "squared normalized feature units",
            true,
        ),
        (
            "variance_matching_mse",
            "Teacher-variance rescaling MSE (diagnostic)",
            "squared normalized feature units",
            true,
        ),
    ] {
        let measured = &report["methods"]["cross"][id];
        if let Some(value) = measured["mean"].as_f64() {
            measures.push(Metric {id:id.into(),label:label.into(),value,unit:unit.into(),lower_is_better:lower,samples:measured["defined_views"].as_u64().context("defined detail views")? as usize,aggregation:"equal target-view means over hidden tokens; undefined zero-power values excluded with explicit counts".into()});
        }
    }
    let capability = Capability {
        id: "completion_detail".into(),
        label: "Completion detail diagnostic".into(),
        status: CapabilityStatus::Evaluated,
        protocol: format!(
            "Post-hoc CPU analysis on {} development targets from {} rooms, bound to this checkpoint. {}",
            report["target_views"],
            report["rooms"],
            report["protocol"].as_str().unwrap()
        ),
        limitations: report["limitations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect(),
        metrics: measures,
    };
    capability.validate()?;
    write_json(
        &c.output.with_extension("head.json"),
        &json!({"schema":1,"checkpoint_sha256":c.checkpoint_sha256,"capability":capability}),
    )?;
    Ok(report)
}
