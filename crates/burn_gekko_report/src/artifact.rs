use crate::experiment::{Experiment, PinnedFile};
use anyhow::{Context, Result, ensure};
use burn_gekko_eval::{
    schema::{Capability, CapabilityStatus, Metric},
    statistics::bootstrap_mean,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::Path};

#[derive(Debug, Clone, Serialize)]
pub struct Source {
    pub path: String,
    pub sha256: String,
    pub bytes: u64,
}
#[derive(Debug, Serialize)]
pub struct Report {
    pub checkpoint_sha256: String,
    pub capabilities: Vec<Capability>,
    pub latent: Value,
    pub benchmarks: Vec<Value>,
    pub training: Value,
    pub efficiency: Option<burn_gekko_eval::efficiency::Efficiency>,
    pub sources: Vec<Source>,
}
pub fn record(path: &Path, sources: &mut Vec<Source>) -> Result<String> {
    let sha = burn_gekko_data::sha256_file(path)?;
    let name = path
        .strip_prefix(std::env::current_dir()?)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned();
    if !sources.iter().any(|s| s.path == name) {
        sources.push(Source {
            path: name,
            sha256: sha.clone(),
            bytes: path.metadata()?.len(),
        });
    }
    Ok(sha)
}
fn pinned(file: &PinnedFile, sources: &mut Vec<Source>) -> Result<Value> {
    ensure!(
        record(&file.path, sources)? == file.sha256,
        "artifact checksum mismatch: {}",
        file.path.display()
    );
    Ok(serde_json::from_slice(&fs::read(&file.path)?)?)
}
pub fn number(v: &Value, key: &str) -> Result<f64> {
    v[key]
        .as_f64()
        .filter(|x| x.is_finite())
        .with_context(|| format!("missing finite metric {key}"))
}
fn metric(
    id: &str,
    label: &str,
    value: f64,
    unit: &str,
    lower: bool,
    samples: usize,
    aggregation: &str,
) -> Metric {
    Metric {
        id: id.into(),
        label: label.into(),
        value,
        unit: unit.into(),
        lower_is_better: lower,
        samples,
        aggregation: aggregation.into(),
    }
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct HeadEvidence {
    schema: u32,
    checkpoint_sha256: String,
    capability: Capability,
}
pub fn load(e: &Experiment) -> Result<Report> {
    ensure!(
        e.schema == 1 && !e.id.is_empty() && (1..=12).contains(&e.sample_count),
        "invalid experiment schema/id/sample count"
    );
    ensure!(
        matches!(e.latent.evaluation_use.as_str(), "held_out" | "development"),
        "explicit evaluation use required"
    );
    ensure!(
        fs::canonicalize(e.checkpoint.parent().context("checkpoint has no parent")?)?
            == fs::canonicalize(&e.run)?,
        "checkpoint belongs to another run"
    );
    let mut sources = Vec::new();
    ensure!(
        record(&e.checkpoint.join("model.mpk"), &mut sources)? == e.checkpoint_sha256,
        "checkpoint checksum mismatch"
    );
    let metadata: Value = serde_json::from_slice(&fs::read(e.checkpoint.join("metadata.json"))?)?;
    ensure!(
        metadata["model_sha256"] == e.checkpoint_sha256,
        "checkpoint metadata mismatch"
    );
    record(&e.checkpoint.join("metadata.json"), &mut sources)?;
    let provenance: Value = serde_json::from_slice(&fs::read(e.run.join("provenance.json"))?)?;
    ensure!(
        provenance["noncommercial_weight_dependencies"]
            .as_array()
            .is_some_and(Vec::is_empty),
        "missing or noncommercial training provenance"
    );
    record(&e.run.join("provenance.json"), &mut sources)?;
    record(&e.run.join("config.toml"), &mut sources)?;
    let training_report: Value = serde_json::from_slice(&fs::read(e.run.join("report.json"))?)?;
    record(&e.run.join("report.json"), &mut sources)?;
    let selected_step = metadata["completed_steps"]
        .as_u64()
        .context("checkpoint step")?;
    let starting_step = training_report["starting_step"].as_u64().unwrap_or(0);
    ensure!(
        selected_step >= starting_step,
        "checkpoint predates this run"
    );
    let config: Value = burn_gekko_data::read_config(&e.run.join("config.toml"))?;
    let training_log = e.run.join("metrics.jsonl");
    record(&training_log, &mut sources)?;
    let lines = |p: &Path| -> Result<Vec<Value>> {
        fs::read_to_string(p)?
            .lines()
            .map(|line| Ok(serde_json::from_str(line)?))
            .collect()
    };
    let steps = lines(&training_log)?;
    let coverage = burn_gekko_eval::training::coverage(
        &steps,
        starting_step,
        selected_step,
        config["batch_size"]
            .as_u64()
            .context("training batch size")? as usize,
        config["train_rooms"]
            .as_u64()
            .context("training room count")? as usize,
    )?;
    let latent = pinned(
        &PinnedFile {
            path: e.latent.directory.join("metrics.json"),
            sha256: e.latent.metrics_sha256.clone(),
        },
        &mut sources,
    )?;
    let proof = pinned(
        &PinnedFile {
            path: e.latent.directory.join("provenance.json"),
            sha256: e.latent.provenance_sha256.clone(),
        },
        &mut sources,
    )?;
    ensure!(
        proof["checkpoint"]["model_sha256"] == e.checkpoint_sha256,
        "latent assessment belongs to another checkpoint"
    );
    ensure!(
        latent["task"] == "fixed_vjepa21_latent_prediction",
        "wrong completion task"
    );
    let rows = latent["rows"]
        .as_array()
        .context("missing per-view observations")?;
    let count = latent["target_views"]
        .as_u64()
        .context("missing target count")? as usize;
    ensure!(
        rows.len() == count && count > 0,
        "completion observation count mismatch"
    );
    for key in [
        "cross_mse",
        "cross_cosine",
        "monocular_mse",
        "spatial_variance_ratio",
    ] {
        let mean = rows
            .iter()
            .map(|r| number(r, key))
            .collect::<Result<Vec<_>>>()?
            .iter()
            .sum::<f64>()
            / count as f64;
        ensure!(
            (mean - number(&latent, &format!("mean_{key}"))?).abs() < 1e-9,
            "latent summary differs from rows: {key}"
        );
    }
    let mut rooms = BTreeMap::<u64, Vec<f64>>::new();
    for r in rows {
        rooms
            .entry(r["room_seed"].as_u64().context("missing room")?)
            .or_default()
            .push(number(r, "cross_mse")?);
    }
    let ci = bootstrap_mean(
        &rooms
            .values()
            .map(|r| r.iter().sum::<f64>() / r.len() as f64)
            .collect::<Vec<_>>(),
        719,
    )?;
    let mut latent = latent;
    latent["room_bootstrap_mse"] = json!(ci);
    let mut gains = BTreeMap::<u64, Vec<f64>>::new();
    for row in latent["rows"].as_array().context("completion rows")? {
        gains
            .entry(row["room_seed"].as_u64().context("room seed")?)
            .or_default()
            .push(number(row, "monocular_mse")? - number(row, "cross_mse")?);
    }
    latent["room_bootstrap_monocular_gain"] = json!(bootstrap_mean(
        &gains
            .values()
            .map(|v| v.iter().sum::<f64>() / v.len() as f64)
            .collect::<Vec<_>>(),
        719,
    )?);
    let mut capabilities = vec![Capability {
        id: "latent_completion".into(),
        label: "Cross-view latent completion".into(),
        status: CapabilityStatus::Evaluated,
        protocol: format!(
            "{}; hidden tokens only, fixed V-JEPA 2.1 teacher, equal target-view weighting",
            e.latent.evaluation_use
        ),
        limitations: vec![
            "Feature colors visualize latent space; they are not RGB reconstructions.".into(),
        ],
        metrics: vec![
            metric(
                "hidden_mse",
                "Hidden-token MSE",
                number(&latent, "mean_cross_mse")?,
                "squared normalized feature units",
                true,
                count,
                "mean over hidden tokens, channels, then target views",
            ),
            metric(
                "hidden_cosine",
                "Hidden-token cosine",
                number(&latent, "mean_cross_cosine")?,
                "cosine",
                false,
                count,
                "mean over hidden tokens then target views",
            ),
            metric(
                "spatial_variance",
                "Spatial variance / teacher",
                number(&latent, "mean_spatial_variance_ratio")?,
                "ratio",
                false,
                count,
                "mean target-view variance ratio; 1 is teacher variance, larger is not always better",
            ),
        ],
    }];
    let mut controls = Vec::new();
    for (key, label) in [
        ("mean_monocular_mse", "References disabled"),
        (
            "mean_spatially_shuffled_mse",
            "Reference positions shuffled",
        ),
        ("mean_unrelated_mse", "Unrelated reference room"),
        (
            "mean_train_position_mean_mse",
            "Training-set mean at each position",
        ),
    ] {
        if latent.get(key).is_some() {
            controls.push(metric(
                key,
                label,
                number(&latent, key)?,
                "MSE",
                true,
                count,
                "same masked targets; equal target-view weighting",
            ));
        }
    }
    capabilities.push(Capability {
        id: "input_controls".into(),
        label: "Information-use controls".into(),
        status: CapabilityStatus::Evaluated,
        protocol: "Same checkpoint and masked targets as completion; only available reference information changes. Compare these MSE values with cross-view completion above.".into(),
        limitations: vec!["Controls measure information use within this experiment; they are not separately trained model versions.".into()],
        metrics: controls,
    });
    let ri = &latent["learned_ri_covisibility"];
    let n = ri["pixels"].as_u64().context("RI count")? as usize;
    capabilities.push(Capability {
        id: "covisibility".into(),
        label: "Co-visibility".into(),
        status: CapabilityStatus::Evaluated,
        protocol: latent["geometry_protocol"]
            .as_str()
            .context("geometry protocol")?
            .into(),
        limitations: vec![
            "RI uses a separate full-target branch; it is not sparse completion.".into(),
            format!(
                "Positive prevalence {:.1}%; AP depends on this prevalence.",
                100. * number(ri, "positives")? / n as f64
            ),
        ],
        metrics: vec![
            metric(
                "ri_auroc",
                "RI AUROC",
                number(ri, "auroc")?,
                "AUROC",
                false,
                n,
                "pooled known hidden patches; exact score ties grouped",
            ),
            metric(
                "ri_ap",
                "RI average precision",
                number(ri, "average_precision")?,
                "AP",
                false,
                n,
                "pooled known hidden patches",
            ),
        ],
    });
    let mut benchmarks = Vec::new();
    for input in &e.benchmarks {
        let b = pinned(input, &mut sources)?;
        ensure!(
            b["checkpoint_sha256"] == e.checkpoint_sha256,
            "benchmark belongs to another checkpoint"
        );
        let name = b["benchmark"].as_str().context("benchmark name")?;
        let methods = b["methods"].as_object().context("benchmark methods")?;
        let mut metrics = Vec::new();
        let mut limitations = vec![b["comparability"].as_str().context("comparability")?.into()];
        for (method, scores) in methods {
            for (key, label, unit, lower) in [
                ("aepe", "AEPE", "pixels", true),
                ("pck3", "PCK @ 3 px", "fraction", false),
            ] {
                metrics.push(metric(
                    &format!("{method}.{key}"),
                    &format!("{} · {label}", crate::display::readout(method)),
                    number(&scores["primary"], key)?,
                    unit,
                    lower,
                    scores["primary"]["pairs"]
                        .as_u64()
                        .or_else(|| b["pairs"].as_u64())
                        .context("primary pair count")? as usize,
                    b["protocol"].as_str().context("benchmark protocol")?,
                ));
            }
            if let Some(interval) = scores.get("aepe_cluster_interval") {
                limitations.push(format!("{}: AEPE 95% cluster interval [{:.3}, {:.3}] pixels over {} {} clusters; 10,000 deterministic bootstrap replicates.", crate::display::readout(method), number(interval, "low")?, number(interval, "high")?, interval["clusters"], scores["uncertainty_unit"].as_str().unwrap_or("declared")));
            }
        }
        capabilities.push(Capability {
            id: name.into(),
            label: format!(
                "{} correspondence",
                if name == "eth3d" {
                    "ETH3D"
                } else if name == "hpatches" {
                    "HPatches"
                } else {
                    name
                }
            ),
            status: CapabilityStatus::Evaluated,
            protocol: format!(
                "{}; {}",
                b["evaluation_use"].as_str().unwrap_or("unknown"),
                b["protocol"].as_str().unwrap_or("unknown")
            ),
            limitations,
            metrics,
        });
        if let Some(contrasts) = b["contrasts"].as_array() {
            for (i, contrast) in contrasts.iter().enumerate() {
                let candidate = crate::display::readout(
                    contrast["candidate"]
                        .as_str()
                        .context("contrast candidate")?,
                );
                let control = crate::display::readout(
                    contrast["control"].as_str().context("contrast control")?,
                );
                let count = contrast["pairs"].as_u64().context("contrast pairs")? as usize;
                let gain = &contrast["aepe_gain"];
                let pck = &contrast["pck3_gain"];
                let passed = number(gain, "low")? > 0. && number(pck, "mean")? >= 0.;
                capabilities.push(Capability {
                    id: format!("{name}_contrast_{i}"),
                    label: format!("{name}: paired readout gate — {}", if passed { "PASS" } else { "FAIL" }),
                    status: CapabilityStatus::Evaluated,
                    protocol: format!(
                        "{candidate} relative to {control}; {}",
                        contrast["interpretation"].as_str().context("contrast protocol")?
                    ),
                    metrics: vec![
                        metric("aepe_gain", "AEPE reduction (positive favors candidate)",
                            number(gain, "mean")?, "pixels", false, count, "paired scene/sequence means"),
                        metric("pck3_gain", "PCK3 gain", 100. * number(pck, "mean")?,
                            "percentage points", false, count, "paired scene/sequence means"),
                    ],
                    limitations: vec![
                        format!("Paired 95% AEPE-gain interval [{:.4}, {:.4}] pixels; PCK3-gain interval [{:.3}, {:.3}] percentage points. {} clusters, 10,000 deterministic draws.",
                            number(gain, "low")?, number(gain, "high")?,
                            100. * number(pck, "low")?, 100. * number(pck, "high")?, gain["clusters"]),
                        format!("Readout gate {}: AEPE-gain interval must be entirely positive and mean PCK3 gain nonnegative. PCK3 uncertainty is reported separately; this is not an equivalence test or evidence of general SOTA performance.",
                            if passed { "passes" } else { "fails" }),
                    ],
                });
            }
        }
        benchmarks.push(b);
    }
    if let Some(file) = &e.equivariance {
        let value = pinned(file, &mut sources)?;
        ensure!(
            value["checkpoint"]["model_sha256"] == e.checkpoint_sha256
                && value["status"] == "development_diagnostic",
            "warp audit belongs to another checkpoint or protocol"
        );
        let rooms = value["rooms"].as_u64().context("warp room count")? as usize;
        let mut metrics = Vec::new();
        for (method, scores) in value["methods"].as_object().context("warp methods")? {
            let rows = value["records"]
                .as_array()
                .context("warp records")?
                .iter()
                .filter(|r| r["method"] == *method)
                .collect::<Vec<_>>();
            ensure!(
                rooms > 0 && rows.len() == 2 * rooms,
                "warp observation count mismatch"
            );
            for (key, row_key, label, unit, lower) in [
                ("mean_epe", "mean_epe", "AEPE", "pixels", true),
                ("pck8", "pck_half_patch", "PCK8", "fraction", false),
                ("pck16", "pck_one_patch", "PCK16", "fraction", false),
                ("nll", "nll_bidirectional", "NLL", "nats", true),
            ] {
                let sum = rows
                    .iter()
                    .map(|r| number(if key == "nll" { r } else { &r["score"] }, row_key))
                    .collect::<Result<Vec<_>>>()?
                    .iter()
                    .sum::<f64>();
                ensure!(
                    (sum / rows.len() as f64 - number(scores, key)?).abs() < 1e-9,
                    "warp summary differs from rows"
                );
                metrics.push(metric(
                    &format!("{method}.{key}"),
                    &format!("{} / {label}", crate::display::readout(method)),
                    number(scores, key)?,
                    unit,
                    lower,
                    rooms,
                    "equal room and direction weight",
                ));
            }
        }
        capabilities.push(Capability {id:"image_transform".into(),label:"Known image transforms".into(),status:CapabilityStatus::Evaluated,
            protocol:"Development-only RGB homography diagnostic on validation rooms; the same checkpoint supplies pair-conditioned, self-conditioned and centered encoder descriptors.".into(),
            limitations:vec!["This tests geometric augmentation learning, not generalization to real 3D viewpoint changes. All distances use model-input pixels and a 16-pixel descriptor grid.".into()],metrics});
    }
    for file in &e.heads {
        let value = pinned(file, &mut sources)?;
        let head: HeadEvidence = serde_json::from_value(value)?;
        ensure!(
            head.schema == 1 && head.checkpoint_sha256 == e.checkpoint_sha256,
            "head evidence belongs to another checkpoint"
        );
        head.capability.validate()?;
        ensure!(
            !capabilities.iter().any(|c| c.id == head.capability.id),
            "duplicate capability"
        );
        capabilities.push(head.capability);
    }
    for (id, label, detail) in [
        (
            "camera",
            "Camera intrinsics and relative pose",
            "Camera metrics exist in burn_gekko_eval; this checkpoint has no trained camera head. Rotation/translation angular error, focal relative error and pose AUC must be measured before a capability claim.",
        ),
        (
            "rgb_completion",
            "RGB completion",
            "The selected latent model has no trained RGB reconstruction head. Latent visualizations do not establish sharp RGB reconstruction.",
        ),
        (
            "depth",
            "Depth / future decoders",
            "Register a checkpoint-bound head evaluation to populate this capability.",
        ),
    ] {
        if !capabilities.iter().any(|c| c.id == id) {
            capabilities.push(Capability {
                id: id.into(),
                label: label.into(),
                status: CapabilityStatus::NotTrained,
                protocol: detail.into(),
                limitations: Vec::new(),
                metrics: Vec::new(),
            });
        }
    }
    for c in &capabilities {
        c.validate()?;
    }
    let efficiency = if let Some(input) = &e.efficiency {
        let ledger = pinned(&input.ledger, &mut sources)?;
        let command = ledger["commands"]
            .as_array()
            .context("command ledger")?
            .iter()
            .find(|c| c["name"] == input.command)
            .context("missing telemetry command")?;
        ensure!(
            command["status"] == "complete",
            "training command did not complete"
        );
        let argv = command["argv"].as_array().context("training argv")?;
        let run = argv
            .windows(2)
            .find(|w| w[0] == "--run")
            .and_then(|w| w[1].as_str())
            .context("training command has no run")?;
        ensure!(
            fs::canonicalize(run)? == fs::canonicalize(&e.run)?,
            "telemetry belongs to a different training run"
        );
        ensure!(
            record(&input.telemetry.path, &mut sources)? == input.telemetry.sha256,
            "telemetry checksum mismatch"
        );
        let telemetry = lines(&input.telemetry.path)?;
        Some(burn_gekko_eval::efficiency::evaluate(
            &steps,
            &telemetry,
            number(command, "elapsed_seconds")?,
        )?)
    } else {
        None
    };
    let scalar_windows = burn_gekko_eval::training::scalar_windows(
        &steps,
        starting_step,
        selected_step,
        &[
            "cross",
            "monocular",
            "warp_pair_nll",
            "warp_self_nll",
            "augmentation_seconds",
        ],
        32,
    )?;
    let encoder_stages =
        burn_gekko_eval::training::encoder_stages(&steps, starting_step, selected_step)?;
    Ok(Report {
        checkpoint_sha256: e.checkpoint_sha256.clone(),
        capabilities,
        latent,
        benchmarks,
        efficiency,
        training: json!({"config":config,"coverage":coverage,"scalar_windows":scalar_windows,"encoder_stages":encoder_stages,"completed_steps":selected_step-starting_step,"checkpoint_step":selected_step,"starting_step":starting_step,"stage_steps":training_report["stage_steps"],"teacher_max_abs_delta":training_report["teacher_max_abs_delta"],"dataset_id":provenance["dataset_id"],"teacher_id":provenance["teacher_id"],"backend":provenance["backend"],"warm_start":provenance["warm_start"],"resume":provenance["resume"],"training_run":e.run}),
        sources,
    })
}
