//! Recheck signal-normalized completion metrics against every exported target.
use crate::{
    artifact::{Source, number, record},
    experiment::Experiment,
    figures::floats,
};
use anyhow::{Context, Result, ensure};
use serde_json::Value;
use std::{collections::BTreeSet, fs};

pub fn verify(e: &Experiment, metrics: &Value, sources: &mut Vec<Source>) -> Result<Value> {
    let rows = metrics["rows"].as_array().context("completion rows")?;
    let mut identities = BTreeSet::new();
    let mut strata = [(0usize, 0usize, 0f64); 2]; // tokens, target views, summed token MSE
    for row in rows {
        let seed = row["room_seed"].as_u64().context("room seed")?;
        let view = row["target_view"].as_u64().context("target view")?;
        ensure!(
            identities.insert((seed, view)),
            "duplicate completion identity"
        );
        let dir = e.latent.directory.join(format!("room-{seed}-view-{view}"));
        record(&dir.join("metadata.json"), sources)?;
        let meta: Value = serde_json::from_slice(&fs::read(dir.join("metadata.json"))?)?;
        ensure!(
            meta["room_seed"] == seed && meta["target_view"] == view,
            "completion array identity mismatch"
        );
        let channels = meta["latent_shape"][1]
            .as_u64()
            .context("latent channels")? as usize;
        let hidden = meta["hidden_tokens"]
            .as_array()
            .context("hidden mask")?
            .iter()
            .map(|x| x.as_u64().map(|x| x as usize).context("hidden token"))
            .collect::<Result<Vec<_>>>()?;
        let truth = floats(&dir.join("target-latent.f32"), sources)?;
        let prediction = floats(&dir.join("cross-latent.f32"), sources)?;
        ensure!(
            meta["latent_shape"][0]
                .as_u64()
                .is_some_and(|n| n as usize * channels == truth.len()),
            "completion array shape mismatch"
        );
        let check = burn_gekko_eval::metrics::completion(&prediction, &truth, channels, &hidden)?;
        for (key, value, tolerance) in [
            ("cross_mse", check.mse, 2e-6),
            (
                "cross_cosine",
                check.cosine.context("undefined feature cosine")?,
                2e-6,
            ),
            ("teacher_signal_power", check.signal_power, 1e-9),
            (
                "feature_snr_db",
                check.signal_to_error_db.context("undefined feature SNR")?,
                2e-4,
            ),
            (
                "spatial_variance_ratio",
                check
                    .spatial_variance_ratio
                    .context("undefined feature variance")?,
                2e-6,
            ),
        ] {
            ensure!(
                (value - number(row, key)?).abs() < tolerance,
                "exported arrays disagree with {key} at room {seed} view {view}"
            );
        }
        record(&dir.join("visibility.u8"), sources)?;
        let visibility = fs::read(dir.join("visibility.u8"))?;
        ensure!(
            visibility.len() == truth.len() / channels
                && visibility.iter().all(|x| matches!(*x, 0 | 1 | 255)),
            "invalid completion visibility labels"
        );
        for (label, group) in strata.iter_mut().enumerate() {
            let selected = hidden
                .iter()
                .copied()
                .filter(|&i| visibility[i] == label as u8)
                .collect::<Vec<_>>();
            if selected.is_empty() {
                continue;
            }
            let measured =
                burn_gekko_eval::metrics::completion(&prediction, &truth, channels, &selected)?;
            group.0 += selected.len();
            group.1 += 1;
            group.2 += measured.mse * selected.len() as f64;
        }
    }
    for key in ["teacher_signal_power", "feature_snr_db"] {
        let mean = rows
            .iter()
            .map(|r| number(r, key))
            .collect::<Result<Vec<_>>>()?
            .iter()
            .sum::<f64>()
            / rows.len() as f64;
        ensure!(
            (mean - number(metrics, &format!("mean_{key}"))?).abs() < 1e-9,
            "completion summary differs from rows: {key}"
        );
    }
    Ok(
        serde_json::json!({"status":"posthoc_descriptive", "aggregation":"hidden-token weighted MSE; unknown labels excluded; binary majority-visibility labels do not imply every pixel in a patch has the same visibility", "groups":strata.iter().enumerate().map(|(label,(tokens,views,error))| serde_json::json!({"visible":label==1,"hidden_tokens":tokens,"target_views":views,"mse":(*tokens>0).then(||error / *tokens as f64)})).collect::<Vec<_>>()}),
    )
}
