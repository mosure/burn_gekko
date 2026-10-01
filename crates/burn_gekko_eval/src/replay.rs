//! Native loss-trajectory verification for bounded execution replays.
use anyhow::{Context, Result, ensure};
use burn_gekko_data::{sha256_file, write_json};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayConfig {
    pub original: PathBuf,
    pub original_sha256: String,
    pub replay: PathBuf,
    pub replay_sha256: String,
    pub absolute_tolerance: f64,
    pub relative_tolerance: f64,
    /// Explicit engineering prefix when the original completed run is longer.
    #[serde(default)]
    pub prefix_updates: Option<usize>,
    pub output: PathBuf,
}
fn read(p: &Path) -> Result<Vec<Value>> {
    fs::read_to_string(p)?
        .lines()
        .map(|s| Ok(serde_json::from_str(s)?))
        .collect()
}
pub fn compare(a: &[Value], b: &[Value], absolute: f64, relative: f64) -> Result<Value> {
    ensure!(
        absolute.is_finite() && relative.is_finite() && absolute >= 0. && relative >= 0.,
        "invalid replay tolerances"
    );
    ensure!(
        !a.is_empty() && b.len() >= a.len(),
        "replay must contain the original update prefix"
    );
    let mut deltas: BTreeMap<String, f64> = BTreeMap::new();
    let mut passed = true;
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        ensure!(
            a["step"].as_u64() == Some((i + 1) as u64)
                && a["step"] == b["step"]
                && a["samples"] == b["samples"]
                && a["stage"] == b["stage"]
                && a["encoder_gradient_tensors"] == b["encoder_gradient_tensors"],
            "replay update/input/gradient-stage mismatch"
        );
        for key in [
            "total",
            "cross",
            "monocular",
            "visible",
            "ri",
            "attention_kl",
            "dense_latent_mse",
            "descriptor_kl",
            "warp_pair_nll",
            "warp_self_nll",
            "gradient_norm",
            "learning_rate",
        ] {
            let x = a[key]
                .as_f64()
                .filter(|v| v.is_finite())
                .context("original finite scalar")?;
            let y = b[key]
                .as_f64()
                .filter(|v| v.is_finite())
                .context("replay finite scalar")?;
            let delta = (x - y).abs();
            passed &= delta <= absolute + relative * x.abs();
            let v = deltas.entry(key.into()).or_insert(0.);
            *v = v.max(delta);
        }
    }
    Ok(
        json!({"schema":1,"passed":passed,"overlapping_updates":a.len(),"replay_updates":b.len(),"absolute_tolerance":absolute,"relative_tolerance":relative,"maximum_absolute_differences":deltas,"scope":"Same absolute update prefix, sample order, stage and encoder gradient count. Timing excluded; numerical agreement does not prove speedup or accuracy generalization."}),
    )
}
pub fn verify(c: &ReplayConfig) -> Result<Value> {
    ensure!(!c.output.exists(), "preserve existing replay receipt");
    ensure!(
        sha256_file(&c.original)? == c.original_sha256
            && sha256_file(&c.replay)? == c.replay_sha256,
        "replay input checksum mismatch"
    );
    let original = read(&c.original)?;
    let replay = read(&c.replay)?;
    let prefix = c.prefix_updates.unwrap_or(original.len());
    ensure!(
        prefix > 0 && prefix <= original.len() && prefix <= replay.len(),
        "requested replay prefix is incomplete"
    );
    let mut out = compare(
        &original[..prefix],
        &replay,
        c.absolute_tolerance,
        c.relative_tolerance,
    )?;
    out["original_updates"] = json!(original.len());
    out["config"] = json!(c);
    write_json(&c.output, &out)?;
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn short_replay_requires_an_explicit_complete_prefix() {
        let root = tempfile::tempdir().unwrap();
        let row = |step| {
            json!({"step":step,"samples":[[7,0]],"stage":1,"encoder_gradient_tensors":28,"total":0.4,"cross":0.2,"monocular":0.3,"visible":0.1,"ri":0.1,"attention_kl":0.,"dense_latent_mse":0.,"descriptor_kl":0.,"warp_pair_nll":2.,"warp_self_nll":2.,"gradient_norm":0.1,"learning_rate":0.00001}).to_string()
        };
        let original = root.path().join("original.jsonl");
        let replay = root.path().join("replay.jsonl");
        fs::write(&original, (1..=4).map(row).collect::<Vec<_>>().join("\n")).unwrap();
        fs::write(&replay, (1..=2).map(row).collect::<Vec<_>>().join("\n")).unwrap();
        let mut config = ReplayConfig {
            original_sha256: sha256_file(&original).unwrap(),
            replay_sha256: sha256_file(&replay).unwrap(),
            original,
            replay,
            absolute_tolerance: 0.,
            relative_tolerance: 0.,
            prefix_updates: None,
            output: root.path().join("receipt.json"),
        };
        assert!(verify(&config).is_err());
        config.prefix_updates = Some(3);
        assert!(verify(&config).is_err());
        config.prefix_updates = Some(2);
        let result = verify(&config).unwrap();
        assert_eq!(result["original_updates"], 4);
        assert_eq!(result["overlapping_updates"], 2);
        assert_eq!(result["passed"], true);
    }
    #[test]
    fn changed_samples_fail_and_scalar_drift_is_reported() {
        let mut row = json!({"step":1,"samples":[[7,0]],"stage":2,"encoder_gradient_tensors":151});
        for k in [
            "total",
            "cross",
            "monocular",
            "visible",
            "ri",
            "attention_kl",
            "dense_latent_mse",
            "descriptor_kl",
            "warp_pair_nll",
            "warp_self_nll",
            "gradient_norm",
            "learning_rate",
        ] {
            row[k] = json!(0.2);
        }
        let mut changed = row.clone();
        changed["cross"] = json!(0.3);
        assert_eq!(
            compare(&[row.clone()], &[changed.clone()], 1e-6, 1e-5).unwrap()["passed"],
            false
        );
        changed["samples"] = json!([[8, 0]]);
        assert!(compare(&[row], &[changed], 1e-6, 1e-5).is_err());
    }
}
