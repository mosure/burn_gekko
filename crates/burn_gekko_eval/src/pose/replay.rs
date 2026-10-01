//! Verify that exporter engineering preserves a sealed correspondence population.
use super::benchmark::Prediction;
use anyhow::{Context, Result, ensure};
use burn_gekko_data::{sha256_file, write_json};
use serde::Deserialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, path::PathBuf};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Export {
    pub predictions: PathBuf,
    pub predictions_sha256: String,
    pub provenance: PathBuf,
    pub provenance_sha256: String,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayConfig {
    pub original: Export,
    pub replay: Export,
    pub coordinate_tolerance: f64,
    pub output: PathBuf,
}
type Population = BTreeMap<(String, String), Prediction>;
pub(crate) fn load(c: &Export) -> Result<(Population, Value)> {
    ensure!(
        sha256_file(&c.predictions)? == c.predictions_sha256
            && sha256_file(&c.provenance)? == c.provenance_sha256,
        "pose replay source changed"
    );
    let provenance: Value = serde_json::from_slice(&fs::read(&c.provenance)?)?;
    ensure!(
        provenance["predictions_sha256"] == c.predictions_sha256,
        "pose replay provenance mismatch"
    );
    let mut rows = Population::new();
    for line in fs::read_to_string(&c.predictions)?.lines() {
        let p: Prediction = serde_json::from_str(line)?;
        let [h, w] = p.grid;
        ensure!(
            (1..=32).contains(&h) && (1..=32).contains(&w),
            "invalid pose replay grid"
        );
        let n = h * w;
        ensure!(
            h > 0
                && w > 0
                && n <= 1024
                && p.indices.len() == n
                && p.mutual.len() == n
                && p.coordinates.len() == n
                && p.indices.iter().all(|i| *i < n),
            "invalid pose replay shape"
        );
        ensure!(
            p.coordinates.iter().all(|v| v[0].is_finite()
                && v[1].is_finite()
                && (0. ..=(w - 1) as f64).contains(&v[0])
                && (0. ..=(h - 1) as f64).contains(&v[1])),
            "invalid pose replay coordinates"
        );
        ensure!(
            rows.insert((p.pair.clone(), p.method.clone()), p).is_none(),
            "duplicate pose replay identity"
        );
    }
    let pairs = provenance["pairs"]
        .as_u64()
        .context("pose replay pair count")? as usize;
    let methods = provenance["methods"]
        .as_array()
        .context("pose replay methods")?;
    ensure!(
        !rows.is_empty() && rows.len() == pairs * methods.len(),
        "incomplete pose replay population"
    );
    Ok((rows, provenance))
}
fn compare(a: &Population, b: &Population, tolerance: f64) -> Result<Value> {
    ensure!(
        tolerance.is_finite() && tolerance >= 0.,
        "invalid coordinate tolerance"
    );
    ensure!(
        !a.is_empty() && a.keys().eq(b.keys()),
        "pose replay populations differ"
    );
    let mut coordinates: f64 = 0.;
    let mut index_changes = 0;
    let mut mutual_changes = 0;
    for (key, a) in a {
        let b = &b[key];
        ensure!(
            a.grid == b.grid
                && a.indices.len() == b.indices.len()
                && a.coordinates.len() == b.coordinates.len(),
            "pose replay grid changed"
        );
        index_changes += a
            .indices
            .iter()
            .zip(&b.indices)
            .filter(|(a, b)| a != b)
            .count();
        mutual_changes += a
            .mutual
            .iter()
            .zip(&b.mutual)
            .filter(|(a, b)| a != b)
            .count();
        for (a, b) in a.coordinates.iter().zip(&b.coordinates) {
            for (a, b) in a.iter().zip(b) {
                coordinates = coordinates.max((a - b).abs());
            }
        }
    }
    Ok(
        json!({"schema":1,"passed":index_changes == 0 && mutual_changes == 0 && coordinates <= tolerance,"prediction_rows":a.len(),"hard_index_changes":index_changes,"mutual_flag_changes":mutual_changes,"maximum_coordinate_difference_grid_cells":coordinates,"coordinate_tolerance_grid_cells":tolerance,"scope":"Identical checkpoint, RGB manifest, pair/readout population and grid. Hard predictions must be exact; fractional locations use the declared absolute tolerance. No timing or generalization claim."}),
    )
}
pub fn verify(c: &ReplayConfig) -> Result<Value> {
    ensure!(!c.output.exists(), "preserve existing pose replay receipt");
    let (a, pa) = load(&c.original)?;
    let (b, pb) = load(&c.replay)?;
    for key in [
        "checkpoint_sha256",
        "images_sha256",
        "selection_sha256",
        "evaluation_use",
        "methods",
        "pairs",
        "unique_images",
    ] {
        ensure!(
            !pa[key].is_null() && pa[key] == pb[key],
            "pose replay identity differs: {key}"
        );
    }
    let mut result = compare(&a, &b, c.coordinate_tolerance)?;
    result["sources"] = json!({c.original.predictions.display().to_string():c.original.predictions_sha256,c.original.provenance.display().to_string():c.original.provenance_sha256,c.replay.predictions.display().to_string():c.replay.predictions_sha256,c.replay.provenance.display().to_string():c.replay.provenance_sha256});
    write_json(&c.output, &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fractional_tolerance_never_excuses_hard_match_or_population_changes() {
        let p = Prediction {
            pair: "pair".into(),
            method: "method".into(),
            grid: [2, 2],
            indices: vec![0; 4],
            mutual: vec![true; 4],
            coordinates: vec![[0., 0.]; 4],
        };
        let a = BTreeMap::from([((p.pair.clone(), p.method.clone()), p)]);
        let mut b = a.clone();
        b.values_mut().next().unwrap().coordinates[0][0] = 1e-6;
        assert_eq!(compare(&a, &b, 1e-5).unwrap()["passed"], true);
        b.values_mut().next().unwrap().indices[0] = 1;
        assert_eq!(compare(&a, &b, 1e-5).unwrap()["passed"], false);
        b.clear();
        assert!(compare(&a, &b, 1e-5).is_err());
    }
}
