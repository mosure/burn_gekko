//! Actual viewpoint changes with renderer labels confined to CPU scoring.
use crate::{latent_assess::load_assessed_model, latent_pilot::WeightAncestor};
use anyhow::{Result, ensure};
use burn::tensor::{Tensor, backend::Backend};
use burn_gekko_data::{
    Split, load_dataset_rgb, read_dataset_manifest,
    view_targets::{Cache, labels},
    write_config, write_json,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewGeometryAuditConfig {
    pub dataset: PathBuf,
    pub cache: PathBuf,
    pub weights: WeightAncestor,
    pub rooms: usize,
    pub temperature: f64,
}
pub fn run<B: Backend>(c: &ViewGeometryAuditConfig, out: &Path, device: &B::Device) -> Result<()> {
    ensure!(
        !out.exists() && (1..=128).contains(&c.rooms) && (0.01..=1.).contains(&c.temperature),
        "invalid view audit"
    );
    ensure!(
        fs::canonicalize(out.ancestors().find(|p| p.exists()).unwrap())?
            .starts_with(fs::canonicalize(".data")?),
        "audit outside .data"
    );
    let cache = Cache::load(&c.cache, &c.dataset)?;
    let loaded = load_assessed_model::<B>(&c.weights, device)?;
    let model = loaded.model;
    let manifest = read_dataset_manifest(&c.dataset)?;
    let selected = manifest
        .scenes
        .iter()
        .filter(|s| s.split == Split::Validation)
        .take(c.rooms)
        .collect::<Vec<_>>();
    ensure!(
        selected.len() == c.rooms && cache.manifest.patch == 16,
        "invalid validation cohort"
    );
    let grid = cache.manifest.grid;
    fs::create_dir_all(out)?;
    write_config(&out.join("config.toml"), c)?;
    let mut records = Vec::new();
    let mut predictions = Vec::new();
    let mut summary: BTreeMap<String, Vec<(f64, f64, f64, f64)>> = BTreeMap::new();
    for (i, entry) in selected.into_iter().enumerate() {
        let scene = load_dataset_rgb(&c.dataset, &manifest, entry)?;
        let room = cache.room_index(entry.seed, Split::Validation)?;
        let encoded = model.encode_references(&[
            crate::encoder::image_tensor(&scene, 0, device),
            crate::encoder::image_tensor(&scene, 1, device),
        ]);
        let descriptor = |a: usize, b: usize| -> Result<Tensor<B, 3>> {
            model
                .spatial_descriptor(
                    encoded[a].clone(),
                    model
                        .fusion
                        .decoder
                        .pair_training(encoded[a].clone(), encoded[b].clone(), grid)?
                        .features,
                )
                .ok_or_else(|| anyhow::anyhow!("missing spatial descriptor"))
        };
        let width = model.encoder_config.encoder.embed_dim;
        let base = |i: usize| {
            let x = encoded[i].clone().slice_dim(2, width..2 * width);
            x.clone() - x.mean_dim(1)
        };
        let candidates = [
            ("spatial_pair", descriptor(0, 1)?, descriptor(1, 0)?),
            ("spatial_self", descriptor(0, 0)?, descriptor(1, 1)?),
            ("spatial_encoder", base(0), base(1)),
        ];
        let truth = [
            labels(&cache.pair(room, 0, 1)?, grid, 16)?.0,
            labels(&cache.pair(room, 1, 0)?, grid, 16)?.0,
        ];
        for (name, a, b) in candidates {
            for (direction, (a, b)) in [(a.clone(), b.clone()), (b, a)].into_iter().enumerate() {
                let readouts = crate::evaluation::refinement::descriptor_readouts(
                    name,
                    a,
                    b,
                    grid,
                    c.temperature,
                )?;
                for readout in readouts {
                    // Preserve every RGB-derived query before visibility-based EPE scoring.
                    // The separate CPU camera solver must not receive oracle-selected matches.
                    if direction == 0
                        && let Some(coordinates) = &readout.coordinates
                    {
                        predictions.push(burn_gekko_eval::pose::synthetic::Prediction {
                            room_seed: entry.seed,
                            method: readout.method.clone(),
                            indices: readout.indices.clone(),
                            mutual: readout.mutual.clone(),
                            coordinates: coordinates.clone(),
                        });
                    }
                    let score = if let Some(coordinates) = &readout.coordinates {
                        burn_gekko_eval::warp::score_coordinates(
                            coordinates,
                            &truth[direction],
                            grid,
                            16,
                        )?
                    } else {
                        burn_gekko_eval::warp::score(&readout.indices, &truth[direction], grid, 16)?
                    };
                    summary.entry(readout.method.clone()).or_default().push((
                        score.mean_epe,
                        score.pck_half_patch,
                        score.pck_one_patch,
                        score.valid_queries as f64,
                    ));
                    records.push(serde_json::json!({"room_seed":scene.seed,"sample":i,"method":readout.method,"direction":direction,"score":score}));
                }
            }
        }
    }
    let means:BTreeMap<_,_>=summary.into_iter().map(|(name,rows)| {
        let mean=|f:fn(&(f64,f64,f64,f64))->f64| rows.iter().map(f).sum::<f64>()/rows.len() as f64;
        (name,serde_json::json!({"mean_epe":mean(|r|r.0),"pck8":mean(|r|r.1),"pck16":mean(|r|r.2),"valid_queries":rows.iter().map(|r|r.3 as usize).sum::<usize>()}))
    }).collect();
    write_json(
        &out.join("pose-predictions.json"),
        &burn_gekko_eval::pose::synthetic::Export {
            schema: 1,
            checkpoint_sha256: c.weights.model_sha256.clone(),
            dataset_id: manifest.dataset_id.clone(),
            dataset_manifest_sha256: burn_gekko_data::sha256_file(
                &c.dataset.join("manifest.json"),
            )?,
            source_sha256: crate::provenance::identity()?,
            grid,
            rooms: c.rooms,
            target: 0,
            reference: 1,
            predictions,
        },
    )?;
    write_json(
        &out.join("metrics.json"),
        &serde_json::json!({
            "schema":1,"task":"renderer_viewpoint_correspondence","status":"development_diagnostic","checkpoint":c.weights,
            "dataset_id":manifest.dataset_id,"rooms":c.rooms,"width":manifest.config.width,"height":manifest.config.height,
            "aggregation":"equal room and direction weight; view 0 versus view 1; PCK thresholds in model-input pixels",
            "cache_manifest_sha256":burn_gekko_data::sha256_file(&c.cache.join("manifest.json"))?,
            "policy":burn_gekko_data::view_targets::POLICY,"methods":means,"records":records,
            "scope":"RGB-only model inputs; renderer-supervised validation labels are not self-supervision or external transfer evidence",
            "noncommercial_weight_dependencies":[]
        }),
    )?;
    Ok(())
}
