//! Bounded single-checkpoint known-transform diagnostic. This is a development
//! check of geometric learning, not a substitute for external viewpoint tests.
use crate::{
    correspondence::{conditional_matches, self_conditioned_descriptor},
    data::augmentation::{self, EquivarianceConfig},
    latent_assess::load_assessed_model,
    latent_eval::values,
    latent_pilot::WeightAncestor,
};
use anyhow::{Result, ensure};
use burn::tensor::{Tensor, backend::Backend};
use burn_gekko_data::{Split, load_dataset_rgb, read_dataset_manifest, write_config, write_json};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EquivarianceAuditConfig {
    pub dataset: PathBuf,
    pub weights: WeightAncestor,
    pub rooms: usize,
    pub seed: u64,
    pub augmentation: EquivarianceConfig,
}

pub fn run<B: Backend>(c: &EquivarianceAuditConfig, out: &Path, device: &B::Device) -> Result<()> {
    ensure!(
        !out.exists() && (1..=128).contains(&c.rooms),
        "invalid audit output/size"
    );
    ensure!(
        fs::canonicalize(out.ancestors().find(|p| p.exists()).unwrap())?
            .starts_with(fs::canonicalize(".data")?),
        "audit outside .data"
    );
    c.augmentation.validate()?;
    let loaded = load_assessed_model::<B>(&c.weights, device)?;
    let model = loaded.model;
    ensure!(
        model.fusion.spatial_descriptor.is_some(),
        "audit requires a spatial descriptor"
    );
    let manifest = read_dataset_manifest(&c.dataset)?;
    let selected = manifest
        .scenes
        .iter()
        .filter(|s| s.split == Split::Validation)
        .take(c.rooms)
        .cloned()
        .collect::<Vec<_>>();
    ensure!(selected.len() == c.rooms, "insufficient validation scenes");
    let scenes = selected
        .iter()
        .map(|entry| load_dataset_rgb(&c.dataset, &manifest, entry))
        .collect::<Result<Vec<_>>>()?;
    fs::create_dir_all(out)?;
    write_config(&out.join("config.toml"), c)?;
    let mut records = Vec::new();
    let mut samples = Vec::new();
    let mut summary: BTreeMap<String, Vec<(f64, f64, f64, f64)>> = BTreeMap::new();
    for (i, scene) in scenes.iter().enumerate() {
        let grid = [scene.height / 16, scene.width / 16];
        let warped = augmentation::batch::<B>(
            std::slice::from_ref(scene),
            &[(0, 0)],
            &c.augmentation,
            c.seed,
            i,
            16,
            device,
        )?;
        let rgb = crate::encoder::image_tensor(scene, 0, device);
        let encoded = model.encode_references(&[rgb, warped.rgb.clone()]);
        let descriptor = |a: usize, b: usize| -> Result<Tensor<B, 3>> {
            model
                .spatial_descriptor(
                    encoded[a].clone(),
                    model.fusion.decoder.pair_features(
                        encoded[a].clone(),
                        encoded[b].clone(),
                        grid,
                    )?,
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
            (
                "spatial_self",
                self_conditioned_descriptor(&model, encoded[0].clone(), grid)?,
                self_conditioned_descriptor(&model, encoded[1].clone(), grid)?,
            ),
            ("spatial_encoder", base(0), base(1)),
        ];
        let labels = [
            values(warped.forward.clone())?,
            values(warped.backward.clone())?,
        ];
        for (name, a, b) in candidates {
            let nll =
                crate::train::scalar(burn_gekko::objectives::correspondence::bidirectional_nll(
                    a.clone(),
                    b.clone(),
                    warped.forward.clone(),
                    warped.backward.clone(),
                    c.augmentation.temperature,
                ))?;
            for (direction, (a, b)) in [(a.clone(), b.clone()), (b, a)].into_iter().enumerate() {
                let (indices, _) = conditional_matches(a, b, c.augmentation.temperature)?;
                let score = burn_gekko_eval::warp::score(&indices, &labels[direction], grid, 16)?;
                summary.entry(name.into()).or_default().push((
                    score.mean_epe,
                    score.pck_half_patch,
                    score.pck_one_patch,
                    nll,
                ));
                records.push(serde_json::json!({"room_seed":scene.seed,"sample":i,"method":name,"direction":direction,"nll_bidirectional":nll,"score":score}));
            }
        }
        if i < 4 {
            fs::write(
                out.join(format!("sample-{i}-source.f32")),
                scene.views[0]
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            fs::write(
                out.join(format!("sample-{i}-warped.f32")),
                values(warped.rgb.permute([0, 2, 3, 1]))?
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            )?;
            let source = out.join(format!("sample-{i}-source.f32"));
            let warped = out.join(format!("sample-{i}-warped.f32"));
            samples.push(serde_json::json!({"sample":i,"room_seed":scene.seed,
                "source":{"file":source,"sha256":burn_gekko_data::sha256_file(&source)?},
                "warped":{"file":warped,"sha256":burn_gekko_data::sha256_file(&warped)?}}));
        }
    }
    let means: BTreeMap<_,_> = summary.into_iter().map(|(name,rows)| {
        let average = |f: fn(&(f64,f64,f64,f64))->f64| rows.iter().map(f).sum::<f64>()/rows.len() as f64;
        (name,serde_json::json!({"mean_epe":average(|x| x.0),"pck8":average(|x| x.1),"pck16":average(|x| x.2),"nll":average(|x| x.3)}))
    }).collect();
    write_json(
        &out.join("metrics.json"),
        &serde_json::json!({"status":"development_diagnostic","checkpoint":c.weights,"dataset_id":manifest.dataset_id,"rooms":c.rooms,"width":scenes[0].width,"height":scenes[0].height,"aggregation":"equal room and direction weight; PCK thresholds in model-input pixels","methods":means,"records":records,"samples":samples,"noncommercial_weight_dependencies":[]}),
    )?;
    Ok(())
}
