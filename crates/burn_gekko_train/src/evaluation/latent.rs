//! Read-only latent evaluation. Geometry is loaded only after predictions.
use crate::{
    e2e_pilot::host_batch,
    encoder::normalize,
    latent::{LatentModel, normalize_teacher, relative_gain, token_cosine, token_mse},
    masking::mask,
};
use anyhow::Result;
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use burn_gekko_data::{RgbScene, SceneEntry, Visibility, load_geometry, write_json};
use burn_vjepa::VJepaEncoder;
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use std::{fs, path::Path};

/// Permute post-encoder spatial tokens while leaving decoder positions fixed.
/// Encoder features still contain positional context: this is a sensitivity
/// control, not proof that all geometry was removed.
pub fn shuffled_tokens<B: Backend>(tokens: Tensor<B, 3>, seed: u64) -> Tensor<B, 3> {
    let n = tokens.dims()[1];
    let mut ids: Vec<i64> = (0..n as i64).collect();
    ids.shuffle(&mut ChaCha8Rng::seed_from_u64(seed));
    let indices = Tensor::<B, 1, Int>::from_data(TensorData::new(ids, [n]), &tokens.device());
    tokens.select(1, indices)
}

pub use burn_gekko::tensor::values;

fn save(path: &Path, values: &[f32]) -> Result<()> {
    fs::write(
        path,
        values
            .iter()
            .flat_map(|x| x.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    Ok(())
}
fn mean(v: &[f32], hidden: &[usize]) -> f64 {
    hidden.iter().map(|&i| v[i] as f64).sum::<f64>() / hidden.len() as f64
}
fn spatial_variance(v: &[f32], indices: &[usize], d: usize) -> f64 {
    (0..d)
        .map(|k| {
            let avg =
                indices.iter().map(|&i| v[i * d + k] as f64).sum::<f64>() / indices.len() as f64;
            indices
                .iter()
                .map(|&i| (v[i * d + k] as f64 - avg).powi(2))
                .sum::<f64>()
                / indices.len() as f64
        })
        .sum::<f64>()
        / d as f64
}

#[allow(clippy::too_many_arguments)]
pub fn evaluate<B: Backend>(
    model: &LatentModel<B>,
    teacher: &VJepaEncoder<B>,
    scenes: &[RgbScene],
    entries: &[SceneEntry],
    config: &crate::latent_pilot::LatentConfig,
    train_mean: Tensor<B, 3>,
    path: &Path,
    export: bool,
    geometry: bool,
) -> Result<serde_json::Value> {
    fs::create_dir_all(path)?;
    let device = train_mean.device();
    let grid = [scenes[0].height / 16, scenes[0].width / 16];
    let n = grid[0] * grid[1];
    let d = model.encoder_config.encoder.embed_dim;
    let mask = mask(
        grid,
        config.eval_mask_ratio.unwrap_or(config.mask_ratio),
        config.seed ^ 0x4c4154454e54,
        0,
        config.eval_mask,
    )?;
    let hidden: Vec<_> = (0..n).filter(|i| !mask.indices().contains(i)).collect();
    let mut rows = Vec::new();
    let mut gain_scores = Vec::new();
    let mut ri_scores = Vec::new();
    let mut audit = serde_json::json!({});
    for (s, scene) in scenes.iter().enumerate() {
        for v in 0..scene.views.len() {
            let (rgb, refs) = host_batch::<B>(scenes, &[(s, v)], config.references, &device);
            let raw_teacher = teacher
                .forward_image(normalize(rgb.clone(), &model.encoder_config), None)
                .tokens;
            let target = normalize_teacher(raw_teacher);
            let sparse = model.encode(rgb.clone(), Some(&mask));
            let ref_features = model.encode_references(&refs);
            let p = model.predict_encoded(sparse.clone(), ref_features.clone(), &mask, grid)?;
            let ri = model.predict_improvement(
                model.encode(rgb.clone(), None),
                ref_features.clone(),
                grid,
            )?;
            let shuffled = model.predict_encoded(
                sparse,
                ref_features
                    .into_iter()
                    .enumerate()
                    .map(|(j, x)| shuffled_tokens(x, config.seed ^ 0x53485546464c45 ^ j as u64))
                    .collect(),
                &mask,
                grid,
            )?;
            let shuffled_error = values(token_mse(shuffled.cross, target.clone()))?;
            let cross_error = token_mse(p.cross.clone(), target.clone());
            let mono_error = token_mse(p.monocular.clone(), target.clone());
            let gain = values(relative_gain(cross_error.clone(), mono_error.clone()))?;
            let cross = values(cross_error)?;
            let mono = values(mono_error)?;
            let cosine = values(token_cosine(p.cross.clone(), target.clone()))?;
            let mono_cosine = values(token_cosine(p.monocular.clone(), target.clone()))?;
            let constant = values(token_mse(train_mean.clone(), target.clone()))?;
            let ri = values(ri)?;
            let pv = values(p.cross.clone())?;
            let tv = values(target.clone())?;
            let mv = if export && s < config.export_rooms {
                Some(values(p.monocular.clone())?)
            } else {
                None
            };
            let unrelated = if scenes.len() > 1 {
                let (_, other) = host_batch::<B>(
                    scenes,
                    &[((s + 1) % scenes.len(), v)],
                    config.references,
                    &device,
                );
                let u = model.predict(rgb.clone(), &other, &mask)?;
                let err = values(token_mse(u.cross, target))?;
                Some(mean(&err, &hidden))
            } else {
                None
            };
            if s == 0 && v == 0 {
                let visible =
                    crate::matching::visibility(scene.height, scene.width, &mask, &device);
                let changed_rgb = rgb.clone() * visible.clone() + (visible.neg() + 1.) * 0.913;
                let changed = model.predict(changed_rgb, &refs, &mask)?;
                let mut reverse = refs.clone();
                reverse.reverse();
                let reversed = model.predict(rgb.clone(), &reverse, &mask)?;
                let diff = p.cross.clone() - reversed.cross;
                audit = serde_json::json!({
                    "hidden_rgb_max_abs_delta": values((p.cross.clone()-changed.cross).abs().max())?[0],
                    "reference_permutation_max_abs_delta": values(diff.clone().abs().max())?[0],
                    "reference_permutation_rms_delta": values(diff.powf_scalar(2.).mean().sqrt())?[0],
                    "monocular_reference_permutation_max_abs_delta": values((p.monocular-reversed.monocular).abs().max())?[0],
                    "reference_permutation_strict_tolerance": 1e-5,
                });
            }
            // Evaluation geometry never enters teacher, student, loss, or optimizer.
            let mut labels = vec![255u8; n];
            let mut fractions = vec![-1f32; n];
            if geometry {
                let geom = load_geometry(&config.dataset.join("raw").join(&entries[s].file))?;
                for &i in &hidden {
                    let mut known = 0;
                    let mut positives = 0;
                    for py in 0..16 {
                        for px in 0..16 {
                            let pixel =
                                (i / grid[1] * 16 + py) * scene.width + i % grid[1] * 16 + px;
                            let labels: Vec<_> = (1..=config.references)
                                .map(|offset| {
                                    burn_gekko_data::visibility(
                                        &geom,
                                        v,
                                        (v + offset) % scene.views.len(),
                                        pixel,
                                    )
                                })
                                .collect();
                            if labels.contains(&Visibility::Visible) {
                                known += 1;
                                positives += 1;
                            } else if !labels.contains(&Visibility::Unknown) {
                                known += 1;
                            }
                        }
                    }
                    if known >= 128 {
                        fractions[i] = positives as f32 / known as f32;
                        let visible = positives * 2 >= known;
                        labels[i] = u8::from(visible);
                        gain_scores.push((gain[i], visible));
                        ri_scores.push((ri[i], visible));
                    }
                }
            }
            let target_variance = spatial_variance(&tv, &hidden, d);
            let predicted_variance = spatial_variance(&pv, &hidden, d);
            rows.push(serde_json::json!({"room_seed":scene.seed,"target_view":v,
                "cross_mse":mean(&cross,&hidden),"monocular_mse":mean(&mono,&hidden),
                "cross_cosine":mean(&cosine,&hidden),"monocular_cosine":mean(&mono_cosine,&hidden),
                "train_position_mean_mse":mean(&constant,&hidden),"unrelated_mse":unrelated,
                "spatially_shuffled_mse":mean(&shuffled_error,&hidden),
                "teacher_spatial_variance":target_variance,"prediction_spatial_variance":predicted_variance,
                "spatial_variance_ratio":predicted_variance/target_variance.max(1e-12)}));
            if export && s < config.export_rooms {
                let out = path.join(format!("room-{}-view-{v}", scene.seed));
                fs::create_dir(&out)?;
                save(&out.join("target-latent.f32"), &tv)?;
                save(&out.join("cross-latent.f32"), &pv)?;
                save(&out.join("monocular-latent.f32"), &mv.unwrap())?;
                save(&out.join("target-rgb.f32"), &scene.views[v])?;
                for j in 1..=config.references {
                    save(
                        &out.join(format!("reference-{j}-rgb.f32")),
                        &scene.views[(v + j) % scene.views.len()],
                    )?;
                }
                save(&out.join("gain.f32"), &gain)?;
                save(&out.join("ri.f32"), &ri)?;
                save(&out.join("visibility-fraction.f32"), &fractions)?;
                fs::write(out.join("visibility.u8"), &labels)?;
                write_json(
                    &out.join("metadata.json"),
                    &serde_json::json!({"room_seed":scene.seed,
                    "target_view":v,"latent_shape":[n,d],"grid":grid,"rgb_shape":[scene.height,scene.width,3],
                    "hidden_tokens":hidden,"visible_tokens":mask.indices(),"float_encoding":"little_endian_f32",
                    "label_255":"visible target tokens or unknown evaluation geometry",
                    "geometry_diagnostic_only":true}),
                )?;
            }
        }
    }
    let mean_key = |key: &str| -> Option<f64> {
        let v: Vec<_> = rows.iter().filter_map(|r| r[key].as_f64()).collect();
        (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
    };
    let result = serde_json::json!({"schema":1,"task":"fixed_vjepa21_latent_prediction",
        "split":entries[0].split,"diagnostic_only":true,"target_views":rows.len(),"rows":rows,
        "mean_cross_mse":mean_key("cross_mse"),"mean_monocular_mse":mean_key("monocular_mse"),
        "mean_cross_cosine":mean_key("cross_cosine"),"mean_monocular_cosine":mean_key("monocular_cosine"),
        "mean_train_position_mean_mse":mean_key("train_position_mean_mse"),"mean_unrelated_mse":mean_key("unrelated_mse"),
        "mean_spatially_shuffled_mse":mean_key("spatially_shuffled_mse"),
        "mask_pattern":config.eval_mask,"visible_tokens":mask.indices(),"hidden_tokens":hidden,
        "mean_spatial_variance_ratio":mean_key("spatial_variance_ratio"),"input_audit":audit,
        "latent_gain_covisibility":crate::eval::ranking_metrics(&gain_scores)?,
        "learned_ri_covisibility":crate::eval::ranking_metrics(&ri_scores)?,
        "geometry_protocol":"hidden patches only; at least 128 known pixels; majority visible in any reference; ties counted visible; never training supervision"});
    write_json(&path.join("metrics.json"), &result)?;
    Ok(result)
}
