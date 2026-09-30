//! Patch-grid correspondence diagnostics; ground-truth geometry is evaluation only.
use crate::{
    encoder::{image_tensor, normalize},
    latent::LatentModel,
    latent_eval::values,
};
use anyhow::{Result, ensure};
use burn::tensor::{Tensor, activation, backend::Backend};
use burn_gekko_data::{
    RgbScene, SceneEntry, Visibility, load_geometry, project, visibility, write_json,
};
use burn_vjepa::{VJepaConfig, VJepaEncoder};
use std::{collections::BTreeMap, path::Path};

/// Deterministic nearest-neighbour readout; ties select the lowest index.
pub fn nearest(scores: &[f32], n: usize) -> (Vec<usize>, Vec<bool>) {
    assert_eq!(scores.len(), n * n);
    // Start every reduction at its own row/column, not globally at zero.
    let argmax = |ids: Vec<usize>| {
        let first = ids[0];
        ids.into_iter()
            .fold(first, |a, b| if scores[b] > scores[a] { b } else { a })
    };
    let forward: Vec<_> = (0..n)
        .map(|i| argmax((i * n..(i + 1) * n).collect()) % n)
        .collect();
    let backward: Vec<_> = (0..n)
        .map(|j| argmax((0..n).map(|i| i * n + j).collect()) / n)
        .collect();
    let mutual = forward
        .iter()
        .enumerate()
        .map(|(i, &j)| backward[j] == i)
        .collect();
    (forward, mutual)
}
pub fn matches<B: Backend>(a: Tensor<B, 3>, b: Tensor<B, 3>) -> Result<(Vec<usize>, Vec<bool>)> {
    let n = a.dims()[1];
    nearest_checked(&values(cosine_scores(a, b))?, n)
}
pub(crate) fn cosine_scores<B: Backend>(a: Tensor<B, 3>, b: Tensor<B, 3>) -> Tensor<B, 3> {
    let unit = |x: Tensor<B, 3>| x.clone() / x.powf_scalar(2.).sum_dim(2).sqrt().clamp_min(1e-8);
    unit(a).matmul(unit(b).swap_dims(1, 2))
}
/// Apply the same reciprocal conditional readout as attention to descriptors.
/// Temperature is fixed before evaluation, never fitted to test geometry.
pub(crate) fn conditional_matches<B: Backend>(
    a: Tensor<B, 3>,
    b: Tensor<B, 3>,
    temperature: f64,
) -> Result<(Vec<usize>, Vec<bool>)> {
    let n = a.dims()[1];
    let scores = cosine_scores(a, b) / temperature;
    let reciprocal = reciprocal_conditionals(vec![scores.clone()], vec![scores.swap_dims(1, 2)]);
    nearest_checked(&values(reciprocal)?, n)
}

/// Shared trained trunk/head, with only the same image available as context.
/// This control can be computed once per image and reused across image pairs.
pub fn self_conditioned_descriptor<B: Backend>(
    model: &LatentModel<B>,
    encoded: Tensor<B, 3>,
    grid: [usize; 2],
) -> Result<Tensor<B, 3>> {
    let features = model
        .fusion
        .decoder
        .pair_features(encoded.clone(), encoded.clone(), grid)?;
    model
        .spatial_descriptor(encoded, features)
        .ok_or_else(|| anyhow::anyhow!("self-conditioned readout requires a spatial head"))
}

pub fn self_conditioned_readouts<B: Backend>(
    a: Tensor<B, 3>,
    b: Tensor<B, 3>,
) -> Result<Vec<crate::fusion_audit::NamedMatches>> {
    Ok(vec![
        ("spatial_self".into(), matches(a.clone(), b.clone())?),
        (
            "spatial_self_conditional".into(),
            conditional_matches(a, b, 0.07)?,
        ),
    ])
}

/// Compute only the three registered spatial readouts. Reuse the intermediate
/// level already carried in the encoder output; no redundant teacher forwards.
pub fn focused_spatial_readouts<B: Backend>(
    model: &LatentModel<B>,
    a: Tensor<B, 3>,
    b: Tensor<B, 3>,
    grid: [usize; 2],
    layer: usize,
) -> Result<Vec<crate::fusion_audit::NamedMatches>> {
    let width = model.encoder_config.encoder.embed_dim;
    ensure!(
        a.dims()[2] == 2 * width && b.dims()[2] == 2 * width,
        "missing spatial encoder route"
    );
    let spatial = |x: Tensor<B, 3>, context: Tensor<B, 3>| -> Result<Tensor<B, 3>> {
        model
            .spatial_descriptor(
                x.clone(),
                model.fusion.decoder.pair_features(x, context, grid)?,
            )
            .ok_or_else(|| anyhow::anyhow!("missing spatial head"))
    };
    let pair = conditional_matches(
        spatial(a.clone(), b.clone())?,
        spatial(b.clone(), a.clone())?,
        0.07,
    )?;
    let independent = conditional_matches(
        self_conditioned_descriptor(model, a.clone(), grid)?,
        self_conditioned_descriptor(model, b.clone(), grid)?,
        0.07,
    )?;
    let center = |x: Tensor<B, 3>| {
        let x = x.slice_dim(2, width..2 * width);
        x.clone() - x.mean_dim(1)
    };
    Ok(vec![
        ("spatial_residual_conditional".into(), pair),
        ("spatial_self_conditional".into(), independent),
        (
            format!("student_l{layer:02}_centered_conditional"),
            conditional_matches(center(a), center(b), 0.07)?,
        ),
    ])
}
fn conditional_centered_matches<B: Backend>(
    a: Tensor<B, 3>,
    b: Tensor<B, 3>,
    temperature: f64,
) -> Result<(Vec<usize>, Vec<bool>)> {
    conditional_matches(
        a.clone() - a.mean_dim(1),
        b.clone() - b.mean_dim(1),
        temperature,
    )
}
/// Remove each image's shared feature offset without fitting on other images.
pub fn centered_matches<B: Backend>(
    a: Tensor<B, 3>,
    b: Tensor<B, 3>,
) -> Result<(Vec<usize>, Vec<bool>)> {
    matches(a.clone() - a.mean_dim(1), b.clone() - b.mean_dim(1))
}
/// Match the student's dense view batching so baseline differences do not come
/// from a different encoder batch shape or matmul dispatch.
pub fn fixed_views<B: Backend>(
    teacher: &VJepaEncoder<B>,
    config: &VJepaConfig,
    rgb: &[Tensor<B, 4>],
) -> Vec<Tensor<B, 3>> {
    let b = rgb[0].dims()[0];
    let tokens = teacher
        .forward_image(normalize(Tensor::cat(rgb.to_vec(), 0), config), None)
        .tokens;
    (0..rgb.len())
        .map(|i| tokens.clone().slice_dim(0, i * b..(i + 1) * b))
        .collect()
}
pub(crate) fn nearest_checked(x: &[f32], n: usize) -> Result<(Vec<usize>, Vec<bool>)> {
    ensure!(
        x.len() == n * n && x.iter().all(|v| v.is_finite()),
        "invalid correspondence scores"
    );
    Ok(nearest(x, n))
}
pub fn attention_matches<B: Backend>(
    model: &LatentModel<B>,
    a: Tensor<B, 3>,
    b: Tensor<B, 3>,
    grid: [usize; 2],
) -> Result<(Vec<usize>, Vec<bool>)> {
    let forward = model
        .fusion
        .decoder
        .correspondence_attention(a.clone(), b.clone(), grid)?;
    let backward = model.fusion.decoder.correspondence_attention(b, a, grid)?;
    let score = (forward * backward.swap_dims(1, 2)).sqrt();
    nearest_checked(&values(score)?, grid[0] * grid[1])
}

/// Common, declared readouts for both development and independent benchmarks.
/// Normalized log probabilities match the auxiliary conditional objective;
/// raw logits and historical head-mean probabilities are controls.
/// All methods see the same RGB features.
pub fn standard_readouts<B: Backend>(
    model: &LatentModel<B>,
    a: Tensor<B, 3>,
    b: Tensor<B, 3>,
    teacher_a: Tensor<B, 3>,
    teacher_b: Tensor<B, 3>,
    grid: [usize; 2],
) -> Result<Vec<crate::fusion_audit::NamedMatches>> {
    let forward = model
        .fusion
        .decoder
        .pair_training(a.clone(), b.clone(), grid)?;
    let backward = model
        .fusion
        .decoder
        .pair_training(b.clone(), a.clone(), grid)?;
    let layer_mean = |x: Vec<Tensor<B, 3>>| Tensor::cat(x, 0).mean_dim(0);
    let conditional_score =
        reciprocal_conditionals(forward.logits.clone(), backward.logits.clone());
    let score = (layer_mean(forward.logits) + layer_mean(backward.logits).swap_dims(1, 2)) / 2.;
    let n = grid[0] * grid[1];
    let latent_a = model.fusion.prediction.forward(forward.features.clone());
    let latent_b = model.fusion.prediction.forward(backward.features.clone());
    let encoder_a = model.final_encoder_features(a.clone());
    let encoder_b = model.final_encoder_features(b.clone());
    // Match the registered auxiliary teacher temperature. Keep original raw
    // readouts alongside these controls; do not select a readout per example.
    let temperature = 0.07;
    let spatial = model
        .spatial_descriptor(a.clone(), forward.features.clone())
        .zip(model.spatial_descriptor(b.clone(), backward.features.clone()));
    let mut result = vec![
        (
            "fixed_teacher",
            matches(teacher_a.clone(), teacher_b.clone())?,
        ),
        (
            "centered_teacher",
            centered_matches(teacher_a.clone(), teacher_b.clone())?,
        ),
        (
            "conditional_teacher",
            conditional_matches(teacher_a.clone(), teacher_b.clone(), temperature)?,
        ),
        (
            "conditional_centered_teacher",
            conditional_centered_matches(teacher_a, teacher_b, temperature)?,
        ),
        (
            "student_encoder",
            matches(encoder_a.clone(), encoder_b.clone())?,
        ),
        (
            "centered_student",
            centered_matches(encoder_a.clone(), encoder_b.clone())?,
        ),
        (
            "conditional_student",
            conditional_matches(encoder_a.clone(), encoder_b.clone(), temperature)?,
        ),
        (
            "conditional_centered_student",
            conditional_centered_matches(encoder_a, encoder_b, temperature)?,
        ),
        (
            "fused_decoder",
            matches(forward.features.clone(), backward.features.clone())?,
        ),
        (
            "centered_decoder",
            centered_matches(forward.features.clone(), backward.features.clone())?,
        ),
        (
            "conditional_decoder",
            conditional_matches(
                forward.features.clone(),
                backward.features.clone(),
                temperature,
            )?,
        ),
        (
            "conditional_centered_decoder",
            conditional_centered_matches(
                forward.features.clone(),
                backward.features.clone(),
                temperature,
            )?,
        ),
        ("fused_latent", matches(latent_a.clone(), latent_b.clone())?),
        ("centered_latent", centered_matches(latent_a, latent_b)?),
        (
            "reciprocal_attention",
            attention_matches(model, a, b, grid)?,
        ),
        ("reciprocal_logits", nearest_checked(&values(score)?, n)?),
        (
            "reciprocal_log_probability",
            nearest_checked(&values(conditional_score)?, n)?,
        ),
        ("same_position", ((0..n).collect(), vec![true; n])),
    ];
    if let Some((a, b)) = spatial {
        result.push(("spatial_residual", matches(a.clone(), b.clone())?));
        result.push((
            "spatial_residual_conditional",
            conditional_matches(a, b, temperature)?,
        ));
    }
    Ok(result
        .into_iter()
        .map(|(name, prediction)| (name.to_owned(), prediction))
        .collect())
}
/// A row-softmax objective cannot identify additive row offsets. Normalize
/// before reciprocity so those offsets cannot become column preferences.
pub fn reciprocal_conditionals<B: Backend>(
    forward: Vec<Tensor<B, 3>>,
    backward: Vec<Tensor<B, 3>>,
) -> Tensor<B, 3> {
    assert!(!forward.is_empty() && forward.len() == backward.len());
    let mean = |x: Vec<Tensor<B, 3>>| {
        let count = x.len();
        x.into_iter()
            .map(|s| activation::log_softmax(s, 2))
            .reduce(|a, b| a + b)
            .unwrap()
            / count as f64
    };
    (mean(forward) + mean(backward).swap_dims(1, 2)) / 2.
}
fn center(i: usize, width: usize) -> [f32; 2] {
    // Use the actual sampled RGB pixel center (patch pixel [8,8]).
    [(i % width * 16) as f32 + 8.5, (i / width * 16) as f32 + 8.5]
}
fn distance(a: [f32; 2], b: [f32; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt() as f64
}
fn summarize(errors: &[f64], mutual_errors: &[f64], total: usize) -> serde_json::Value {
    let avg = |v: &[f64]| (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64);
    let pck = |v: &[f64], t: f64| {
        (!v.is_empty()).then(|| v.iter().filter(|&&e| e <= t).count() as f64 / v.len() as f64)
    };
    let mut sorted = errors.to_vec();
    sorted.sort_by(f64::total_cmp);
    serde_json::json!({"valid_queries":errors.len(),"total_queries":total,
        "mean_epe":avg(errors),"median_epe":sorted.get(sorted.len()/2),
        "pck8":pck(errors,8.),"pck16":pck(errors,16.),"pck32":pck(errors,32.),
        "mutual_matches":mutual_errors.len(),"mutual_coverage":if errors.is_empty(){0.}else{mutual_errors.len() as f64/errors.len() as f64},
        "mutual_epe":avg(mutual_errors),"mutual_pck16":pck(mutual_errors,16.)})
}

pub fn evaluate<B: Backend>(
    model: &LatentModel<B>,
    teacher: &VJepaEncoder<B>,
    scenes: &[RgbScene],
    entries: &[SceneEntry],
    dataset: &Path,
    path: &Path,
) -> Result<serde_json::Value> {
    let device = model.fusion.prediction.weight.device();
    let mut rows = Vec::new();
    let mut samples = Vec::new();
    let mut aggregate: BTreeMap<String, (Vec<f64>, Vec<f64>, usize)> = BTreeMap::new();
    for (s, scene) in scenes.iter().enumerate() {
        let grid = [scene.height / 16, scene.width / 16];
        let n = grid[0] * grid[1];
        let rgb: Vec<_> = (0..scene.views.len())
            .map(|v| image_tensor::<B>(scene, v, &device))
            .collect();
        let frozen = fixed_views(teacher, &model.encoder_config, &rgb);
        let student = model.encode_references(&rgb);
        // Geometry is never an input to any feature computation or match score.
        let geometry = load_geometry(&dataset.join("raw").join(&entries[s].file))?;
        for t in 0..rgb.len() {
            for r in 0..rgb.len() {
                if t == r {
                    continue;
                }
                let predictions = standard_readouts(
                    model,
                    student[t].clone(),
                    student[r].clone(),
                    frozen[t].clone(),
                    frozen[r].clone(),
                    grid,
                )?;
                let mut truth = vec![None; n];
                for (i, slot) in truth.iter_mut().enumerate() {
                    let pixel = (i / grid[1] * 16 + 8) * scene.width + i % grid[1] * 16 + 8;
                    if visibility(&geometry, t, r, pixel) != Visibility::Visible {
                        continue;
                    }
                    let p = &geometry.position[t][pixel * 3..pixel * 3 + 3];
                    let q = project(
                        [p[0], p[1], p[2]],
                        &geometry.world_from_view[r],
                        geometry.fovy[r],
                        scene.width,
                        scene.height,
                    )
                    .unwrap();
                    *slot = Some([q[0], q[1]]);
                }
                for (name, (indices, mutual)) in predictions {
                    let mut errors = Vec::new();
                    let mut mutual_errors = Vec::new();
                    let mut points = Vec::new();
                    for (i, gt) in truth.iter().enumerate() {
                        if let Some(gt) = gt {
                            let predicted = center(indices[i], grid[1]);
                            let error = distance(predicted, *gt);
                            errors.push(error);
                            if mutual[i] {
                                mutual_errors.push(error);
                            }
                            if s < 3 && t == 0 && r == 1 {
                                points.push(serde_json::json!({"query":center(i,grid[1]),"predicted":predicted,"truth":gt,"epe":error,"mutual":mutual[i]}));
                            }
                        }
                    }
                    let mut row = summarize(&errors, &mutual_errors, n);
                    row["room_seed"] = scene.seed.into();
                    row["target_view"] = t.into();
                    row["reference_view"] = r.into();
                    row["method"] = name.clone().into();
                    rows.push(row);
                    let all = aggregate.entry(name.clone()).or_default();
                    all.0.extend(errors);
                    all.1.extend(mutual_errors);
                    all.2 += n;
                    if !points.is_empty() {
                        samples.push(serde_json::json!({"room_seed":scene.seed,"target_view":t,"reference_view":r,"method":name,"points":points}));
                    }
                }
                let oracle: Vec<_> = truth
                    .iter()
                    .flatten()
                    .map(|gt| {
                        (0..n)
                            .map(|j| distance(center(j, grid[1]), *gt))
                            .min_by(f64::total_cmp)
                            .unwrap()
                    })
                    .collect();
                let a = aggregate.entry("grid_oracle".to_owned()).or_default();
                a.0.extend(oracle);
                a.2 += n;
            }
        }
        eprintln!("correspondence room {}/{}", s + 1, scenes.len());
    }
    let summaries: BTreeMap<_, _> = aggregate
        .iter()
        .map(|(k, (e, m, n))| (k.clone(), summarize(e, m, *n)))
        .collect();
    let result = serde_json::json!({"schema":1,"status":"synthetic_patch_grid_diagnostic",
        "protocol":"all directed view pairs; cosine hard nearest-neighbour; patch pixel [8,8] centers; GT depth visibility filters scoring only; 16px grid; no refinement or GT input; pair-conditioned final decoder features",
        "not_comparable_to":"ETH3D/HPatches full-resolution AEPE, ZeroCo attention aggregation or official Gekko benchmark",
        "rooms":scenes.len(),"rows":rows,"summary":summaries,"annotated_samples":samples});
    write_json(path, &result)?;
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn conditional_reciprocity_ignores_unidentifiable_row_offsets() {
        use burn::{backend::NdArray, tensor::TensorData};
        type B = NdArray<f32>;
        let d = Default::default();
        let x = Tensor::<B, 3>::from_data(
            TensorData::new(vec![3., 1., -1., 0., 4., 2., 2., 0., 5.], [1, 3, 3]),
            &d,
        );
        let y = x.clone().swap_dims(1, 2);
        let bias = Tensor::<B, 3>::from_data(TensorData::new(vec![13., -7., 5.], [1, 3, 1]), &d);
        let original = reciprocal_conditionals(vec![x.clone()], vec![y.clone()]);
        let shifted = reciprocal_conditionals(vec![x + bias.clone()], vec![y - bias]);
        assert!(crate::train::scalar((original - shifted).abs().max()).unwrap() < 1e-6);
    }
    #[test]
    fn fixed_and_student_readouts_use_identical_dense_batches() {
        use crate::model::DecoderConfig;
        use burn::backend::NdArray;
        use burn::module::Module;
        type B = NdArray<f32>;
        let d = Default::default();
        let ec = VJepaConfig::tiny_for_tests();
        let teacher = VJepaEncoder::<B>::new(&ec, &d);
        // Materialize Burn's lazy random parameters before cloning, just as
        // load_encoder does for every actual teacher/student package.
        let teacher = teacher.clone().load_record(teacher.into_record());
        let model = LatentModel::new(
            teacher.clone(),
            ec.clone(),
            &DecoderConfig {
                encoder_dim: ec.encoder.embed_dim,
                width: 32,
                depth: 1,
                heads: 4,
                patch: 16,
            },
            &d,
        )
        .unwrap();
        let rgb = vec![
            Tensor::full([1, 3, 32, 32], 0.2, &d),
            Tensor::full([1, 3, 32, 32], 0.8, &d),
        ];
        let fixed = fixed_views(&teacher, &ec, &rgb);
        let adapted = model.encode_references(&rgb);
        for (a, b) in fixed.iter().zip(&adapted) {
            assert_eq!(values((a.clone() - b.clone()).abs().max()).unwrap()[0], 0.);
        }
        let map = model
            .fusion
            .decoder
            .correspondence_attention(fixed[0].clone(), fixed[1].clone(), [2, 2])
            .unwrap();
        assert_eq!(map.dims(), [1, 4, 4]);
        assert!(values((map.sum_dim(2) - 1.).abs().max()).unwrap()[0] < 1e-6);
        let readouts: BTreeMap<_, _> = standard_readouts(
            &model,
            adapted[0].clone(),
            adapted[1].clone(),
            fixed[0].clone(),
            fixed[1].clone(),
            [2, 2],
        )
        .unwrap()
        .into_iter()
        .collect();
        assert_eq!(readouts.len(), 18);
        for (teacher, student) in [
            ("fixed_teacher", "student_encoder"),
            ("centered_teacher", "centered_student"),
            ("conditional_teacher", "conditional_student"),
            (
                "conditional_centered_teacher",
                "conditional_centered_student",
            ),
        ] {
            assert_eq!(readouts[teacher], readouts[student]);
        }
    }
    #[test]
    fn centered_readout_removes_image_offsets_without_losing_correspondence() {
        use burn::{backend::NdArray, tensor::TensorData};
        type B = NdArray<f32>;
        let d = Default::default();
        let a = Tensor::<B, 3>::from_data(
            TensorData::new(vec![1., 0., 0., 0., 2., 0., 0., 0., 3.], [1, 3, 3]),
            &d,
        );
        let batch = Tensor::<B, 3>::zeros([2, 3, 3], &d);
        assert_eq!(
            reciprocal_conditionals(
                vec![batch.clone(), batch.clone()],
                vec![batch.clone(), batch]
            )
            .dims(),
            [2, 3, 3]
        );
        let b = Tensor::<B, 3>::from_data(
            TensorData::new(vec![0., 0., 3., 1., 0., 0., 0., 2., 0.], [1, 3, 3]),
            &d,
        );
        let (indices, mutual) = centered_matches(a.clone() + 7., b.clone() - 3.).unwrap();
        assert_eq!(indices, vec![1, 2, 0]);
        assert!(mutual.iter().all(|&x| x));
        let conditional = conditional_centered_matches(a + 7., b - 3., 0.07).unwrap();
        assert_eq!(conditional, (indices, mutual));
    }
    #[test]
    fn matching_handles_transposition_ties_and_mutual_filtering() {
        let (a, m) = nearest(&[0., 3., 1., 4., 2., 0., 0., 1., 5.], 3);
        assert_eq!(a, vec![1, 0, 2]);
        assert_eq!(m, vec![true; 3]);
        let (a, m) = nearest(&[1., 1., 1., 1.], 2);
        assert_eq!(a, vec![0, 0]);
        assert_eq!(m, vec![true, false]);
        let v = summarize(&[0., 16., 32.], &[0.], 4);
        assert_eq!(v["mean_epe"], 16.);
        assert_eq!(v["pck16"], 2. / 3.);
        assert!(summarize(&[], &[], 0)["mean_epe"].is_null());
    }
}
