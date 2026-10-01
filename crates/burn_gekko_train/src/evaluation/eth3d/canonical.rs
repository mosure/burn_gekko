//! Preserve the broad exporter's calculation order and retain its score arrays.
//! All three refined controls use the very same arrays as their hard matches.
use super::*;
use crate::{correspondence, encoder_audit, evaluation::refinement};

pub(super) fn readouts<B: Backend>(
    model: &crate::latent::LatentModel<B>,
    teacher: &burn_vjepa::VJepaEncoder<B>,
    views: &[Tensor<B, 4>],
    student: &[Tensor<B, 3>],
    layer: usize,
) -> Result<Vec<refinement::Readout>> {
    let fixed = fixed_views(teacher, &model.encoder_config, views);
    let scored = correspondence::standard_readouts_with_spatial_scores(
        model,
        student[0].clone(),
        student[1].clone(),
        fixed[0].clone(),
        fixed[1].clone(),
        [16, 16],
    )?;
    let mut hard = scored.readouts;
    let pair_scores = scored.scores;
    let a = correspondence::self_conditioned_descriptor(model, student[0].clone(), [16, 16])?;
    let b = correspondence::self_conditioned_descriptor(model, student[1].clone(), [16, 16])?;
    let scored = correspondence::self_conditioned_readouts_with_scores(a, b)?;
    hard.extend(scored.readouts);
    let self_scores = scored.scores;
    let encoder_name = format!("student_l{layer:02}_centered_conditional");
    let mut encoder_scores = None;
    for (name, encoder) in [("student", &model.encoder), ("teacher", teacher)] {
        let layers = encoder_audit::capture_view_layers(
            encoder,
            &model.encoder_config,
            views,
            &[layer - 1],
        )?;
        let mut scored = encoder_audit::readouts_with_scores(name, &layers, 0, 1)?;
        hard.extend(scored.readouts);
        if name == "student" {
            encoder_scores = scored.scores.remove(&encoder_name);
        }
    }
    let pair_scores =
        pair_scores.ok_or_else(|| anyhow::anyhow!("missing canonical spatial scores"))?;
    let encoder_scores =
        encoder_scores.ok_or_else(|| anyhow::anyhow!("missing canonical encoder scores"))?;
    let mut result = Vec::new();
    for (name, scores) in [
        ("spatial_residual_conditional", pair_scores),
        ("spatial_self_conditional", self_scores),
        (encoder_name.as_str(), encoder_scores),
    ] {
        let readouts = refinement::score_readouts(name, &scores, [16, 16])?;
        ensure!(
            hard.iter().any(|(n, (indices, mutual))| n == name
                && indices == &readouts[0].indices
                && mutual == &readouts[0].mutual),
            "canonical hard/local score consistency failed for {name}"
        );
        result.extend(readouts);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{backend::NdArray, module::Param, tensor::TensorData};

    #[test]
    fn canonical_local_controls_share_hard_matches_and_match_cpu_reference() {
        type B = NdArray<f32>;
        let device = Default::default();
        let mut config = burn_vjepa::VJepaConfig::tiny_for_tests();
        config.encoder.depth = 12;
        config.encoder.n_output_distillation = 4;
        let teacher = burn_vjepa::VJepaEncoder::<B>::new(&config, &device);
        let mut model = crate::latent::LatentModel::new(
            teacher.clone(),
            config.clone(),
            &crate::model::DecoderConfig {
                encoder_dim: config.encoder.embed_dim,
                width: 32,
                depth: 1,
                heads: 4,
                patch: 16,
            },
            &device,
        )
        .unwrap()
        .with_spatial_input_layer(Some(6), true)
        .unwrap()
        .prepare_spatial_descriptor(Some(&burn_gekko::heads::spatial::SpatialDescriptorConfig {
            residual_radius: 0.25,
        }))
        .unwrap();
        let weight = &mut model
            .fusion
            .spatial_descriptor
            .as_mut()
            .unwrap()
            .correction
            .weight;
        let shape = weight.dims();
        *weight = Param::from_tensor(Tensor::from_data(
            TensorData::new(
                (0..shape.iter().product())
                    .map(|i| (i as f32 * 0.17).sin() * 0.03)
                    .collect(),
                shape,
            ),
            &device,
        ));
        let rgb = Tensor::<B, 4>::from_data(
            TensorData::new(
                (0..3 * 256 * 256)
                    .map(|i| (i as f32 * 0.013).sin() * 0.5 + 0.5)
                    .collect(),
                [1, 3, 256, 256],
            ),
            &device,
        );
        let views = vec![rgb.clone(), rgb.flip([3])];
        let encoded = model.encode_references(&views);
        let canonical = readouts(&model, &teacher, &views, &encoded, 6).unwrap();
        let focused = refinement::spatial_readouts(
            &model,
            encoded[0].clone(),
            encoded[1].clone(),
            [16, 16],
            6,
            None,
        )
        .unwrap();
        assert_eq!(canonical.len(), 6);
        for (a, b) in canonical.iter().zip(&focused) {
            assert_eq!(a.method, b.method);
            assert_eq!(a.indices, b.indices);
            assert_eq!(a.mutual, b.mutual);
            assert_eq!(a.coordinates.is_some(), b.coordinates.is_some());
            if let (Some(a), Some(b)) = (&a.coordinates, &b.coordinates) {
                assert!(
                    a.iter()
                        .zip(b)
                        .all(|(a, b)| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5))
                );
            }
        }
        for pair in canonical.as_chunks::<2>().0 {
            assert_eq!(pair[0].indices, pair[1].indices);
            assert_eq!(pair[0].mutual, pair[1].mutual);
        }
    }
}
