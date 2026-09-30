use burn::{
    backend::{Autodiff, NdArray},
    tensor::{Tensor, TensorData},
};
use burn_gekko::{
    encoder::{normalize, visible_mask},
    loss::{gekko_loss, rgb_patches},
    model::{DecoderConfig, GekkoDecoder, Predictions},
    tensor::scalar,
};
use burn_vjepa::{SparseTokenMask, VJepaConfig, VJepaEncoder};
type Cpu = NdArray<f32>;
type Ad = Autodiff<Cpu>;

#[test]
fn descriptor_only_forward_preserves_outputs_and_input_gradients() {
    let d = Default::default();
    let decoder = GekkoDecoder::<Ad>::new(
        &DecoderConfig {
            encoder_dim: 8,
            width: 16,
            depth: 2,
            heads: 2,
            patch: 16,
        },
        &d,
    )
    .unwrap();
    let a = Tensor::<Ad, 3>::from_data(
        TensorData::new(
            (0..32).map(|x| (x as f32 * 0.37).sin()).collect::<Vec<_>>(),
            [1, 4, 8],
        ),
        &d,
    )
    .require_grad();
    let b = (a.clone() * 0.7 + 0.3).detach().require_grad();
    let legacy = decoder
        .pair_training(a.clone(), b.clone(), [2, 2])
        .unwrap()
        .features;
    let fast = decoder.pair_features(a.clone(), b.clone(), [2, 2]).unwrap();
    assert!(scalar((legacy.clone() - fast.clone()).abs().max()).unwrap() < 1e-7);
    let old_grad = legacy.exp().mean().backward();
    let new_grad = fast.exp().mean().backward();
    for x in [a, b] {
        assert!(
            scalar(
                (x.grad(&old_grad).unwrap() - x.grad(&new_grad).unwrap())
                    .abs()
                    .max()
            )
            .unwrap()
                < 1e-6
        );
    }
}

#[test]
fn pair_trace_reproduces_forward_and_isolates_cross_position_intervention() {
    use burn_gekko::model::DecoderPosition;
    let device = Default::default();
    let config = DecoderConfig {
        encoder_dim: 8,
        width: 16,
        depth: 2,
        heads: 2,
        patch: 16,
    };
    let decoder = GekkoDecoder::<Cpu>::with_position(&config, DecoderPosition::Rope2d, &device)
        .unwrap()
        .with_attention_normalization(true, &device);
    let make = |offset: f32| {
        Tensor::from_data(
            TensorData::new(
                (0..32)
                    .map(|i| (i as f32 * 0.37 + offset).sin())
                    .collect::<Vec<_>>(),
                [1, 4, 8],
            ),
            &device,
        )
    };
    let (a, b) = (make(0.), make(0.6));
    let trace = decoder
        .pair_trace(a.clone(), b.clone(), [2, 2], true)
        .unwrap();
    let training = decoder.pair_training(a.clone(), b.clone(), [2, 2]).unwrap();
    assert!(scalar((training.features - trace.features.clone()).abs().max()).unwrap() < 1e-6);
    for (score, layer) in training.logits.into_iter().zip(&trace.layers) {
        assert!(scalar((score - layer.logits.clone()).abs().max()).unwrap() < 1e-6);
    }
    let output = decoder
        .relative_improvement_features(a.clone(), vec![b.clone()], [2, 2])
        .unwrap();
    assert!(scalar((trace.features - output).abs().max()).unwrap() < 1e-6);
    let means = Tensor::cat(
        trace.layers.iter().map(|x| x.probability.clone()).collect(),
        0,
    )
    .mean_dim(0);
    let old_map = decoder
        .correspondence_attention(a.clone(), b.clone(), [2, 2])
        .unwrap();
    assert!(scalar((means - old_map).abs().max()).unwrap() < 1e-6);
    let changed = decoder.pair_trace(a, b, [2, 2], false).unwrap();
    // Before the first cross update the Q/K content is identical. Removing
    // only cross-view RoPE must alter its logits, but not these content scores.
    assert!(
        scalar(
            (trace.layers[0].content.clone() - changed.layers[0].content.clone())
                .abs()
                .max()
        )
        .unwrap()
            < 1e-6
    );
    assert!(
        scalar(
            (trace.layers[0].logits.clone() - changed.layers[0].logits.clone())
                .abs()
                .max()
        )
        .unwrap()
            > 1e-4
    );
    for layer in &trace.layers {
        assert!(scalar((layer.probability.clone().sum_dim(2) - 1.).abs().max()).unwrap() < 1e-6);
        assert!(scalar(layer.centered_content.clone().sum_dim(1).abs().max()).unwrap() < 1e-5);
        assert!(scalar(layer.centered_content.clone().sum_dim(2).abs().max()).unwrap() < 1e-5);
    }
}

#[test]
fn cross_view_position_policy_retains_monocular_attention_and_record_compatibility() {
    use burn_gekko::model::DecoderPosition;
    let device = Default::default();
    let config = DecoderConfig {
        encoder_dim: 8,
        width: 16,
        depth: 2,
        heads: 2,
        patch: 16,
    };
    let decoder = GekkoDecoder::<Cpu>::with_position(&config, DecoderPosition::Rope2d, &device)
        .unwrap()
        .with_mae_context_before_self(true)
        .with_attention_normalization(true, &device);
    let target = Tensor::from_data(
        TensorData::new(
            (0..16).map(|i| (i as f32 * 0.31).sin()).collect(),
            [1, 2, 8],
        ),
        &device,
    );
    let reference = Tensor::from_data(
        TensorData::new(
            (0..32).map(|i| (i as f32 * 0.77).cos()).collect(),
            [1, 4, 8],
        ),
        &device,
    );
    let mask = SparseTokenMask::new(vec![0, 3], 4).unwrap();
    let original = decoder
        .reconstruction_features(target.clone(), vec![reference.clone()], &mask, [2, 2])
        .unwrap();
    let changed = decoder
        .with_cross_view_rope(false)
        .reconstruction_features(target, vec![reference], &mask, [2, 2])
        .unwrap();
    assert!(scalar((original.0 - changed.0).abs().max()).unwrap() > 1e-4);
    assert_eq!(scalar((original.1 - changed.1).abs().max()).unwrap(), 0.);
}

#[test]
fn predicted_statistics_produce_rgb_and_receive_only_reconstruction_gradients() {
    use burn_gekko::loss::{calibrated_rgb, reconstruction_loss};
    let device = Default::default();
    let target = Tensor::<Ad, 3>::from_data(
        TensorData::new(vec![0.1, 0.3, 0.5, 0.2, 0.4, 0.6], [1, 2, 3]),
        &device,
    );
    let prediction = Tensor::<Ad, 3>::from_data(
        TensorData::new(vec![0., 0., 0., 0.4, -2., 0., 0., 0., 0.4, -2.], [1, 2, 5]),
        &device,
    )
    .require_grad();
    let rgb = calibrated_rgb(prediction.clone(), 3);
    assert!((scalar(rgb.mean()).unwrap() - 0.4).abs() < 1e-6);
    let mask = SparseTokenMask::new(vec![0], 2).unwrap();
    let output = || Predictions {
        cross_rgb: prediction.clone(),
        mae_rgb: prediction.clone(),
        ri: Tensor::<Ad, 3>::zeros([1, 2, 1], &device).require_grad(),
    };
    let losses = reconstruction_loss(output(), target.clone(), &mask, true, true);
    let grads = losses.total.backward();
    let g = prediction
        .grad(&grads)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert!(g[..5].iter().all(|v| v.abs() < 1e-7));
    assert!(g[9].abs() > 1e-3, "scale calibration must train");
    let ri = reconstruction_loss(output(), target, &mask, true, true).ri;
    assert!(
        prediction.grad(&ri.backward()).is_none(),
        "RI must not backpropagate into either RGB content or statistics"
    );
}

#[test]
fn appearance_bypass_cannot_read_hidden_rgb_but_retains_visible_detail() {
    use burn_gekko::encoder::{ImageFeatures, image_features};
    let device = Default::default();
    let mask = SparseTokenMask::new(vec![0, 3], 4).unwrap();
    let a = vec![0.25f32; 3 * 32 * 32];
    let mut b = a.clone();
    for c in 0..3 {
        for y in 0..32 {
            for x in 0..32 {
                if !mask.indices().contains(&((y / 16) * 2 + x / 16)) {
                    b[c * 32 * 32 + y * 32 + x] = 0.9;
                }
            }
        }
    }
    let make = |v| {
        image_features(
            Tensor::<Cpu, 3>::zeros([1, 2, 8], &device),
            Tensor::from_data(TensorData::new(v, [1, 3, 32, 32]), &device),
            Some(&mask),
            ImageFeatures::SemanticRgb,
        )
    };
    let first = make(a);
    assert_eq!(first.dims(), [1, 2, 776]);
    assert_eq!(
        scalar((first.clone() - make(b.clone())).abs().max()).unwrap(),
        0.0
    );
    b[0] = 0.6;
    assert!(scalar((first - make(b)).abs().max()).unwrap() > 0.5);
}

#[test]
fn hidden_target_pixels_cannot_change_visible_encoder_features() {
    let config = VJepaConfig::tiny_for_tests();
    let device = Default::default();
    let encoder = VJepaEncoder::<Cpu>::new(&config, &device);
    let mask = SparseTokenMask::new(vec![0, 3], 4).unwrap();
    let mut a = vec![0.25; 3 * 32 * 32];
    let b = a.clone();
    for c in 0..3 {
        for y in 0..32 {
            for x in 0..32 {
                if !mask.indices().contains(&((y / 16) * 2 + x / 16)) {
                    a[c * 32 * 32 + y * 32 + x] = 0.95;
                }
            }
        }
    }
    let a = normalize(
        Tensor::from_data(TensorData::new(a, [1, 3, 32, 32]), &device),
        &config,
    );
    let b = normalize(
        Tensor::from_data(TensorData::new(b, [1, 3, 32, 32]), &device),
        &config,
    );
    let masked_a = encoder.forward_image(a.clone(), Some(&mask)).tokens;
    let masked_b = encoder.forward_image(b.clone(), Some(&mask)).tokens;
    assert!(scalar((masked_a - masked_b).abs().max()).unwrap() < 1e-6);
    let full_a = encoder.forward_image(a, None).tokens;
    let full_b = encoder.forward_image(b, None).tokens;
    assert!(scalar((full_a - full_b).abs().max()).unwrap() > 1e-4);
}

#[test]
fn released_ri_equation_matches_scalar_oracle_and_stops_rgb_gradients() {
    let device = Default::default();
    let mae = Tensor::<Ad, 3>::full([1, 2, 3], 2.0, &device).require_grad();
    let cross = Tensor::<Ad, 3>::full([1, 2, 3], 1.0, &device).require_grad();
    let ri = Tensor::<Ad, 3>::full([1, 2, 1], 0.5, &device).require_grad();
    let mask = SparseTokenMask::new(vec![0], 2).unwrap();
    let losses = gekko_loss(
        Predictions {
            mae_rgb: mae.clone(),
            cross_rgb: cross.clone(),
            ri: ri.clone(),
        },
        Tensor::zeros([1, 2, 3], &device),
        &mask,
    );
    assert!((scalar(losses.total).unwrap() - 6.0).abs() < 1e-6);
    let grads = losses.ri.backward();
    assert!(mae.grad(&grads).is_none());
    assert!(cross.grad(&grads).is_none());
    let grad = ri
        .grad(&grads)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(grad, vec![0.0, -8.0]);
}

#[test]
fn patch_targets_keep_rgb_order_and_normalize_per_patch() {
    let device = Default::default();
    let image = Tensor::<Cpu, 4>::from_data(
        TensorData::new(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            [1, 3, 2, 2],
        ),
        &device,
    );
    let raw = rgb_patches(image.clone(), 2, false)
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    assert_eq!(raw, vec![1., 5., 9., 2., 6., 10., 3., 7., 11., 4., 8., 12.]);
    let normalized = rgb_patches(image, 2, true);
    assert!(scalar(normalized.clone().mean().abs()).unwrap() < 1e-6);
    assert!((scalar(normalized.powf_scalar(2.0).sum() / 11.0).unwrap() - 1.0).abs() < 1e-5);
}

#[test]
fn reference_set_order_is_invariant_and_all_three_heads_receive_gradients() {
    let device = Default::default();
    let model = GekkoDecoder::<Ad>::with_position(
        &DecoderConfig {
            encoder_dim: 8,
            width: 16,
            depth: 1,
            heads: 4,
            patch: 16,
        },
        burn_gekko::model::DecoderPosition::Rope2d,
        &device,
    )
    .unwrap();
    let masked = Tensor::full([1, 1, 8], 0.1, &device);
    let full = Tensor::full([1, 4, 8], 0.2, &device);
    let r1 = Tensor::full([1, 4, 8], 0.3, &device);
    let r2 = Tensor::full([1, 4, 8], 0.7, &device);
    let mask = SparseTokenMask::new(vec![2], 4).unwrap();
    let a = model
        .forward(
            masked.clone(),
            full.clone(),
            vec![r1.clone(), r2.clone()],
            &mask,
            [2, 2],
        )
        .unwrap();
    let b = model
        .forward(masked, full, vec![r2, r1], &mask, [2, 2])
        .unwrap();
    assert!(scalar((a.ri.clone() - b.ri).abs().max()).unwrap() < 2e-5);
    assert!(scalar((a.cross_rgb.clone() - b.cross_rgb).abs().max()).unwrap() < 2e-5);
    assert!(scalar((a.mae_rgb.clone() - b.mae_rgb).abs().max()).unwrap() < 1e-6);
    let loss = gekko_loss(a, Tensor::zeros([1, 4, 768], &device), &mask);
    let grads = loss.total.backward();
    for weight in [
        &model.cross_rgb.weight,
        &model.mae_rgb.weight,
        &model.ri.weight,
    ] {
        let grad = weight.val().grad(&grads).expect("head has gradient");
        assert!(scalar(grad.abs().max()).unwrap() > 0.0);
    }
}

#[test]
fn masks_replay_from_step_and_seed_and_are_not_prefix_masks() {
    let a = visible_mask(64, 0.75, 17, 3).unwrap();
    assert_eq!(a, visible_mask(64, 0.75, 17, 3).unwrap());
    assert_ne!(a, visible_mask(64, 0.75, 17, 4).unwrap());
    assert_eq!(a.len(), 16);
    assert_ne!(a.indices(), &(0..16).collect::<Vec<_>>());
    assert!(visible_mask(1, 0.75, 17, 0).is_err());
}

#[test]
fn full_target_branch_cannot_leak_into_either_reconstruction() {
    let device = Default::default();
    let model = GekkoDecoder::<Cpu>::new(
        &DecoderConfig {
            encoder_dim: 8,
            width: 16,
            depth: 1,
            heads: 4,
            patch: 16,
        },
        &device,
    )
    .unwrap();
    let masked = Tensor::full([1, 1, 8], 0.1, &device);
    let full = Tensor::full([1, 4, 8], 0.2, &device);
    let altered = Tensor::from_data(
        TensorData::new((0..32).map(|i| i as f32 / 7.0).collect(), [1, 4, 8]),
        &device,
    );
    let references = vec![Tensor::full([1, 4, 8], 0.7, &device)];
    let mask = SparseTokenMask::new(vec![2], 4).unwrap();
    let a = model
        .forward(masked.clone(), full, references.clone(), &mask, [2, 2])
        .unwrap();
    let b = model
        .forward(masked, altered, references, &mask, [2, 2])
        .unwrap();
    assert_eq!(
        scalar((a.cross_rgb - b.cross_rgb).abs().max()).unwrap(),
        0.0
    );
    assert_eq!(scalar((a.mae_rgb - b.mae_rgb).abs().max()).unwrap(), 0.0);
    assert!(scalar((a.ri - b.ri).abs().max()).unwrap() > 1e-4);
}
