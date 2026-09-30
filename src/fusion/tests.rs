use super::*;
#[test]
fn direct_spatial_input_starts_neutral_learns_and_preserves_hidden_rgb_isolation() {
    use crate::latent::LatentModel;
    use burn_vjepa::{VJepaConfig, VJepaEncoder};
    type B = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
    let device = Default::default();
    let mut ec = VJepaConfig::tiny_for_tests();
    ec.encoder.depth = 12;
    ec.encoder.n_output_distillation = 4;
    let model = LatentModel::<B>::new(
        VJepaEncoder::new(&ec, &device),
        ec.clone(),
        &DecoderConfig {
            encoder_dim: 32,
            width: 32,
            depth: 1,
            heads: 4,
            patch: 16,
        },
        &device,
    )
    .unwrap()
    .train_encoder(0);
    assert!(
        model
            .clone()
            .with_spatial_input_layer(Some(6), false)
            .is_err()
    );
    assert!(
        model
            .clone()
            .with_spatial_input_layer(Some(5), true)
            .is_err()
    );
    let routed = model
        .clone()
        .with_spatial_input_layer(Some(6), true)
        .unwrap();
    assert!(model.clone().with_spatial_input_scale(0.).is_err());
    assert!(routed.clone().with_spatial_input_scale(f64::NAN).is_err());
    let control = routed.clone().with_spatial_input_scale(0.).unwrap();
    assert!(
        routed
            .clone()
            .with_spatial_input_layer(None, false)
            .is_err()
    );
    let mut values: Vec<_> = (0..3 * 32 * 32)
        .map(|i| (i as f32 * 0.031).sin() * 0.4 + 0.5)
        .collect();
    let image = Tensor::<B, 4>::from_data(TensorData::new(values.clone(), [1, 3, 32, 32]), &device);
    let mask = SparseTokenMask::new(vec![0, 2], 4).unwrap();
    let encoded = routed.encode(image.clone(), Some(&mask));
    assert_eq!(encoded.dims(), [1, 2, 64]);
    assert_eq!(
        crate::tensor::scalar(
            control
                .encode(image.clone(), Some(&mask))
                .slice_dim(2, 32..64)
                .abs()
                .max()
        )
        .unwrap(),
        0.
    );
    assert!(
        crate::tensor::scalar(
            (routed.final_encoder_features(encoded.clone())
                - model.encode(image.clone(), Some(&mask)))
            .abs()
            .max()
        )
        .unwrap()
            < 1e-6
    );
    for channel in 0..3 {
        for y in 0..32 {
            for x in 16..32 {
                values[channel * 32 * 32 + y * 32 + x] =
                    1.0 - values[channel * 32 * 32 + y * 32 + x];
            }
        }
    }
    let changed = Tensor::from_data(TensorData::new(values, [1, 3, 32, 32]), &device);
    assert_eq!(
        crate::tensor::scalar((encoded - routed.encode(changed, Some(&mask))).abs().max()).unwrap(),
        0.0
    );
    let references = vec![image.clone() * 0.7];
    let old = model.predict(image.clone(), &references, &mask).unwrap();
    let balanced = control.predict(image.clone(), &references, &mask).unwrap();
    let new = routed.predict(image, &references, &mask).unwrap();
    assert!(
        crate::tensor::scalar((balanced.cross - new.cross.clone()).abs().max()).unwrap() < 1e-6
    );
    assert!(crate::tensor::scalar((old.cross - new.cross.clone()).abs().max()).unwrap() < 1e-5);
    assert!(crate::tensor::scalar((old.monocular - new.monocular).abs().max()).unwrap() < 1e-5);
    let gradients = new.cross.powf_scalar(2.).mean().backward();
    let projection = routed
        .fusion
        .decoder
        .projection
        .weight
        .val()
        .grad(&gradients)
        .unwrap();
    assert!(crate::tensor::scalar(projection.slice_dim(0, 32..64).abs().max()).unwrap() > 1e-6);
    assert!(
        routed.encoder.blocks[0]
            .attn
            .qkv
            .weight
            .val()
            .grad(&gradients)
            .is_none()
    );
}
#[test]
fn stable_attention_preserves_output_gradient_and_set_symmetry() {
    type B = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
    let d = Default::default();
    let mut attention = Attention::<B>::new(16, 4, &d);
    let input = || {
        Tensor::from_data(
            TensorData::new(
                (0..64).map(|i| (i as f32 * 0.17).sin()).collect(),
                [1, 4, 16],
            ),
            &d,
        )
        .require_grad()
    };
    let context = Tensor::from_data(
        TensorData::new(
            (0..128).map(|i| (i as f32 * 0.11).cos()).collect(),
            [1, 8, 16],
        ),
        &d,
    );
    let x = input();
    let fast = attention.forward(x.clone(), context.clone(), None);
    let fast_values = fast.clone().detach();
    let fast_grad = x.grad(&fast.powf_scalar(2.).sum().backward()).unwrap();
    attention.stable_attention = true;
    let x = input();
    let stable = attention.forward(x.clone(), context.clone(), None);
    assert!(
        crate::tensor::scalar((fast_values - stable.clone().detach()).abs().max()).unwrap() < 1e-6
    );
    let grad = x
        .grad(&stable.clone().powf_scalar(2.).sum().backward())
        .unwrap();
    assert!(crate::tensor::scalar((fast_grad - grad).abs().max()).unwrap() < 1e-5);
    let permuted = Tensor::cat(
        vec![
            context.clone().slice_dim(1, 4..8),
            context.slice_dim(1, 0..4),
        ],
        1,
    );
    let changed = attention.forward(input(), permuted, None);
    assert!(
        crate::tensor::scalar((stable.detach() - changed.detach()).abs().max()).unwrap() < 1e-6
    );
}
#[test]
fn corrected_mae_matches_passing_the_pre_block_target_as_its_reference() {
    type B = burn::backend::NdArray<f32>;
    let device = Default::default();
    let block = Block::<B>::new(16, 4, &device);
    let x = Tensor::from_data(
        TensorData::new(
            (0..64).map(|i| (i as f32 * 0.19).sin()).collect(),
            [1, 4, 16],
        ),
        &device,
    );
    let expected = block.forward(x.clone(), Some(x.clone()), None, false, true);
    let corrected = block.forward(x.clone(), None, None, true, true);
    let legacy = block.forward(x, None, None, false, true);
    assert!(crate::tensor::scalar((corrected.clone() - expected).abs().max()).unwrap() < 1e-6);
    assert!(crate::tensor::scalar((corrected - legacy).abs().max()).unwrap() > 1e-4);
}
