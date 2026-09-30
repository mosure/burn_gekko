use burn::{
    backend::{Autodiff, NdArray},
    module::AutodiffModule,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    tensor::{Tensor, TensorData},
};
use burn_gekko_train::{
    e2e::{RgbHead, direct_rgb_loss, take_gradients},
    e2e_pilot::{E2eConfig, Initialization, RgbCache, host_batch, initialize},
    encoder::visible_mask,
    train::scalar,
};
type B = Autodiff<NdArray<f32>>;
fn config() -> E2eConfig {
    E2eConfig {
        dataset: ".data/unused".into(),
        initialization: Initialization::Scratch {
            width: 32,
            depth: 2,
            heads: 4,
        },
        seed: 91,
        train_rooms: 1,
        batch_size: 1,
        steps: 2,
        decay_steps: 4,
        warmup_steps: 1,
        max_seconds: 60,
        eval_every: 1,
        decoder_width: 32,
        decoder_depth: 1,
        decoder_heads: 4,
        decoder_qk_norm: true,
        decoder_stable_attention: false,
        learning_rate: 0.001,
        encoder_lr_ratio: 0.1,
        weight_decay: 0.,
        mask_ratio: 0.5,
        references: 2,
        appearance_bypass: true,
        edge_loss_weight: 1.,
        gradient_energy_weight: 0.,
        unfreeze: true,
        unfreeze_min_steps: 1,
        unfreeze_min_improvement: 0.01,
        fixed_example: false,
        rgb_head: RgbHead::Direct,
        rgb_cache: RgbCache::Host,
        rgb_cache_max_mib: 128,
        checkpoint_every: 1,
        ri_start_step: 0,
        ri_weight: 0.1,
        probe_rooms: 4,
        appearance_transport: false,
        transport_max_displacement: 4.,
        transport_loss_weight: 5.,
        transport_pyramid_loss: false,
        transport_coarse_smoothness_weight: 0.,
        transport_matching: false,
        synthetic_reference_probability: 0.,
    }
}

#[test]
fn direct_rgb_loss_has_zero_perfect_error_and_no_observed_pixel_gradient() {
    let target = rgb(0.);
    let mask = visible_mask(4, 0.5, 17, 0).unwrap();
    assert_eq!(
        scalar(direct_rgb_loss(target.clone(), target.clone(), &mask, 1.)).unwrap(),
        0.
    );
    let prediction = Tensor::<B, 4>::zeros([1, 3, 32, 32], &Default::default()).require_grad();
    let loss = direct_rgb_loss(prediction.clone(), target, &mask, 1.);
    assert!(scalar(loss.clone()).unwrap() > 0.);
    let grad = prediction
        .grad(&loss.backward())
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let mut hidden_signal = 0.;
    for (i, g) in grad.into_iter().enumerate() {
        if mask
            .indices()
            .contains(&((i % 1024) / 32 / 16 * 2 + i % 32 / 16))
        {
            assert_eq!(g, 0.);
        } else {
            hidden_signal += g.abs();
        }
    }
    assert!(hidden_signal > 0.);
}

#[test]
fn host_batch_preserves_room_view_and_pixel_order() {
    let scenes: Vec<_> = (0..3)
        .map(|s| burn_gekko_data::RgbScene {
            seed: s,
            width: 32,
            height: 32,
            views: (0..3)
                .map(|v| {
                    (0..3072)
                        .map(|i| (s * 10000 + v * 3072 + i) as f32 / 50000.)
                        .collect()
                })
                .collect(),
        })
        .collect();
    let samples = [(2, 1), (0, 2)];
    let (target, references) =
        host_batch::<NdArray<f32>>(&scenes, &samples, 2, &Default::default());
    for (offset, batched) in std::iter::once(target).chain(references).enumerate() {
        let expected = Tensor::cat(
            samples
                .iter()
                .map(|&(s, v)| {
                    burn_gekko_train::encoder::image_tensor(
                        &scenes[s],
                        (v + offset) % 3,
                        &Default::default(),
                    )
                })
                .collect(),
            0,
        );
        assert_eq!(scalar((batched - expected).abs().max()).unwrap(), 0.);
    }
}
fn rgb(offset: f32) -> Tensor<B, 4> {
    Tensor::from_data(
        TensorData::new(
            (0..3072)
                .map(|i| 0.4 + 0.2 * (i as f32 * 0.17 + offset).sin())
                .collect(),
            [1, 3, 32, 32],
        ),
        &Default::default(),
    )
}
#[test]
fn frozen_tail_and_full_stages_route_gradients_and_update_only_enabled_parameters() {
    let (mut model, _) = initialize::<B>(&config(), &Default::default()).unwrap();
    let mask = visible_mask(4, 0.5, 17, 0).unwrap();
    let target = rgb(0.);
    let refs = vec![rgb(0.1), rgb(0.2)];
    let mut enc_opt = AdamWConfig::new()
        .with_weight_decay(0.)
        .init::<B, burn_vjepa::VJepaEncoder<B>>();
    let mut dec_opt = AdamWConfig::new()
        .with_weight_decay(0.)
        .init::<B, burn_gekko_train::model::GekkoDecoder<B>>();
    for blocks in [0, 1, 3] {
        model = model.train_encoder(blocks);
        let first = model.encoder.blocks[0].attn.qkv.weight.val().detach();
        let last = model.encoder.blocks[1].attn.qkv.weight.val().detach();
        let stem = model.encoder.image_patch_embed.proj.weight.val().detach();
        let head = model.decoder.cross_rgb.weight.val().detach();
        let loss = model
            .training_loss(target.clone(), &refs, &mask, 1., 0.)
            .unwrap()
            .0;
        let mut grads = GradientsParams::from_grads(loss.backward(), &model);
        let enc_grads = take_gradients(&model.encoder, &mut grads);
        assert_eq!(enc_grads.is_empty(), blocks == 0);
        if blocks > 0 {
            model.encoder = enc_opt.step(0.0001, model.encoder, enc_grads);
        }
        model.decoder = dec_opt.step(0.001, model.decoder, grads);
        assert_eq!(
            scalar(
                (model.encoder.blocks[0].attn.qkv.weight.val().detach() - first)
                    .abs()
                    .max()
            )
            .unwrap()
                > 0.,
            blocks == 3
        );
        assert_eq!(
            scalar(
                (model.encoder.blocks[1].attn.qkv.weight.val().detach() - last)
                    .abs()
                    .max()
            )
            .unwrap()
                > 0.,
            blocks > 0
        );
        assert_eq!(
            scalar(
                (model.encoder.image_patch_embed.proj.weight.val().detach() - stem)
                    .abs()
                    .max()
            )
            .unwrap()
                > 0.,
            blocks == 3
        );
        assert!(
            scalar(
                (model.decoder.cross_rgb.weight.val().detach() - head)
                    .abs()
                    .max()
            )
            .unwrap()
                > 0.
        );
    }
}
#[test]
fn end_to_end_rgb_is_invariant_to_hidden_target_and_reference_order() {
    assert_input_isolation(false);
}
#[test]
fn transported_rgb_is_invariant_to_hidden_target_and_reference_order() {
    assert_input_isolation(true);
}
fn assert_input_isolation(transport: bool) {
    check_input_isolation(transport, false);
}
#[test]
fn matched_rgb_is_invariant_to_hidden_target_and_reference_order() {
    check_input_isolation(true, true);
}
fn check_input_isolation(transport: bool, matching: bool) {
    let mut c = config();
    c.appearance_transport = transport;
    c.transport_matching = matching;
    if matching {
        c.transport_max_displacement = 16.;
    }
    c.transport_coarse_smoothness_weight = if transport { 0.01 } else { 0. };
    let (model, _) = initialize::<B>(&c, &Default::default()).unwrap();
    let model = model.train_encoder(3);
    let mask = visible_mask(4, 0.5, 17, 0).unwrap();
    let target = rgb(0.).require_grad();
    let refs = vec![rgb(0.1), rgb(0.2)];
    let output = model.complete(target.clone(), &refs, &mask).unwrap();
    let gradients = output.rgb.clone().sum().backward();
    let grad = target
        .grad(&gradients)
        .unwrap()
        .into_data()
        .to_vec::<f32>()
        .unwrap();
    let mut pixels = target.clone().into_data().to_vec::<f32>().unwrap();
    for c in 0..3 {
        for y in 0..32 {
            for x in 0..32 {
                if !mask.indices().contains(&(y / 16 * 2 + x / 16)) {
                    let i = c * 1024 + y * 32 + x;
                    assert_eq!(grad[i], 0.);
                    pixels[i] = 1.7;
                }
            }
        }
    }
    let changed = Tensor::from_data(TensorData::new(pixels, [1, 3, 32, 32]), &Default::default());
    let other = model.complete(changed, &refs, &mask).unwrap();
    if transport {
        for (normal, changed) in output
            .transport
            .as_ref()
            .unwrap()
            .coarse_flows
            .iter()
            .zip(&other.transport.as_ref().unwrap().coarse_flows)
        {
            assert_eq!(
                scalar((normal.clone() - changed.clone()).abs().max()).unwrap(),
                0.
            );
        }
    }
    assert_eq!(
        scalar((output.rgb.clone() - other.rgb).abs().max()).unwrap(),
        0.
    );
    let swapped = model
        .complete(target, &[refs[1].clone(), refs[0].clone()], &mask)
        .unwrap();
    assert!(scalar((output.rgb - swapped.rgb).abs().max()).unwrap() < 1e-5);
    assert_eq!(
        scalar((output.monocular - swapped.monocular).abs().max()).unwrap(),
        0.
    );
    // Validation conversion must not change the model or parameter data.
    let before = model.decoder.cross_rgb.weight.val().inner();
    assert_eq!(
        scalar(
            (before - model.valid().decoder.cross_rgb.weight.val())
                .abs()
                .max()
        )
        .unwrap(),
        0.
    );
}
#[test]
fn appearance_head_receives_gradients_and_ri_uses_the_blended_rgb() {
    assert_appearance_training(false, false);
}
#[test]
fn appearance_image_pyramid_receives_gradients_and_preserves_ri_contract() {
    assert_appearance_training(true, false);
}
#[test]
fn correspondence_encoder_updates_from_rgb_loss() {
    assert_appearance_training(true, true);
}
fn assert_appearance_training(pyramid: bool, matching: bool) {
    let mut c = config();
    c.appearance_transport = true;
    c.transport_pyramid_loss = pyramid;
    c.transport_matching = matching;
    if matching {
        c.transport_max_displacement = 16.;
    }
    c.transport_coarse_smoothness_weight = if pyramid { 0.01 } else { 0. };
    let (model, _) = initialize::<B>(&c, &Default::default()).unwrap();
    let model = model.train_encoder(3);
    let mask = visible_mask(4, 0.5, 17, 0).unwrap();
    let target = rgb(0.);
    let refs = vec![rgb(0.1), rgb(0.2)];
    let out = model.complete(target.clone(), &refs, &mask).unwrap();
    let expected = burn_gekko_train::loss::rgb_patches(
        burn_gekko_train::encoder::normalize(out.rgb, &model.encoder_config),
        16,
        false,
    );
    assert_eq!(
        scalar((expected - out.cross_prediction).abs().max()).unwrap(),
        0.
    );
    assert!(out.transport.is_some());
    let before = model
        .decoder
        .appearance_head
        .as_ref()
        .unwrap()
        .flow
        .weight
        .val()
        .detach();
    let matcher_before = model
        .decoder
        .appearance_head
        .as_ref()
        .and_then(|h| h.matcher.as_ref())
        .map(|m| m.first.weight.val().detach());
    let loss = model.training_loss(target, &refs, &mask, 1., 0.).unwrap().0;
    assert!(scalar(loss.clone()).unwrap().is_finite());
    let mut grads = GradientsParams::from_grads(loss.backward(), &model);
    assert!(!take_gradients(&model.encoder, &mut grads).is_empty());
    let mut optimizer = AdamWConfig::new().init::<B, burn_gekko_train::model::GekkoDecoder<B>>();
    let decoder = optimizer.step(0.0001, model.decoder, grads);
    if let Some(before) = matcher_before {
        let after = decoder
            .appearance_head
            .as_ref()
            .unwrap()
            .matcher
            .as_ref()
            .unwrap()
            .first
            .weight
            .val()
            .detach();
        assert!(scalar((after - before).abs().max()).unwrap() > 0.);
    }
    assert!(
        scalar(
            (decoder.appearance_head.unwrap().flow.weight.val().detach() - before)
                .abs()
                .max()
        )
        .unwrap()
            > 0.
    );
}

#[test]
fn legacy_record_without_optional_appearance_head_loads_unchanged() {
    legacy_optional_record(false);
}
#[test]
fn legacy_appearance_record_without_matcher_loads_unchanged() {
    legacy_optional_record(true);
}
fn legacy_optional_record(appearance: bool) {
    use burn::module::Module;
    use burn::record::{FullPrecisionSettings, Record};
    let d = Default::default();
    let mut c = config();
    c.appearance_transport = appearance;
    let (model, _) = initialize::<B>(&c, &d).unwrap();
    let mask = visible_mask(4, 0.5, 17, 0).unwrap();
    let refs = vec![rgb(0.1), rgb(0.2)];
    let before = model.complete(rgb(0.), &refs, &mask).unwrap().rgb;
    let mut serialized = serde_json::to_value(
        model
            .clone()
            .into_record()
            .into_item::<FullPrecisionSettings>(),
    )
    .unwrap();
    let removed = if appearance {
        serialized["decoder"]["appearance_head"]
            .as_object_mut()
            .unwrap()
            .remove("matcher")
    } else {
        serialized["decoder"]
            .as_object_mut()
            .unwrap()
            .remove("appearance_head")
    };
    assert!(removed.is_some());
    let item = serde_json::from_value(serialized).unwrap();
    let record = <burn_gekko_train::e2e::ReconstructionModel<B> as Module<B>>::Record::from_item::<
        FullPrecisionSettings,
    >(item, &d);
    let loaded = model.load_record(record);
    if appearance {
        assert!(
            loaded
                .decoder
                .appearance_head
                .as_ref()
                .unwrap()
                .matcher
                .is_none()
        );
    } else {
        assert!(loaded.decoder.appearance_head.is_none());
    }
    let after = loaded.complete(rgb(0.), &refs, &mask).unwrap().rgb;
    assert_eq!(scalar((before - after).abs().max()).unwrap(), 0.);
}
#[test]
fn unreviewed_pretrained_identity_is_rejected() {
    let mut c = config();
    c.initialization = Initialization::Vjepa21 {
        directory: ".data/released-gekko".into(),
        expected_encoder_id: "arbitrary".into(),
    };
    assert!(c.validate().unwrap_err().to_string().contains("unreviewed"));
}

#[test]
fn batched_completion_matches_independent_target_reference_groups() {
    let (model, _) = initialize::<B>(&config(), &Default::default()).unwrap();
    let mask = visible_mask(4, 0.5, 17, 0).unwrap();
    let targets = [rgb(0.), rgb(1.3)];
    let references = [[rgb(0.1), rgb(0.2)], [rgb(0.7), rgb(0.8)]];
    let batched = model
        .complete(
            Tensor::cat(targets.to_vec(), 0),
            &(0..2)
                .map(|v| Tensor::cat(vec![references[0][v].clone(), references[1][v].clone()], 0))
                .collect::<Vec<_>>(),
            &mask,
        )
        .unwrap();
    let individual: Vec<_> = (0..2)
        .map(|i| {
            model
                .complete(targets[i].clone(), &references[i], &mask)
                .unwrap()
        })
        .collect();
    let expected = Tensor::cat(individual.iter().map(|x| x.rgb.clone()).collect(), 0);
    let expected_mono = Tensor::cat(individual.iter().map(|x| x.monocular.clone()).collect(), 0);
    assert!(scalar((batched.rgb - expected).abs().max()).unwrap() < 1e-5);
    assert!(scalar((batched.monocular - expected_mono).abs().max()).unwrap() < 1e-5);
}
