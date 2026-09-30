//! Multi-view prediction in a fixed V-JEPA representation space.
//! Teacher features enter only the loss. No RGB head or appearance transport is used.
use crate::{
    encoder::normalize,
    heads::spatial::{SpatialDescriptor, SpatialDescriptorConfig},
    model::{DecoderConfig, DecoderPosition, GekkoDecoder},
};
use anyhow::{Result, ensure};
use burn::{
    module::Module,
    nn::{Linear, LinearConfig},
    tensor::{Int, Tensor, TensorData, backend::Backend},
};
use burn_vjepa::{SparseTokenMask, VJepaConfig, VJepaEncoder};

#[derive(Module, Debug)]
pub struct LatentFusion<B: Backend> {
    pub decoder: GekkoDecoder<B>,
    /// Exactly the same head and objective are used for both information sets.
    pub prediction: Linear<B>,
    pub improvement: Linear<B>,
    /// Optional dense spatial head. Old named records deserialize this as None.
    pub spatial_descriptor: Option<SpatialDescriptor<B>>,
}

#[derive(Module, Debug)]
pub struct LatentModel<B: Backend> {
    pub encoder: VJepaEncoder<B>,
    pub fusion: LatentFusion<B>,
    #[module(skip)]
    pub encoder_config: VJepaConfig,
    /// Optional one-based trained feature level appended to final tokens.
    /// Set from the checksummed run config after loading the parameter record.
    #[module(skip)]
    spatial_input_layer: Option<usize>,
    #[module(skip)]
    spatial_input_scale: f64,
}

pub struct LatentPrediction<B: Backend> {
    pub cross: Tensor<B, 3>,
    pub monocular: Tensor<B, 3>,
}

impl<B: Backend> LatentModel<B> {
    pub fn new(
        encoder: VJepaEncoder<B>,
        encoder_config: VJepaConfig,
        config: &DecoderConfig,
        device: &B::Device,
    ) -> Result<Self> {
        let mut decoder = GekkoDecoder::with_position(config, DecoderPosition::Rope2d, device)?
            .with_attention_normalization(true, device)
            .with_mae_context_before_self(true);
        // Retain record compatibility with the RGB decoder, but do not train its unused heads.
        decoder.cross_rgb = decoder.cross_rgb.no_grad();
        decoder.mae_rgb = decoder.mae_rgb.no_grad();
        decoder.ri = decoder.ri.no_grad();
        let fusion = LatentFusion {
            decoder,
            prediction: LinearConfig::new(config.width, encoder_config.encoder.embed_dim)
                .init(device),
            improvement: LinearConfig::new(config.width, 1).init(device),
            spatial_descriptor: None,
        };
        let model = Self {
            encoder,
            fusion,
            encoder_config,
            spatial_input_layer: None,
            spatial_input_scale: 1.0,
        };
        Ok(model.clone().load_record(model.into_record()))
    }

    pub fn train_encoder(mut self, blocks: usize) -> Self {
        self.encoder = self.encoder.no_grad();
        if blocks > 0 {
            self.encoder = self.encoder.with_last_blocks_require_grad(blocks, true);
        }
        if blocks > self.encoder.blocks.len() {
            self.encoder.image_patch_embed = self.encoder.image_patch_embed.with_require_grad(true);
            self.encoder.image_mod_embed = self.encoder.image_mod_embed.set_require_grad(true);
        }
        self
    }

    /// Construct the expected record shape before loading checkpoint parameters.
    pub fn prepare_spatial_descriptor(
        mut self,
        config: Option<&SpatialDescriptorConfig>,
    ) -> Result<Self> {
        if let Some(c) = config {
            c.validate()?;
            self.fusion.spatial_descriptor = Some(SpatialDescriptor::new(
                self.fusion.prediction.weight.dims()[0],
                self.encoder_config.encoder.embed_dim,
                c.residual_radius,
                &self.fusion.prediction.weight.device(),
            ));
        } else {
            self.fusion.spatial_descriptor = None;
        }
        Ok(self)
    }

    /// Optional Burn modules otherwise silently discard records when initialized as None.
    pub fn load_checked_record(self, record: LatentModelRecord<B>) -> Result<Self> {
        ensure!(
            self.fusion.spatial_descriptor.is_some() == record.fusion.spatial_descriptor.is_some(),
            "spatial head record/config mismatch"
        );
        Ok(self.load_record(record))
    }

    pub fn validate_spatial_descriptor(&self) -> Result<()> {
        ensure!(
            self.fusion.spatial_descriptor.is_none()
                || (self.spatial_input_layer.is_some() && self.spatial_input_scale == 1.0),
            "spatial descriptor requires an unscaled hierarchical input route"
        );
        Ok(())
    }

    /// A matching-only readout; never substitute its dense inputs into completion.
    pub fn spatial_descriptor(
        &self,
        encoded: Tensor<B, 3>,
        fused: Tensor<B, 3>,
    ) -> Option<Tensor<B, 3>> {
        let base = self.encoder_config.encoder.embed_dim;
        self.fusion.spatial_descriptor.as_ref().map(|head| {
            assert_eq!(encoded.dims()[2], 2 * base);
            head.forward(encoded.slice_dim(2, base..2 * base), fused)
        })
    }

    /// Configure a direct spatial path without adding hidden RGB to the input.
    /// Only a new weights-only phase may extend the decoder projection with
    /// zeros; loaded checkpoints must already contain the declared dimensions.
    pub fn with_spatial_input_layer(
        mut self,
        layer: Option<usize>,
        allow_extension: bool,
    ) -> Result<Self> {
        let base = self.encoder_config.encoder.embed_dim;
        if let Some(layer) = layer {
            ensure!(
                layer > 0
                    && self
                        .encoder_config
                        .encoder
                        .hierarchical_layers()
                        .contains(&(layer - 1)),
                "spatial input requires a trained one-based encoder level"
            );
            if allow_extension && self.fusion.decoder.encoder_input_width() == base {
                self.fusion.decoder = self.fusion.decoder.extend_encoder_input_zero(base);
            }
        }
        ensure!(
            self.fusion.decoder.encoder_input_width() == base * (1 + usize::from(layer.is_some())),
            "checkpoint projection disagrees with spatial input configuration"
        );
        self.spatial_input_layer = layer;
        Ok(self)
    }

    /// A zero scale is a matched-layout control: identical projection shape,
    /// with the intermediate feature route disabled. It is never fitted on a test set.
    pub fn with_spatial_input_scale(mut self, scale: f64) -> Result<Self> {
        ensure!(
            (0.0..=1.0).contains(&scale) && (self.spatial_input_layer.is_some() || scale == 1.0),
            "invalid spatial input scale or missing feature level"
        );
        self.spatial_input_scale = scale;
        Ok(self)
    }

    /// Keep the historical final-layer encoder control separate from any new
    /// intermediate input route used by fusion.
    pub fn final_encoder_features(&self, features: Tensor<B, 3>) -> Tensor<B, 3> {
        let base = self.encoder_config.encoder.embed_dim;
        assert_eq!(
            features.dims()[2],
            base * (1 + usize::from(self.spatial_input_layer.is_some()))
        );
        if self.spatial_input_layer.is_some() {
            features.slice_dim(2, 0..base)
        } else {
            features
        }
    }

    pub fn encode(&self, rgb: Tensor<B, 4>, mask: Option<&SparseTokenMask>) -> Tensor<B, 3> {
        // Sparse extraction precedes every encoder attention block.
        let levels: Vec<_> = self
            .spatial_input_layer
            .map(|x| x - 1)
            .into_iter()
            .collect();
        let mut output = self.encoder.forward_image_capture_layers(
            normalize(rgb, &self.encoder_config),
            mask,
            &levels,
        );
        if self.spatial_input_layer.is_some() {
            assert_eq!(output.hierarchical.len(), 1);
            Tensor::cat(
                vec![
                    output.tokens,
                    output.hierarchical.remove(0) * self.spatial_input_scale,
                ],
                2,
            )
        } else {
            output.tokens
        }
    }

    pub fn encode_references(&self, rgb: &[Tensor<B, 4>]) -> Vec<Tensor<B, 3>> {
        let b = rgb[0].dims()[0];
        let encoded = self.encode(Tensor::cat(rgb.to_vec(), 0), None);
        (0..rgb.len())
            .map(|i| encoded.clone().slice_dim(0, i * b..(i + 1) * b))
            .collect()
    }

    pub fn predict_encoded(
        &self,
        masked: Tensor<B, 3>,
        references: Vec<Tensor<B, 3>>,
        mask: &SparseTokenMask,
        grid: [usize; 2],
    ) -> Result<LatentPrediction<B>> {
        let (cross, monocular) = self
            .fusion
            .decoder
            .reconstruction_features(masked, references, mask, grid)?;
        Ok(LatentPrediction {
            cross: self.fusion.prediction.forward(cross),
            monocular: self.fusion.prediction.forward(monocular),
        })
    }

    pub fn predict(
        &self,
        target: Tensor<B, 4>,
        references: &[Tensor<B, 4>],
        mask: &SparseTokenMask,
    ) -> Result<LatentPrediction<B>> {
        let [_, _, h, w] = target.dims();
        self.predict_encoded(
            self.encode(target, Some(mask)),
            self.encode_references(references),
            mask,
            [h / 16, w / 16],
        )
    }

    pub fn predict_improvement(
        &self,
        full_target: Tensor<B, 3>,
        references: Vec<Tensor<B, 3>>,
        grid: [usize; 2],
    ) -> Result<Tensor<B, 3>> {
        Ok(self
            .fusion
            .improvement
            .forward(self.fusion.decoder.relative_improvement_features(
                full_target,
                references,
                grid,
            )?))
    }
}

/// Parameter-free, per-token layer normalization of fixed teacher features.
/// Predictions remain unconstrained; normalizing them would discard magnitude error.
pub fn normalize_teacher<B: Backend>(tokens: Tensor<B, 3>) -> Tensor<B, 3> {
    let centered = tokens.clone() - tokens.mean_dim(2);
    centered.clone() / (centered.powf_scalar(2.).mean_dim(2) + 1e-6).sqrt()
}

pub fn token_mse<B: Backend>(prediction: Tensor<B, 3>, target: Tensor<B, 3>) -> Tensor<B, 3> {
    (prediction - target).powf_scalar(2.).mean_dim(2)
}

pub fn token_cosine<B: Backend>(prediction: Tensor<B, 3>, target: Tensor<B, 3>) -> Tensor<B, 3> {
    let denom = (prediction.clone().powf_scalar(2.).sum_dim(2)
        * target.clone().powf_scalar(2.).sum_dim(2))
    .sqrt()
    .clamp_min(1e-8);
    (prediction * target).sum_dim(2) / denom
}

pub fn select_tokens<B: Backend>(
    tokens: Tensor<B, 3>,
    mask: &SparseTokenMask,
    hidden: bool,
) -> Tensor<B, 3> {
    let ids: Vec<i64> = (0..mask.dense_len())
        .filter(|i| mask.indices().contains(i) != hidden)
        .map(|i| i as i64)
        .collect();
    let n = ids.len();
    let device = tokens.device();
    tokens.select(
        1,
        Tensor::<B, 1, Int>::from_data(TensorData::new(ids, [n]), &device),
    )
}

/// Self-supervised patch target, not a probability of geometric co-visibility.
pub fn relative_gain<B: Backend>(
    cross_error: Tensor<B, 3>,
    mono_error: Tensor<B, 3>,
) -> Tensor<B, 3> {
    ((mono_error.clone() - cross_error) / mono_error.clamp_min(1e-6))
        .clamp(0., 1.)
        .detach()
}

pub struct LatentLoss<B: Backend> {
    pub total: Tensor<B, 1>,
    pub cross: Tensor<B, 1>,
    pub monocular: Tensor<B, 1>,
    pub visible: Tensor<B, 1>,
    pub improvement: Tensor<B, 1>,
}

pub fn latent_loss<B: Backend>(
    prediction: LatentPrediction<B>,
    teacher: Tensor<B, 3>,
    improvement: Option<Tensor<B, 3>>,
    mask: &SparseTokenMask,
    visible_weight: f64,
    ri_weight: f64,
) -> LatentLoss<B> {
    // Defensive detach also covers callers that supply an autodiff teacher tensor.
    let target = normalize_teacher(teacher.detach());
    let cross_error = token_mse(prediction.cross, target.clone());
    let mono_error = token_mse(prediction.monocular, target);
    let cross = select_tokens(cross_error.clone(), mask, true).mean();
    let monocular = select_tokens(mono_error.clone(), mask, true).mean();
    let visible = (select_tokens(cross_error.clone(), mask, false).mean()
        + select_tokens(mono_error.clone(), mask, false).mean())
        * 0.5;
    let improvement = improvement.map_or_else(
        || Tensor::zeros([1], &cross.device()),
        |prediction| {
            select_tokens(
                (prediction - relative_gain(cross_error, mono_error)).powf_scalar(2.),
                mask,
                true,
            )
            .mean()
        },
    );
    let total = (cross.clone() + monocular.clone()) * 0.5
        + visible.clone() * visible_weight
        + improvement.clone() * ri_weight;
    LatentLoss {
        total,
        cross,
        monocular,
        visible,
        improvement,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::scalar;
    use burn::backend::{Autodiff, NdArray};
    type B = Autodiff<NdArray<f32>>;
    #[test]
    fn final_only_image_path_preserves_sparse_outputs_and_parameter_gradients() {
        let d = Default::default();
        let mut ec = VJepaConfig::tiny_for_tests();
        ec.encoder.depth = 12;
        ec.encoder.n_output_distillation = 4;
        let e = VJepaEncoder::<B>::new(&ec, &d);
        let e = e
            .clone()
            .load_record(e.into_record())
            .no_grad()
            .with_last_blocks_require_grad(1, true);
        let rgb = Tensor::<B, 4>::from_data(
            TensorData::new(
                (0..3 * 32 * 32).map(|i| (i as f32 * 0.019).sin()).collect(),
                [1, 3, 32, 32],
            ),
            &d,
        );
        let mask = SparseTokenMask::new(vec![0, 2], 4).unwrap();
        for mask in [None, Some(&mask)] {
            let mut results = Vec::new();
            for capture in [true, false] {
                // Separate autodiff leaves: a backward pass consumes its graph.
                let encoder = e.clone().no_grad().with_last_blocks_require_grad(1, true);
                let output = if capture {
                    encoder.forward_image(rgb.clone(), mask)
                } else {
                    encoder.forward_image_capture_layers(rgb.clone(), mask, &[])
                };
                assert_eq!(output.hierarchical.len(), if capture { 4 } else { 0 });
                let shape = output.tokens.dims();
                let data = crate::tensor::values(output.tokens.clone().inner()).unwrap();
                let target = Tensor::<B, 3>::from_data(
                    TensorData::new(
                        (0..shape[1] * 32)
                            .map(|i| (i as f32 * 0.11).cos())
                            .collect(),
                        shape,
                    ),
                    &d,
                );
                let gradients = (output.tokens - target).powf_scalar(2.).mean().backward();
                let grad = encoder.blocks[11]
                    .attn
                    .qkv
                    .weight
                    .val()
                    .grad(&gradients)
                    .unwrap();
                assert!(
                    encoder.norms_block[0]
                        .gamma
                        .val()
                        .grad(&gradients)
                        .is_none()
                );
                results.push((data, crate::tensor::values(grad).unwrap()));
            }
            assert_eq!(results[0], results[1]);
        }
    }

    fn tokens() -> Tensor<B, 3> {
        Tensor::from_data(
            TensorData::new(
                (0..32).map(|i| (i as f32 * 0.17).sin()).collect(),
                [1, 4, 8],
            ),
            &Default::default(),
        )
    }
    #[test]
    fn target_detach_and_hidden_only_loss_are_enforced() {
        let teacher = tokens().require_grad();
        let prediction = (tokens() * 0.3).require_grad();
        let mask = SparseTokenMask::new(vec![0, 2], 4).unwrap();
        let loss = latent_loss(
            LatentPrediction {
                cross: prediction.clone(),
                monocular: prediction.clone(),
            },
            teacher.clone(),
            None,
            &mask,
            0.,
            0.,
        );
        let grads = loss.total.backward();
        assert!(teacher.grad(&grads).is_none());
        let g = prediction.grad(&grads).unwrap();
        assert_eq!(
            scalar(select_tokens(g.clone(), &mask, false).abs().max()).unwrap(),
            0.
        );
        assert!(scalar(select_tokens(g, &mask, true).abs().max()).unwrap() > 0.);
        let normalized = normalize_teacher(teacher.detach());
        assert!(scalar(normalized.clone().mean_dim(2).abs().max()).unwrap() < 1e-6);
        assert!(scalar((normalized.powf_scalar(2.).mean_dim(2) - 1.).abs().max()).unwrap() < 1e-3);
    }
    #[test]
    fn improvement_target_has_no_gradient_into_completion() {
        let cross = Tensor::<B, 3>::full([1, 4, 1], 0.25, &Default::default()).require_grad();
        let mono = Tensor::<B, 3>::full([1, 4, 1], 0.5, &Default::default()).require_grad();
        let ri = Tensor::<B, 3>::zeros([1, 4, 1], &Default::default()).require_grad();
        let target = relative_gain(cross.clone(), mono.clone());
        assert_eq!(scalar(target.clone().mean()).unwrap(), 0.5);
        let grads = (ri.clone() - target).powf_scalar(2.).mean().backward();
        assert!(cross.grad(&grads).is_none());
        assert!(mono.grad(&grads).is_none());
        assert!(ri.grad(&grads).is_some());
    }
    #[test]
    fn sparse_latent_prediction_has_no_hidden_rgb_dependency_and_is_a_reference_set() {
        let d = Default::default();
        let ec = VJepaConfig::tiny_for_tests();
        let model = LatentModel::<B>::new(
            VJepaEncoder::new(&ec, &d),
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
        .unwrap()
        .train_encoder(0);
        let rgb = Tensor::<B, 4>::from_data(
            TensorData::new(
                (0..3 * 32 * 32)
                    .map(|i| (i as f32 * 0.01).sin() * 0.4 + 0.5)
                    .collect(),
                [1, 3, 32, 32],
            ),
            &d,
        )
        .require_grad();
        let refs = vec![rgb.clone().detach() * 0.8, rgb.clone().detach() * 0.6 + 0.2];
        let mask = SparseTokenMask::new(vec![0, 2], 4).unwrap();
        let original = model.predict(rgb.clone(), &refs, &mask).unwrap();
        let visible = crate::matching::visibility(32, 32, &mask, &d);
        let changed = rgb.clone() * visible.clone() + (visible.clone().neg() + 1.) * 0.981;
        let altered = model.predict(changed, &refs, &mask).unwrap();
        assert_eq!(
            scalar((original.cross.clone() - altered.cross).abs().max()).unwrap(),
            0.
        );
        let reversed = model
            .predict(rgb.clone(), &[refs[1].clone(), refs[0].clone()], &mask)
            .unwrap();
        assert!(scalar((original.cross.clone() - reversed.cross).abs().max()).unwrap() < 1e-5);
        assert_eq!(
            scalar((original.monocular - reversed.monocular).abs().max()).unwrap(),
            0.
        );
        let grad = rgb
            .grad(&original.cross.powf_scalar(2.).mean().backward())
            .unwrap();
        let hidden = crate::matching::visibility(32, 32, &mask, &Default::default()).neg() + 1.;
        assert_eq!(scalar((grad * hidden).abs().max()).unwrap(), 0.);
    }
}
