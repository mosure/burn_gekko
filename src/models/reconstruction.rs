//! Reconstruction training with fresh fusion weights and either random or MIT
//! V-JEPA 2.1 initialization. No released Gekko weights or teacher are involved.
use crate::{
    encoder::{ImageFeatures, image_features, normalize},
    hybrid::{HybridOutput, hybrid_loss},
    loss::{content_prediction, gekko_loss, normalize_patches, rgb_patches},
    model::{GekkoDecoder, Predictions},
};
use anyhow::{Result, ensure};
use burn::{
    module::{Module, ModuleVisitor, Param},
    optim::GradientsParams,
    tensor::{
        Tensor,
        backend::{AutodiffBackend, Backend},
    },
};
use burn_vjepa::{SparseTokenMask, VJepaConfig, VJepaEncoder};

#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RgbHead {
    #[default]
    Calibrated,
    /// Unconstrained ImageNet-space RGB, without per-patch prediction normalization.
    Direct,
}

#[derive(Module, Debug)]
pub struct ReconstructionModel<B: Backend> {
    pub encoder: VJepaEncoder<B>,
    pub decoder: GekkoDecoder<B>,
    #[module(skip)]
    pub encoder_config: VJepaConfig,
    #[module(skip)]
    pub appearance_bypass: bool,
    #[module(skip)]
    pub rgb_head: RgbHead,
}

pub struct Completion<B: Backend> {
    pub rgb: Tensor<B, 4>,
    pub monocular: Tensor<B, 4>,
    pub cross_prediction: Tensor<B, 3>,
    pub mae_prediction: Tensor<B, 3>,
    pub transport: Option<crate::transport::TransportOutput<B>>,
    pub appearance_mixture: Option<Tensor<B, 4>>,
    pub generated_rgb: Option<Tensor<B, 4>>,
}

impl<B: Backend> ReconstructionModel<B> {
    /// 0 = frozen, 1..depth = last blocks; depth+1 also trains image patch stem.
    /// Unused video-tokenizer parameters always remain frozen.
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
    pub fn encode(&self, rgb: Tensor<B, 4>, mask: Option<&SparseTokenMask>) -> Tensor<B, 3> {
        let tokens = self
            .encoder
            .forward_image(normalize(rgb.clone(), &self.encoder_config), mask)
            .tokens;
        image_features(
            tokens,
            rgb,
            mask,
            if self.appearance_bypass {
                ImageFeatures::SemanticRgb
            } else {
                ImageFeatures::Semantic
            },
        )
    }
    pub fn encode_references(&self, references: &[Tensor<B, 4>]) -> Vec<Tensor<B, 3>> {
        let batch = references[0].dims()[0];
        let tokens = self.encode(Tensor::cat(references.to_vec(), 0), None);
        (0..references.len())
            .map(|i| tokens.clone().slice_dim(0, i * batch..(i + 1) * batch))
            .collect()
    }
    /// Dense target RGB is not an argument to the decoder or RGB calibration.
    pub fn complete(
        &self,
        target: Tensor<B, 4>,
        references: &[Tensor<B, 4>],
        mask: &SparseTokenMask,
    ) -> Result<Completion<B>> {
        let [_, _, h, w] = target.dims();
        let observed = crate::matching::observed_rgb(target.clone(), mask);
        let masked = self.encode(target, Some(mask));
        self.complete_with_appearance(
            masked,
            self.encode_references(references),
            references,
            observed,
            mask,
            [h / 16, w / 16],
        )
    }
    pub fn complete_encoded(
        &self,
        masked: Tensor<B, 3>,
        references: Vec<Tensor<B, 3>>,
        mask: &SparseTokenMask,
        grid: [usize; 2],
    ) -> Result<Completion<B>> {
        ensure!(
            self.decoder.appearance_head.is_none(),
            "appearance transport requires reference RGB"
        );
        self.generated_completion(masked, references, mask, grid)
    }
    fn generated_completion(
        &self,
        masked: Tensor<B, 3>,
        references: Vec<Tensor<B, 3>>,
        mask: &SparseTokenMask,
        grid: [usize; 2],
    ) -> Result<Completion<B>> {
        let (cross, mae) = self.decoder.reconstruct(masked, references, mask, grid)?;
        let (cross_prediction, rgb) = decode_rgb(cross, grid, self.rgb_head);
        let (mae_prediction, monocular) = decode_rgb(mae, grid, self.rgb_head);
        Ok(Completion {
            rgb,
            monocular,
            cross_prediction,
            mae_prediction,
            transport: None,
            appearance_mixture: None,
            generated_rgb: None,
        })
    }
    fn complete_with_appearance(
        &self,
        masked: Tensor<B, 3>,
        references: Vec<Tensor<B, 3>>,
        reference_rgb: &[Tensor<B, 4>],
        observed_rgb: Tensor<B, 4>,
        mask: &SparseTokenMask,
        grid: [usize; 2],
    ) -> Result<Completion<B>> {
        let mut out = self.generated_completion(masked.clone(), references.clone(), mask, grid)?;
        if let Some(head) = &self.decoder.appearance_head {
            let features = references
                .into_iter()
                .map(|reference| {
                    self.decoder
                        .completion_features(masked.clone(), reference, mask, grid)
                })
                .collect::<Result<Vec<_>>>()?;
            out.generated_rgb = Some(out.rgb.clone());
            let visible =
                crate::matching::visibility(grid[0] * 16, grid[1] * 16, mask, &out.rgb.device());
            let query_rgb = observed_rgb + out.rgb.clone() * (visible.neg() + 1.);
            let appearance = head.forward(features, reference_rgb, out.rgb, query_rgb, grid);
            out.rgb = appearance.rgb;
            // RI must compare the actual blended reconstruction with MAE.
            out.cross_prediction =
                rgb_patches(normalize(out.rgb.clone(), &self.encoder_config), 16, false);
            out.transport = Some(appearance.transport);
            out.appearance_mixture = Some(appearance.mixture);
        }
        Ok(out)
    }
    pub fn training_loss(
        &self,
        target: Tensor<B, 4>,
        references: &[Tensor<B, 4>],
        mask: &SparseTokenMask,
        edge_weight: f64,
        energy_weight: f64,
    ) -> Result<(Tensor<B, 1>, Tensor<B, 1>)> {
        self.training_loss_with_ri(target, references, mask, edge_weight, energy_weight, 0.1)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn training_loss_with_ri(
        &self,
        target: Tensor<B, 4>,
        references: &[Tensor<B, 4>],
        mask: &SparseTokenMask,
        edge_weight: f64,
        energy_weight: f64,
        ri_weight: f64,
    ) -> Result<(Tensor<B, 1>, Tensor<B, 1>)> {
        let [_, _, h, w] = target.dims();
        let refs = self.encode_references(references);
        let out = self.complete_with_appearance(
            self.encode(target.clone(), Some(mask)),
            refs.clone(),
            references,
            crate::matching::observed_rgb(target.clone(), mask),
            mask,
            [h / 16, w / 16],
        )?;
        let ri_loss = if ri_weight > 0. {
            let ri = self.decoder.relative_improvement(
                self.encode(target.clone(), None),
                refs,
                [h / 16, w / 16],
            )?;
            // The RI target and error weights are detached inside gekko_loss.
            let raw_target =
                rgb_patches(normalize(target.clone(), &self.encoder_config), 16, false);
            gekko_loss(
                Predictions {
                    cross_rgb: content_prediction(out.cross_prediction.clone(), 768),
                    mae_rgb: content_prediction(out.mae_prediction.clone(), 768),
                    ri,
                },
                if self.rgb_head == RgbHead::Calibrated {
                    normalize_patches(raw_target)
                } else {
                    raw_target
                },
                mask,
            )
            .ri
        } else {
            Tensor::zeros([1], &target.device())
        };
        if self.rgb_head == RgbHead::Direct {
            let auxiliary = out
                .transport
                .as_ref()
                .map(|transport| {
                    let head = self.decoder.appearance_head.as_ref().unwrap();
                    let mut loss = if head.pyramid_loss {
                        crate::transport::transport_pyramid_loss(
                            transport,
                            target.clone(),
                            references,
                        )
                    } else {
                        crate::transport::transport_loss(transport, target.clone())
                    };
                    if head.coarse_smoothness_weight > 0. {
                        loss = loss
                            + crate::transport::coarse_flow_smoothness(
                                &transport.coarse_flows,
                                target.clone(),
                            ) * head.coarse_smoothness_weight;
                    }
                    loss * head.auxiliary_weight
                })
                .unwrap_or_else(|| Tensor::zeros([1], &target.device()));
            // The primary hidden-pixel objectives match across branches. Optional
            // transport also receives the explicitly configured photometric auxiliary.
            let cross = direct_rgb_loss(out.rgb, target.clone(), mask, edge_weight);
            let mono = direct_rgb_loss(out.monocular, target, mask, edge_weight);
            return Ok((
                cross + mono + auxiliary + ri_loss.clone() * ri_weight,
                ri_loss,
            ));
        }
        // Reuse the mathematical RGB/statistics/edge objective only. HybridOutput
        // contains tensors; it has no parameters or released-model dependency.
        let rgb_loss = hybrid_loss(
            &HybridOutput {
                rgb: out.rgb,
                normalized_content: Vec::new(),
                pair_predictions: vec![out.cross_prediction, out.mae_prediction],
                auxiliary_rgb: vec![out.monocular],
            },
            target,
            mask,
            edge_weight,
            energy_weight,
        );
        Ok((rgb_loss + ri_loss.clone() * ri_weight, ri_loss))
    }
}

fn decode_rgb<B: Backend>(
    prediction: Tensor<B, 3>,
    grid: [usize; 2],
    head: RgbHead,
) -> (Tensor<B, 3>, Tensor<B, 4>) {
    let b = prediction.dims()[0];
    let prediction = if head == RgbHead::Calibrated {
        Tensor::cat(
            vec![
                normalize_patches(prediction.clone().slice_dim(2, 0..768)),
                prediction.slice_dim(2, 768..770),
            ],
            2,
        )
    } else {
        prediction
    };
    let x = if head == RgbHead::Calibrated {
        crate::loss::calibrated_rgb(prediction.clone(), 768)
    } else {
        prediction.clone()
    };
    let x = x
        .reshape([b, grid[0], grid[1], 16, 16, 3])
        .permute([0, 5, 1, 3, 2, 4])
        .reshape([b, 3, grid[0] * 16, grid[1] * 16]);
    let d = x.device();
    let mean = Tensor::from_data(
        burn::tensor::TensorData::new(vec![0.485, 0.456, 0.406], [1, 3, 1, 1]),
        &d,
    );
    let std = Tensor::from_data(
        burn::tensor::TensorData::new(vec![0.229, 0.224, 0.225], [1, 3, 1, 1]),
        &d,
    );
    (prediction, x * std + mean)
}

/// Matched cross-view and monocular objective. Hidden pixels only; derivative
/// pairs contribute only when both endpoints are hidden. Multiple offsets
/// supervise aligned structure without rewarding unaligned gradient energy.
pub fn direct_rgb_loss<B: Backend>(
    prediction: Tensor<B, 4>,
    target: Tensor<B, 4>,
    mask: &SparseTokenMask,
    edge_weight: f64,
) -> Tensor<B, 1> {
    let [b, _, h, w] = target.dims();
    let hidden: Vec<f32> = (0..h * w)
        .map(|i| {
            if mask
                .indices()
                .binary_search(&(i / w / 16 * (w / 16) + i % w / 16))
                .is_ok()
            {
                0.
            } else {
                1.
            }
        })
        .collect();
    let valid = Tensor::<B, 4>::from_data(
        burn::tensor::TensorData::new(hidden, [1, 1, h, w]),
        &target.device(),
    );
    let error = prediction.clone() - target.clone();
    let denom = valid.clone().sum() * (3 * b) as f32;
    let mut loss =
        ((error.clone().powf_scalar(2.) * 10. + error.abs()) * valid.clone()).sum() / denom;
    if edge_weight > 0. {
        for axis in [2, 3] {
            let n = if axis == 2 { h } else { w };
            for delta in [1, 2, 4, 8] {
                let diff = |x: Tensor<B, 4>| {
                    x.clone().slice_dim(axis, delta..n) - x.slice_dim(axis, 0..n - delta)
                };
                let pair = valid.clone().slice_dim(axis, delta..n)
                    * valid.clone().slice_dim(axis, 0..n - delta);
                let denom = pair.clone().sum() * (3 * b) as f32 + 1e-6;
                loss = loss
                    + ((diff(prediction.clone()) - diff(target.clone())).abs() * pair).sum()
                        / denom
                        * (edge_weight / 4.);
            }
        }
    }
    loss
}

/// Move gradients by parameter id so AdamW receives genuine, separate learning
/// rates. Scaling gradients would not implement an Adam learning-rate ratio.
pub fn take_gradients<B: AutodiffBackend>(
    module: &impl Module<B>,
    all: &mut GradientsParams,
) -> GradientsParams {
    struct Take<'a> {
        all: &'a mut GradientsParams,
        selected: GradientsParams,
    }
    impl<B: AutodiffBackend> ModuleVisitor<B> for Take<'_> {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<B, D>>) {
            if let Some(g) = self.all.remove::<B::InnerBackend, D>(p.id) {
                self.selected.register(p.id, g);
            }
        }
    }
    let mut visitor = Take {
        all,
        selected: GradientsParams::new(),
    };
    module.visit(&mut visitor);
    visitor.selected
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct UnfreezeGate {
    pub stage: usize,
    pub entered_step: usize,
    pub initial_mse: f64,
    pub stage_start_mse: f64,
    pub best_mse: f64,
}
impl UnfreezeGate {
    pub fn new(initial_mse: f64, scratch: bool) -> Self {
        Self {
            stage: if scratch { 2 } else { 0 },
            entered_step: 0,
            initial_mse,
            stage_start_mse: initial_mse,
            best_mse: initial_mse,
        }
    }
    /// Require measured improvement and no >10% regression before a transition.
    pub fn observe(&mut self, step: usize, mse: f64, minimum_steps: usize, reduction: f64) -> bool {
        if !mse.is_finite() || self.stage == 2 {
            return false;
        }
        let stable = step >= self.entered_step + minimum_steps
            && mse <= self.stage_start_mse * (1. - reduction)
            && mse <= self.best_mse * 1.1;
        self.best_mse = self.best_mse.min(mse);
        if stable {
            self.stage += 1;
            self.entered_step = step;
            self.stage_start_mse = mse;
            self.best_mse = mse;
        }
        stable
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn unfreezing_requires_both_wait_and_measured_stability() {
        let mut g = UnfreezeGate::new(1., false);
        assert!(!g.observe(9, 0.5, 10, 0.1));
        assert!(!g.observe(10, 0.8, 10, 0.1)); // regression from best
        assert!(g.observe(11, 0.5, 10, 0.1));
        assert!(!g.observe(20, 0.3, 10, 0.1));
        assert!(!g.observe(21, f64::NAN, 10, 0.1));
        assert!(g.observe(22, 0.3, 10, 0.1));
        assert!(!g.observe(40, 0.1, 10, 0.1));
    }
}
