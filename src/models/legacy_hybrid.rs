//! Pretrained geometric appearance features with a learned V-JEPA residual adapter.
//! The RGB statistics head predicts its own patch mean/scale; inference takes no
//! dense target, renderer annotations, or target-derived denormalization values.
use crate::{
    loss::{calibrated_rgb, normalize_patches, rgb_patches},
    released::ReleasedDecoder,
};
use burn::{
    module::{Module, Param},
    nn::{Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Int, Tensor, TensorData, activation, backend::Backend},
};
use burn_vjepa::SparseTokenMask;

#[derive(Module, Debug)]
pub struct HybridFusion<B: Backend> {
    pub decoder: ReleasedDecoder<B>,
    adapter_norm: LayerNorm<B>,
    adapter_up: Linear<B>,
    pub adapter_down: Linear<B>,
    statistics_up: Linear<B>,
    pub statistics_out: Linear<B>,
    #[module(skip)]
    normalize_predicted_content: bool,
}
pub struct HybridOutput<B: Backend> {
    pub rgb: Tensor<B, 4>,
    pub pair_predictions: Vec<Tensor<B, 3>>,
    pub normalized_content: Vec<Tensor<B, 3>>,
    pub auxiliary_rgb: Vec<Tensor<B, 4>>,
}
impl<B: Backend> HybridFusion<B> {
    pub fn new(
        decoder: ReleasedDecoder<B>,
        jepa_dim: usize,
        normalize_predicted_content: bool,
        d: &B::Device,
    ) -> Self {
        let mut statistics_out = LinearConfig::new(128, 2)
            .with_initializer(Initializer::Normal {
                mean: 0.,
                std: 0.001,
            })
            .init(d);
        statistics_out.bias = Some(Param::from_tensor(Tensor::from_data(
            TensorData::new(vec![-0.5f32, -1.3], [2]),
            d,
        )));
        Self {
            decoder: decoder.no_grad(),
            adapter_norm: LayerNormConfig::new(jepa_dim).init(d),
            adapter_up: LinearConfig::new(jepa_dim, 256).init(d),
            adapter_down: LinearConfig::new(256, 1024)
                .with_initializer(Initializer::Zeros)
                .init(d),
            statistics_up: LinearConfig::new(768, 128).init(d),
            statistics_out,
            normalize_predicted_content,
        }
    }
    fn adapted(&self, appearance: Tensor<B, 3>, jepa: Tensor<B, 3>, enabled: bool) -> Tensor<B, 3> {
        if !enabled {
            return appearance;
        }
        appearance
            + self.adapter_down.forward(activation::gelu(
                self.adapter_up.forward(self.adapter_norm.forward(jepa)),
            ))
    }
    fn prediction(
        &self,
        features: Tensor<B, 3>,
        content: Tensor<B, 3>,
        grid: [usize; 2],
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let content = if self.normalize_predicted_content {
            normalize_patches(content)
        } else {
            content
        };
        let statistics = self
            .statistics_out
            .forward(activation::gelu(self.statistics_up.forward(features)));
        let prediction = Tensor::cat(vec![content, statistics], 2);
        let b = prediction.dims()[0];
        let pixels = calibrated_rgb(prediction.clone(), 768)
            .reshape([b, grid[0], grid[1], 16, 16, 3])
            .permute([0, 5, 1, 3, 2, 4])
            .reshape([b, 3, grid[0] * 16, grid[1] * 16]);
        let d = pixels.device();
        let mean = Tensor::from_data(TensorData::new(vec![0.485, 0.456, 0.406], [1, 3, 1, 1]), &d);
        let std = Tensor::from_data(TensorData::new(vec![0.229, 0.224, 0.225], [1, 3, 1, 1]), &d);
        (prediction, pixels * std + mean)
    }
    /// MAE shares the calibration head and adapter, but accepts no reference.
    pub fn monocular(
        &self,
        target: Tensor<B, 3>,
        jepa_target: Tensor<B, 3>,
        mask: &SparseTokenMask,
        grid: [usize; 2],
        adapter: bool,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let target = self.adapted(target, jepa_target, adapter);
        let features = self.decoder.features(target, None, Some(mask), grid, true);
        let content = self.decoder.mae_head.forward(features.clone());
        self.prediction(features, content, grid)
    }
    /// Dense RI is a separate inference operation. Its output is never an input
    /// to masked RGB completion. Max over shared pairwise heads models a union.
    pub fn relative_improvement(
        &self,
        full_target: Tensor<B, 3>,
        jepa_target: Tensor<B, 3>,
        references: Vec<Tensor<B, 3>>,
        jepa_references: Vec<Tensor<B, 3>>,
        grid: [usize; 2],
        adapter: bool,
    ) -> Tensor<B, 3> {
        let target = self.adapted(full_target, jepa_target, adapter);
        let scores: Vec<_> = references
            .into_iter()
            .zip(jepa_references)
            .map(|(r, j)| {
                let reference = self.adapted(r, j, adapter);
                let features =
                    self.decoder
                        .features(target.clone(), Some(reference), None, grid, false);
                self.decoder
                    .cross_head
                    .forward(features)
                    .slice_dim(2, 768..1024)
            })
            .collect();
        Tensor::stack::<4>(scores, 0).max_dim(0).squeeze_dim(0)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        target: Tensor<B, 3>,
        jepa_target: Tensor<B, 3>,
        references: Vec<Tensor<B, 3>>,
        jepa_references: Vec<Tensor<B, 3>>,
        mask: &SparseTokenMask,
        grid: [usize; 2],
        adapter: bool,
    ) -> HybridOutput<B> {
        assert_eq!(references.len(), jepa_references.len());
        assert!(!references.is_empty());
        let target = self.adapted(target, jepa_target, adapter);
        let mut pair_predictions = Vec::new();
        let mut normalized_content = Vec::new();
        let mut images = Vec::new();
        for (reference, jepa) in references.into_iter().zip(jepa_references) {
            let reference = self.adapted(reference, jepa, adapter);
            let features =
                self.decoder
                    .features(target.clone(), Some(reference), Some(mask), grid, false);
            let content = self
                .decoder
                .cross_head
                .forward(features.clone())
                .slice_dim(2, 0..768);
            let (pred, rgb) = self.prediction(features, content, grid);
            images.push(rgb);
            normalized_content.push(pred.clone().slice_dim(2, 0..768));
            pair_predictions.push(pred);
        }
        let rgb = Tensor::stack::<5>(images, 0).mean_dim(0).squeeze_dim(0);
        HybridOutput {
            rgb,
            pair_predictions,
            normalized_content,
            auxiliary_rgb: Vec::new(),
        }
    }
}

pub fn hybrid_loss<B: Backend>(
    out: &HybridOutput<B>,
    rgb: Tensor<B, 4>,
    mask: &SparseTokenMask,
    edge_weight: f64,
    gradient_energy_weight: f64,
) -> Tensor<B, 1> {
    let target = rgb_patches(crate::released::normalize(rgb.clone()), 16, false);
    let [_, n, d] = target.dims();
    let hidden: Vec<i64> = (0..n)
        .filter(|i| !mask.indices().contains(i))
        .map(|i| i as i64)
        .collect();
    let k = hidden.len();
    let ids = Tensor::<B, 1, Int>::from_data(TensorData::new(hidden, [k]), &target.device());
    let mean = target.clone().mean_dim(2);
    let log_std = (((target.clone() - mean.clone()).powf_scalar(2.).sum_dim(2) / (d - 1) as f32)
        + 1e-6)
        .sqrt()
        .log();
    let normalized = normalize_patches(target);
    let mut terms = Vec::new();
    for p in &out.pair_predictions {
        let mean_loss = (p.clone().slice_dim(2, d..d + 1) - mean.clone()).powf_scalar(2.);
        let scale_loss = (p.clone().slice_dim(2, d + 1..d + 2) - log_std.clone()).powf_scalar(2.);
        let content_loss = (p.clone().slice_dim(2, 0..d) - normalized.clone())
            .powf_scalar(2.)
            .mean_dim(2);
        terms.push(
            (mean_loss + scale_loss * 0.05 + content_loss * 0.1)
                .select(1, ids.clone())
                .mean(),
        );
    }
    let rgb_loss = (rgb_patches(out.rgb.clone(), 16, false) - rgb_patches(rgb.clone(), 16, false))
        .powf_scalar(2.)
        .select(1, ids.clone())
        .mean();
    let mut total = Tensor::cat(terms, 0).mean() + rgb_loss * 10.;
    for prediction in &out.auxiliary_rgb {
        total = total
            + (rgb_patches(prediction.clone(), 16, false) - rgb_patches(rgb.clone(), 16, false))
                .powf_scalar(2.)
                .select(1, ids.clone())
                .mean()
                * 5.;
    }
    if edge_weight > 0. || gradient_energy_weight > 0. {
        let [b, _, h, w] = rgb.dims();
        let grid_width = w / 16;
        let hidden: Vec<f32> = (0..h * w)
            .map(|i| {
                if mask
                    .indices()
                    .contains(&(i / w / 16 * grid_width + i % w / 16))
                {
                    0.
                } else {
                    1.
                }
            })
            .collect();
        let hidden =
            Tensor::<B, 4>::from_data(TensorData::new(hidden, [1, 1, h, w]), &rgb.device());
        for axis in [2, 3] {
            let n = if axis == 2 { h } else { w };
            let diff =
                |x: Tensor<B, 4>| x.clone().slice_dim(axis, 1..n) - x.slice_dim(axis, 0..n - 1);
            let valid =
                hidden.clone().slice_dim(axis, 1..n) * hidden.clone().slice_dim(axis, 0..n - 1);
            let prediction = diff(out.rgb.clone());
            let target = diff(rgb.clone());
            let pixels = valid.clone().sum() * 3.;
            total = total
                + ((prediction.clone() - target.clone()).abs() * valid.clone()).sum()
                    / (pixels.clone() * b as f32 + 1e-6)
                    * edge_weight;
            if gradient_energy_weight > 0. {
                // Training-only per-image contrast constraint. Target gradient
                // energy is never computed by or passed to RGB inference.
                let energy = |x: Tensor<B, 4>| {
                    (x.powf_scalar(2.) * valid.clone())
                        .sum_dim(1)
                        .sum_dim(2)
                        .sum_dim(3)
                        .reshape([b])
                        / (pixels.clone() + 1e-6)
                };
                let ratio = ((energy(prediction) + 1e-7) / (energy(target).detach() + 1e-7)).sqrt();
                total = total + (ratio - 1.).powf_scalar(2.).mean() * gradient_energy_weight;
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};

    #[test]
    fn hybrid_rgb_and_edge_losses_ignore_observed_pixels_and_boundaries() {
        type B = Autodiff<NdArray<f32>>;
        let device = Default::default();
        let mask = SparseTokenMask::new(vec![0], 4).unwrap();
        let prediction = Tensor::<B, 4>::zeros([1, 3, 32, 32], &device).require_grad();
        let out = HybridOutput {
            rgb: prediction.clone(),
            pair_predictions: vec![Tensor::zeros([1, 4, 770], &device)],
            normalized_content: Vec::new(),
            auxiliary_rgb: Vec::new(),
        };
        let pixels: Vec<f32> = (0..3 * 32 * 32)
            .map(|i| 0.1 + ((i / 32 + i % 32) % 7) as f32 / 10.)
            .collect();
        let target =
            Tensor::<B, 4>::from_data(TensorData::new(pixels.clone(), [1, 3, 32, 32]), &device);
        let loss = hybrid_loss(&out, target.clone(), &mask, 1., 0.05);
        let base = loss.clone().into_scalar();
        let mut intervention = pixels;
        for c in 0..3 {
            for y in 0..16 {
                for x in 0..16 {
                    intervention[c * 1024 + y * 32 + x] = 100.;
                }
            }
        }
        let changed =
            Tensor::<B, 4>::from_data(TensorData::new(intervention, [1, 3, 32, 32]), &device);
        assert!((hybrid_loss(&out, changed, &mask, 1., 0.05).into_scalar() - base).abs() < 1e-6);
        assert!(base > hybrid_loss(&out, target, &mask, 0., 0.).into_scalar());
        let grad = prediction
            .grad(&loss.backward())
            .unwrap()
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let mut hidden_signal = 0.;
        for (i, g) in grad.into_iter().enumerate() {
            if (i % 1024) / 32 < 16 && i % 32 < 16 {
                assert_eq!(g, 0.);
            } else {
                hidden_signal += g.abs();
            }
        }
        assert!(hidden_signal > 0.);
    }
}
