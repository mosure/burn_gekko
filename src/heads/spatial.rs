//! A bounded spatial correction to an independently encoded view.
//! Dense matching has its own head; masked completion does not call this module.
use anyhow::{Result, ensure};
use burn::{
    module::Module,
    nn::{Initializer, Linear, LinearConfig},
    tensor::{Tensor, backend::Backend},
};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpatialDescriptorConfig {
    /// Maximum correction norm relative to the centered encoder token norm.
    pub residual_radius: f64,
}
impl SpatialDescriptorConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (0.0..=0.5).contains(&self.residual_radius),
            "invalid spatial residual radius"
        );
        Ok(())
    }
}

#[derive(Module, Debug)]
pub struct SpatialDescriptor<B: Backend> {
    pub correction: Linear<B>,
    #[module(skip)]
    pub residual_radius: f64,
}
impl<B: Backend> SpatialDescriptor<B> {
    pub fn new(fusion_width: usize, encoder_width: usize, radius: f64, device: &B::Device) -> Self {
        Self {
            correction: LinearConfig::new(fusion_width, encoder_width)
                .with_bias(false)
                .with_initializer(Initializer::Zeros)
                .init(device),
            residual_radius: radius,
        }
    }
    /// At initialization this equals the centered encoder descriptor exactly.
    /// Each tanh component is bounded, so ||correction|| <= radius * ||base||.
    pub fn forward(&self, spatial: Tensor<B, 3>, fused: Tensor<B, 3>) -> Tensor<B, 3> {
        let base = spatial.clone() - spatial.mean_dim(1);
        let width = base.dims()[2];
        let norm = base.clone().powf_scalar(2.).sum_dim(2).sqrt().detach();
        base + self.correction.forward(fused).tanh()
            * norm
            * (self.residual_radius / (width as f64).sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::{Autodiff, NdArray},
        module::Param,
        tensor::TensorData,
    };
    type B = Autodiff<NdArray<f32>>;
    #[test]
    fn zero_initialization_preserves_baseline_and_correction_is_bounded_and_trainable() {
        let d = Default::default();
        let mut head = SpatialDescriptor::<B>::new(4, 8, 0.25, &d);
        let spatial = Tensor::<B, 3>::from_data(
            TensorData::new(
                (0..32).map(|i| (i as f32 * 0.13).sin()).collect(),
                [1, 4, 8],
            ),
            &d,
        );
        let fused = Tensor::<B, 3>::from_data(
            TensorData::new(
                (0..16).map(|i| (i as f32 * 0.19).cos()).collect(),
                [1, 4, 4],
            ),
            &d,
        )
        .require_grad();
        let base = spatial.clone() - spatial.clone().mean_dim(1);
        let output = head.forward(spatial.clone(), fused.clone());
        assert_eq!(
            crate::tensor::values(output.clone()).unwrap(),
            crate::tensor::values(base.clone()).unwrap()
        );
        let grads = output.powf_scalar(2.).mean().backward();
        assert!(
            crate::tensor::scalar(
                head.correction
                    .weight
                    .val()
                    .grad(&grads)
                    .unwrap()
                    .abs()
                    .sum()
            )
            .unwrap()
                > 0.
        );
        // A zero output projection initially isolates the old trunk from this new objective.
        assert_eq!(
            crate::tensor::scalar(fused.grad(&grads).unwrap().abs().max()).unwrap(),
            0.
        );
        head.correction.weight = Param::from_tensor(Tensor::ones([4, 8], &d) * 100.);
        let residual = head.forward(spatial, fused.detach()) - base.clone();
        let ratio =
            residual.powf_scalar(2.).sum_dim(2).sqrt() / base.powf_scalar(2.).sum_dim(2).sqrt();
        assert!(crate::tensor::scalar(ratio.max()).unwrap() <= 0.250001);
    }
}
