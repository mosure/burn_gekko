//! RGB-feature-only calibration. Supervision and SO(3) scoring live outside the head.
use burn::{
    module::Module,
    nn::{Initializer, Linear, LinearConfig},
    tensor::{Tensor, TensorData, activation, backend::Backend},
};

/// Predict two rotation columns, a signed translation direction and log focal lengths.
/// Inputs are sixteen spatially pooled tokens from each direction of a dense pair.
#[derive(Module, Debug)]
pub struct CalibrationHead<B: Backend> {
    pub projection: Linear<B>,
    pub hidden: Linear<B>,
    pub output: Linear<B>,
}
impl<B: Backend> CalibrationHead<B> {
    pub fn new(features: usize, width: usize, device: &B::Device) -> Self {
        assert!(features > 0 && width > 0);
        Self {
            projection: LinearConfig::new(features, 16).init(device),
            hidden: LinearConfig::new(32 * 16, width).init(device),
            output: LinearConfig::new(width, 11)
                .with_initializer(Initializer::Zeros)
                .init(device),
        }
    }

    /// Camera labels, intrinsics and hidden RGB are never arguments to this method.
    pub fn forward(&self, pair: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, tokens, _] = pair.dims();
        assert_eq!(tokens, 32);
        let x = activation::gelu(self.projection.forward(pair)).reshape([batch, 32 * 16]);
        let x = self
            .output
            .forward(activation::gelu(self.hidden.forward(x)));
        // Identity rotation, zero direction and normalized focal length 1 at initialization.
        let offset = Tensor::from_data(
            TensorData::new(vec![1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.], [1, 11]),
            &x.device(),
        );
        x + offset
    }
}

/// Component-balanced regression avoids acos/sqrt singularities in backpropagation.
/// SO(3), angular and focal-percentage metrics are evaluated independently.
pub fn calibration_loss<B: Backend>(
    prediction: Tensor<B, 2>,
    target: Tensor<B, 2>,
    translation_valid: Tensor<B, 2>,
) -> Tensor<B, 1> {
    assert_eq!(prediction.dims(), target.dims());
    let error = (prediction - target.detach()).powf_scalar(2.);
    let rotation = error.clone().slice_dim(1, 0..6).mean();
    let translation = (error.clone().slice_dim(1, 6..9) * translation_valid.clone()).sum()
        / (translation_valid.sum() * 3.).clamp_min(1.);
    rotation + translation + error.slice_dim(1, 9..11).mean()
}
