//! A small independent RGB decoder for the predicted V-JEPA latent space.
use burn::{
    module::Module,
    nn::{Initializer, Linear, LinearConfig},
    tensor::{Tensor, activation, backend::Backend},
};

#[derive(Module, Debug)]
pub struct RgbReconstructionHead<B: Backend> {
    pub hidden: Linear<B>,
    pub output: Linear<B>,
}
impl<B: Backend> RgbReconstructionHead<B> {
    pub fn new(features: usize, width: usize, device: &B::Device) -> Self {
        assert!(features > 0 && width > 0);
        Self {
            hidden: LinearConfig::new(features, width).init(device),
            output: LinearConfig::new(width, 16 * 16 * 3)
                .with_initializer(Initializer::Zeros)
                .init(device),
        }
    }

    /// Pixel-major, sRGB `[0,1]` patches. No teacher or image statistics enter inference.
    pub fn forward(&self, predicted_latent: Tensor<B, 3>) -> Tensor<B, 3> {
        activation::sigmoid(
            self.output
                .forward(activation::gelu(self.hidden.forward(predicted_latent))),
        )
    }
}
