//! Shared appearance features and explicit RGB-only correspondence refinement.
//! Queries contain observed target RGB and the model's own completion, never
//! hidden target pixels. No camera, depth, flow labels or pretrained weights.
use crate::transport::{pixel_grid, resize};
use burn::{
    module::Module,
    nn::{
        PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Tensor, TensorData, activation, backend::Backend, module::avg_pool2d},
};
use burn_vjepa::SparseTokenMask;

#[derive(Module, Debug)]
pub struct RgbMatcher<B: Backend> {
    pub first: Conv2d<B>,
    second: Conv2d<B>,
    last: Conv2d<B>,
}

fn pool<B: Backend>(x: Tensor<B, 4>, factor: usize) -> Tensor<B, 4> {
    avg_pool2d(x, [factor, factor], [factor, factor], [0, 0], false, false)
}

pub fn visibility<B: Backend>(
    h: usize,
    w: usize,
    mask: &SparseTokenMask,
    device: &B::Device,
) -> Tensor<B, 4> {
    assert_eq!(mask.dense_len(), h / 16 * (w / 16));
    let pixels = (0..h * w)
        .map(|i| {
            f32::from(
                mask.indices()
                    .binary_search(&(i / w / 16 * (w / 16) + i % w / 16))
                    .is_ok(),
            )
        })
        .collect();
    Tensor::from_data(TensorData::new(pixels, [1, 1, h, w]), device)
}

/// Sanitize before handing target appearance to any matching operation.
pub fn observed_rgb<B: Backend>(target: Tensor<B, 4>, mask: &SparseTokenMask) -> Tensor<B, 4> {
    let [_, _, h, w] = target.dims();
    let v = visibility(h, w, mask, &target.device());
    target * v
}

impl<B: Backend> RgbMatcher<B> {
    pub fn new(device: &B::Device) -> Self {
        let conv = |input, output| {
            Conv2dConfig::new([input, output], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .init(device)
        };
        Self {
            first: conv(3, 16),
            second: conv(16, 32),
            last: conv(32, 64),
        }
    }

    /// Average-pooling centers agree with the sampler's half-pixel convention.
    pub fn encode(&self, rgb: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = activation::gelu(self.first.forward(pool(rgb * 2. - 1., 2)));
        let x = activation::gelu(self.second.forward(pool(x, 2)));
        let x = self.last.forward(pool(x, 2));
        let norm = (x.clone().powf_scalar(2.).sum_dim(1) + 1e-8).sqrt();
        x / norm
    }
}

/// Match every coarse query to candidate reference features. The existing flow
/// supplies a broad 32-pixel prior, not a fixed correspondence. Keep its finer
/// variations when replacing the coarse component with the matching estimate.
pub fn refine_flow<B: Backend>(
    query: Tensor<B, 4>,
    reference: Tensor<B, 4>,
    base: Tensor<B, 4>,
    maximum: f32,
) -> Tensor<B, 4> {
    let [b, c, h, w] = query.dims();
    assert_eq!(query.dims(), reference.dims());
    assert_eq!(base.dims(), [b, 2, h * 8, w * 8]);
    let n = h * w;
    let coarse = pool(base.clone(), 8);
    let (x, y) = pixel_grid(1, h, w, &query.device());
    let coordinates = Tensor::cat(vec![x, y], 1)
        .reshape([1, 2, n])
        .swap_dims(1, 2);
    let prior = coordinates.clone() + coarse.clone().reshape([b, 2, n]).swap_dims(1, 2) / 8.;
    let mut logits = query
        .reshape([b, c, n])
        .swap_dims(1, 2)
        .matmul(reference.reshape([b, c, n]))
        / 0.05;
    for axis in 0..2 {
        let keys = coordinates
            .clone()
            .slice_dim(2, axis..axis + 1)
            .swap_dims(1, 2);
        let delta = keys.clone() - prior.clone().slice_dim(2, axis..axis + 1);
        logits = logits - delta.powf_scalar(2.) / 32.;
        let outside = (keys - coordinates.clone().slice_dim(2, axis..axis + 1))
            .abs()
            .greater_elem(maximum / 8.);
        logits = logits.mask_fill(outside, -10000.);
    }
    let weights = activation::softmax(logits, 2);
    let estimate = (weights.matmul(coordinates.clone().expand([b, n, 2])) - coordinates)
        .swap_dims(1, 2)
        .reshape([b, 2, h, w])
        * 8.;
    (base + resize(estimate - coarse, h * 8, w * 8)).clamp(-maximum, maximum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    type B = Autodiff<NdArray<f32>>;

    #[test]
    fn explicit_matching_recovers_a_known_coarse_permutation_and_has_gradients() {
        let device = Default::default();
        // Each location has a unique one-hot descriptor. Swap horizontal halves.
        let mut a = vec![0.; 16 * 16];
        let mut r = a.clone();
        for i in 0..16 {
            a[i * 16 + i] = 1.;
            r[i * 16 + (i / 4 * 4 + (i % 4 + 2) % 4)] = 1.;
        }
        let q =
            Tensor::<B, 4>::from_data(TensorData::new(a, [1, 16, 4, 4]), &device).require_grad();
        let r =
            Tensor::<B, 4>::from_data(TensorData::new(r, [1, 16, 4, 4]), &device).require_grad();
        let flow = refine_flow(q.clone(), r, Tensor::zeros([1, 2, 32, 32], &device), 32.);
        let coarse = pool(flow.clone(), 8).into_data().to_vec::<f32>().unwrap();
        // Edge cells are least affected by interpolation across the discontinuity.
        assert!(coarse[0] > 13. && coarse[3] < -13.);
        assert!(coarse[16..].iter().all(|v| v.abs() < 1e-4));
        let grad = q.grad(&flow.powf_scalar(2.).mean().backward()).unwrap();
        assert!(crate::tensor::scalar(grad.abs().sum()).unwrap() > 0.);
    }
}
