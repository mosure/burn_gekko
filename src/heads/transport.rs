//! RGB-only reference transport. Hidden target pixels, depth and camera poses are
//! not inputs. Each reference is processed by the same network, then fused.
use crate::model::GekkoDecoder;
use anyhow::{Result, ensure};
use burn::{
    module::Module,
    nn::{Initializer, Linear, LinearConfig},
    tensor::{Int, Tensor, TensorData, activation, backend::Backend, module::avg_pool2d},
};
use burn_vjepa::SparseTokenMask;

/// Differentiable bilinear sampling in pixel coordinates, with border padding.
/// Expressed using gathers because CUDA's interpolation backward does not
/// support bilinear mode in Burn 0.21. The floor/index path is detached.
pub fn sample_pixels<B: Backend>(
    image: Tensor<B, 4>,
    x: Tensor<B, 4>,
    y: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [b, c, h, w] = image.dims();
    let [_, _, oh, ow] = x.dims();
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let xf = x.clone().floor().detach();
    let yf = y.clone().floor().detach();
    let dx = x - xf.clone();
    let dy = y - yf.clone();
    let x0 = xf.int();
    let y0 = yf.int();
    let x1 = (x0.clone() + 1).clamp(0, w as i64 - 1);
    let y1 = (y0.clone() + 1).clamp(0, h as i64 - 1);
    let image = image.reshape([b, c, h * w]);
    let take = |xx: Tensor<B, 4, Int>, yy: Tensor<B, 4, Int>| {
        image
            .clone()
            .gather(
                2,
                (yy * w as i64 + xx)
                    .reshape([b, 1, oh * ow])
                    .expand([b, c, oh * ow]),
            )
            .reshape([b, c, oh, ow])
    };
    take(x0.clone(), y0.clone()) * (dx.clone().neg() + 1.0) * (dy.clone().neg() + 1.0)
        + take(x1.clone(), y0) * dx.clone() * (dy.clone().neg() + 1.0)
        + take(x0, y1.clone()) * (dx.clone().neg() + 1.0) * dy.clone()
        + take(x1, y1) * dx * dy
}

pub fn pixel_grid<B: Backend>(
    b: usize,
    h: usize,
    w: usize,
    device: &B::Device,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let x = Tensor::<B, 4>::from_data(
        TensorData::new((0..h * w).map(|i| (i % w) as f32).collect(), [1, 1, h, w]),
        device,
    )
    .expand([b, 1, h, w]);
    let y = Tensor::<B, 4>::from_data(
        TensorData::new((0..h * w).map(|i| (i / w) as f32).collect(), [1, 1, h, w]),
        device,
    )
    .expand([b, 1, h, w]);
    (x, y)
}

pub fn resize<B: Backend>(image: Tensor<B, 4>, h: usize, w: usize) -> Tensor<B, 4> {
    let [b, _, ih, iw] = image.dims();
    let (x, y) = pixel_grid(b, h, w, &image.device());
    sample_pixels(
        image,
        (x + 0.5) * (iw as f32 / w as f32) - 0.5,
        (y + 0.5) * (ih as f32 / h as f32) - 0.5,
    )
}

#[derive(Module, Debug)]
pub struct TransportDecoder<B: Backend> {
    pub backbone: GekkoDecoder<B>,
    pub flow_head: Linear<B>,
    #[module(skip)]
    pub max_displacement: f32,
}
pub struct TransportOutput<B: Backend> {
    pub rgb: Tensor<B, 4>,
    pub warped: Vec<Tensor<B, 4>>,
    pub flows: Vec<Tensor<B, 4>>,
    /// Optional native control field in quarter-resolution pixel units.
    pub coarse_flows: Vec<Tensor<B, 4>>,
    pub weights: Tensor<B, 4>,
}
impl<B: Backend> TransportDecoder<B> {
    pub fn new(
        backbone: GekkoDecoder<B>,
        width: usize,
        max_displacement: f32,
        device: &B::Device,
    ) -> Self {
        Self {
            backbone,
            flow_head: LinearConfig::new(width, 4 * 4 * 3)
                .with_initializer(Initializer::Normal {
                    mean: 0.0,
                    std: 0.0001,
                })
                .init(device),
            max_displacement,
        }
    }
    pub fn forward(
        &self,
        target: Tensor<B, 3>,
        references: Vec<Tensor<B, 3>>,
        rgb: Vec<Tensor<B, 4>>,
        mask: &SparseTokenMask,
        grid: [usize; 2],
    ) -> Result<TransportOutput<B>> {
        ensure!(
            !references.is_empty() && references.len() == rgb.len(),
            "transport reference mismatch"
        );
        let b = target.dims()[0];
        let [h, w] = [grid[0] * 16, grid[1] * 16];
        let (x, y) = pixel_grid(b, h, w, &target.device());
        let mut flows = Vec::new();
        let mut logits = Vec::new();
        let mut warped = Vec::new();
        for (reference, image) in references.into_iter().zip(rgb) {
            ensure!(image.dims() == [b, 3, h, w], "transport RGB shape mismatch");
            let features =
                self.backbone
                    .completion_features(target.clone(), reference, mask, grid)?;
            let coarse = self
                .flow_head
                .forward(features)
                .reshape([b, grid[0], grid[1], 4, 4, 3])
                .permute([0, 5, 1, 3, 2, 4])
                .reshape([b, 3, grid[0] * 4, grid[1] * 4]);
            let dense = resize(coarse, h, w);
            let flow = dense.clone().slice_dim(1, 0..2).tanh() * self.max_displacement;
            warped.push(sample_pixels(
                image,
                x.clone() + flow.clone().slice_dim(1, 0..1),
                y.clone() + flow.clone().slice_dim(1, 1..2),
            ));
            logits.push(dense.slice_dim(1, 2..3));
            flows.push(flow);
        }
        let weights = activation::softmax(Tensor::cat(logits, 1), 1);
        let fused: Vec<_> = warped
            .iter()
            .enumerate()
            .map(|(i, image)| image.clone() * weights.clone().slice_dim(1, i..i + 1))
            .collect();
        let rgb = Tensor::stack::<5>(fused, 0).sum_dim(0).squeeze_dim(0);
        Ok(TransportOutput {
            rgb,
            warped,
            flows,
            coarse_flows: Vec::new(),
            weights,
        })
    }
}

/// Multi-scale RGB supervision and a small flow smoothness term; RGB is a loss
/// target only. No renderer annotation or correspondence labels enter training.
pub fn transport_loss<B: Backend>(out: &TransportOutput<B>, target: Tensor<B, 4>) -> Tensor<B, 1> {
    let mut loss = transport_loss_base(out, target.clone());
    for scale in [2, 4, 8, 16] {
        let pool = |x| avg_pool2d(x, [scale, scale], [scale, scale], [0, 0], false, false);
        loss = loss
            + (pool(out.rgb.clone()) - pool(target.clone()))
                .powf_scalar(2.0)
                .mean();
    }
    loss
}

/// Warp the reference image pyramid at each scale. Pooling an already warped
/// full-resolution image does not give the sampler a coarse matching basin:
/// subpixel shifts inside a pooling cell can leave its average unchanged.
pub fn transport_pyramid_loss<B: Backend>(
    out: &TransportOutput<B>,
    target: Tensor<B, 4>,
    references: &[Tensor<B, 4>],
) -> Tensor<B, 1> {
    assert_eq!(out.flows.len(), references.len());
    let mut loss = transport_loss_base(out, target.clone());
    for scale in [2, 4, 8, 16, 32] {
        let pool = |x| avg_pool2d(x, [scale, scale], [scale, scale], [0, 0], false, false);
        loss = loss
            + (pyramid_rgb(out, references, scale) - pool(target.clone()))
                .powf_scalar(2.)
                .mean();
    }
    loss
}

fn pyramid_rgb<B: Backend>(
    out: &TransportOutput<B>,
    references: &[Tensor<B, 4>],
    scale: usize,
) -> Tensor<B, 4> {
    let pool = |x| avg_pool2d(x, [scale, scale], [scale, scale], [0, 0], false, false);
    let weights = pool(out.weights.clone());
    let [b, _, h, w] = weights.dims();
    let (x, y) = pixel_grid(b, h, w, &weights.device());
    let warped: Vec<_> = references
        .iter()
        .zip(&out.flows)
        .enumerate()
        .map(|(index, (rgb, flow))| {
            let flow = pool(flow.clone()) / scale as f32;
            sample_pixels(
                pool(rgb.clone()),
                x.clone() + flow.clone().slice_dim(1, 0..1),
                y.clone() + flow.slice_dim(1, 1..2),
            ) * weights.clone().slice_dim(1, index..index + 1)
        })
        .collect();
    Tensor::stack::<5>(warped, 0).sum_dim(0).squeeze_dim(0)
}

fn transport_loss_base<B: Backend>(out: &TransportOutput<B>, target: Tensor<B, 4>) -> Tensor<B, 1> {
    let mut loss = (out.rgb.clone() - target.clone()).powf_scalar(2.).mean();
    let [_, _, h, w] = target.dims();
    for axis in [2, 3] {
        let size = if axis == 2 { h } else { w };
        let delta =
            |x: Tensor<B, 4>| x.clone().slice_dim(axis, 1..size) - x.slice_dim(axis, 0..size - 1);
        loss = loss
            + (delta(out.rgb.clone()) - delta(target.clone()))
                .abs()
                .mean()
                * 0.05;
        for flow in &out.flows {
            loss = loss + delta(flow.clone()).powf_scalar(2.0).mean() * 0.00001;
        }
    }
    loss
}

/// Edge-aware curvature of the native flow controls. Constant and affine fields
/// have zero curvature, while image edges permit flow discontinuities. RGB is a
/// detached loss target; no geometry annotation or teacher is used.
pub fn coarse_flow_smoothness<B: Backend>(
    flows: &[Tensor<B, 4>],
    target: Tensor<B, 4>,
) -> Tensor<B, 1> {
    assert!(!flows.is_empty());
    let rgb = avg_pool2d(target.detach(), [4, 4], [4, 4], [0, 0], false, false);
    let mut terms = Vec::new();
    for flow in flows {
        let [_, _, h, w] = flow.dims();
        assert_eq!([h, w], [rgb.dims()[2], rgb.dims()[3]]);
        for axis in [2, 3] {
            let size = if axis == 2 { h } else { w };
            assert!(size >= 3);
            let left = flow.clone().slice_dim(axis, 0..size - 2);
            let centre = flow.clone().slice_dim(axis, 1..size - 1);
            let right = flow.clone().slice_dim(axis, 2..size);
            let image_left = rgb.clone().slice_dim(axis, 0..size - 2);
            let image_centre = rgb.clone().slice_dim(axis, 1..size - 1);
            let image_right = rgb.clone().slice_dim(axis, 2..size);
            let edge = ((image_centre.clone() - image_left).abs()
                + (image_right - image_centre).abs())
            .mean_dim(1);
            let weight = (edge * -10.).exp();
            terms.push(((right + left - centre * 2.).abs() * weight).mean());
        }
    }
    Tensor::stack::<2>(terms, 0).mean()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    #[test]
    fn coarse_regularizer_preserves_affine_flow_and_backpropagates_curvature() {
        type B = Autodiff<NdArray<f32>>;
        let d = Default::default();
        let make = |quadratic: bool| {
            Tensor::<B, 4>::from_data(
                TensorData::new(
                    (0..2 * 8 * 8)
                        .map(|i| {
                            let x = (i % 8) as f32;
                            let y = (i / 8 % 8) as f32;
                            if quadratic {
                                x * x + y
                            } else {
                                2. * x - 3. * y + 4.
                            }
                        })
                        .collect(),
                    [1, 2, 8, 8],
                ),
                &d,
            )
        };
        let target = Tensor::zeros([1, 3, 32, 32], &d);
        assert_eq!(
            crate::tensor::scalar(coarse_flow_smoothness(&[make(false)], target.clone())).unwrap(),
            0.
        );
        let flow = make(true).require_grad();
        let loss = coarse_flow_smoothness(std::slice::from_ref(&flow), target);
        assert!((crate::tensor::scalar(loss.clone()).unwrap() - 1.).abs() < 1e-6);
        let grads = loss.backward();
        assert!(crate::tensor::scalar(flow.grad(&grads).unwrap().abs().sum()).unwrap() > 0.);
    }

    #[test]
    fn coarse_regularizer_relaxes_at_rgb_edges_and_preserves_reference_order() {
        type B = NdArray<f32>;
        let d = Default::default();
        let flow = Tensor::<B, 4>::from_data(
            TensorData::new(
                (0..2 * 8 * 8)
                    .map(|i| if i % 8 >= 4 { 4. } else { 0. })
                    .collect(),
                [1, 2, 8, 8],
            ),
            &d,
        );
        let edge = Tensor::<B, 4>::from_data(
            TensorData::new(
                (0..3 * 32 * 32)
                    .map(|i| if i % 32 >= 16 { 1. } else { 0. })
                    .collect(),
                [1, 3, 32, 32],
            ),
            &d,
        );
        let flat = Tensor::zeros_like(&edge);
        let unweighted = crate::tensor::scalar(coarse_flow_smoothness(
            std::slice::from_ref(&flow),
            flat.clone(),
        ))
        .unwrap();
        let weighted =
            crate::tensor::scalar(coarse_flow_smoothness(std::slice::from_ref(&flow), edge))
                .unwrap();
        assert!(unweighted > 0. && weighted < unweighted * 0.001);
        let other = flow.clone() * 2.;
        let a = coarse_flow_smoothness(&[flow.clone(), other.clone()], flat.clone());
        let b = coarse_flow_smoothness(&[other, flow], flat);
        assert!(crate::tensor::scalar((a - b).abs()).unwrap() < 1e-7);
    }
    #[test]
    fn image_pyramid_supplies_displacement_gradient_when_pooled_output_is_flat() {
        type B = Autodiff<NdArray<f32>>;
        let d = Default::default();
        let stripe = |start: usize| {
            Tensor::<B, 4>::from_data(
                TensorData::new(
                    (0..64 * 64)
                        .map(|i| {
                            if (start..start + 8).contains(&(i % 64)) {
                                1.
                            } else {
                                0.
                            }
                        })
                        .collect(),
                    [1, 1, 64, 64],
                ),
                &d,
            )
        };
        let reference = stripe(38);
        let target = stripe(22);
        let example = || {
            let shift = Tensor::<B, 4>::full([1, 1, 1, 1], 0.5, &d).require_grad();
            let dx = shift.clone().expand([1, 1, 64, 64]);
            let flow = Tensor::cat(vec![dx.clone(), Tensor::zeros_like(&dx)], 1);
            let (x, y) = pixel_grid(1, 64, 64, &d);
            let rgb = sample_pixels(reference.clone(), x + dx, y);
            (
                shift,
                TransportOutput {
                    rgb: rgb.clone(),
                    warped: vec![rgb],
                    flows: vec![flow],
                    coarse_flows: Vec::new(),
                    weights: Tensor::ones([1, 1, 64, 64], &d),
                },
            )
        };
        let pool = |x| avg_pool2d(x, [16, 16], [16, 16], [0, 0], false, false);
        let (shift, out) = example();
        let old = (pool(out.rgb.clone()) - pool(target.clone()))
            .powf_scalar(2.)
            .mean();
        let old_grads = old.backward();
        let old_gradient = crate::tensor::scalar(shift.grad(&old_grads).unwrap().mean()).unwrap();
        let (shift, out) = example();
        let new = (pyramid_rgb(&out, &[reference], 16) - pool(target))
            .powf_scalar(2.)
            .mean();
        let new_grads = new.backward();
        let new_gradient = crate::tensor::scalar(shift.grad(&new_grads).unwrap().mean()).unwrap();
        assert!(old_gradient.abs() < 1e-8, "old gradient {old_gradient}");
        assert!(
            new_gradient < -0.001,
            "new gradient must move toward +16px: {new_gradient}"
        );
    }
    #[test]
    fn sampler_matches_affine_ramp_and_coordinate_derivatives() {
        type B = Autodiff<NdArray<f32>>;
        let d = Default::default();
        let image = Tensor::<B, 4>::from_data(
            TensorData::new(
                (0..16).map(|i| (i % 4 + 2 * (i / 4)) as f32).collect(),
                [1, 1, 4, 4],
            ),
            &d,
        );
        let x = Tensor::<B, 4>::full([1, 1, 1, 1], 1.25, &d).require_grad();
        let y = Tensor::<B, 4>::full([1, 1, 1, 1], 1.5, &d).require_grad();
        let sampled = sample_pixels(image, x.clone(), y.clone());
        assert!((crate::tensor::scalar(sampled.clone().mean()).unwrap() - 4.25).abs() < 1e-6);
        let grads = sampled.sum().backward();
        assert!(
            (crate::tensor::scalar(x.grad(&grads).unwrap().mean()).unwrap() - 1.0).abs() < 1e-6
        );
        assert!(
            (crate::tensor::scalar(y.grad(&grads).unwrap().mean()).unwrap() - 2.0).abs() < 1e-6
        );
    }
    #[test]
    fn zero_displacement_is_identity_including_borders() {
        type B = NdArray<f32>;
        let d = Default::default();
        let image = Tensor::<B, 4>::from_data(
            TensorData::new((0..48).map(|i| i as f32).collect(), [1, 3, 4, 4]),
            &d,
        );
        let (x, y) = pixel_grid(1, 4, 4, &d);
        assert_eq!(
            crate::tensor::scalar((sample_pixels(image.clone(), x, y) - image).abs().max())
                .unwrap(),
            0.0
        );
    }
    #[test]
    fn transport_is_reference_permutation_invariant() {
        type B = NdArray<f32>;
        let d = Default::default();
        let backbone = GekkoDecoder::<B>::new(
            &crate::model::DecoderConfig {
                encoder_dim: 8,
                width: 16,
                depth: 1,
                heads: 4,
                patch: 16,
            },
            &d,
        )
        .unwrap();
        let model = TransportDecoder::new(backbone, 16, 16., &d);
        let model = model.clone().load_record(model.into_record());
        let mask = SparseTokenMask::new(vec![0], 4).unwrap();
        let target = Tensor::zeros([1, 1, 8], &d);
        let r0 = Tensor::zeros([1, 4, 8], &d);
        let r1 = Tensor::ones([1, 4, 8], &d);
        let i0 = Tensor::zeros([1, 3, 32, 32], &d);
        let i1 = Tensor::ones([1, 3, 32, 32], &d);
        let a = model
            .forward(
                target.clone(),
                vec![r0.clone(), r1.clone()],
                vec![i0.clone(), i1.clone()],
                &mask,
                [2, 2],
            )
            .unwrap();
        let b = model
            .forward(target, vec![r1, r0], vec![i1, i0], &mask, [2, 2])
            .unwrap();
        assert!(crate::tensor::scalar((a.rgb - b.rgb).abs().max()).unwrap() < 1e-6);
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_sampler_gradient_and_resize_backward_are_correct() {
        type B = Autodiff<burn::backend::Cuda<f32, i32>>;
        let d = Default::default();
        let image = Tensor::<B, 4>::from_data(
            TensorData::new(
                (0..16).map(|i| (i % 4 + 2 * (i / 4)) as f32).collect(),
                [1, 1, 4, 4],
            ),
            &d,
        );
        let x = Tensor::<B, 4>::full([1, 1, 1, 1], 1.25, &d).require_grad();
        let y = Tensor::<B, 4>::full([1, 1, 1, 1], 1.5, &d).require_grad();
        let sampled = sample_pixels(image, x.clone(), y.clone());
        assert!((crate::tensor::scalar(sampled.clone().mean()).unwrap() - 4.25).abs() < 1e-5);
        let grads = sampled.sum().backward();
        assert!((crate::tensor::scalar(x.grad(&grads).unwrap().mean()).unwrap() - 1.).abs() < 1e-5);
        assert!((crate::tensor::scalar(y.grad(&grads).unwrap().mean()).unwrap() - 2.).abs() < 1e-5);
        let coarse = Tensor::<B, 4>::ones([1, 1, 4, 4], &d).require_grad();
        let grads = resize(coarse.clone(), 16, 16).mean().backward();
        assert!(
            (crate::tensor::scalar(coarse.grad(&grads).unwrap().sum()).unwrap() - 1.).abs() < 1e-5
        );
    }
}
