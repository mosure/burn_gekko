//! RGB-only synthetic view curriculum. A training image and independently
//! sampled projective views form a copying task before full 3D room matching.
//! These are training augmentations, never hidden-target inputs at inference.
use crate::transport::{pixel_grid, sample_pixels};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

/// Output-to-source projective coefficients in coordinates centered on the image.
/// Recording them permits replay without relying on a mutable global RNG.
#[derive(Clone, Debug, serde::Serialize)]
pub struct Warp {
    pub coefficients: [f32; 8],
}
fn rng(seed: u64, step: usize) -> ChaCha8Rng {
    ChaCha8Rng::seed_from_u64(
        seed ^ (step as u64).wrapping_mul(0x9e3779b97f4a7c15) ^ 0xd18fcb20726a0e39,
    )
}
pub fn use_synthetic(seed: u64, step: usize, probability: f64) -> bool {
    rng(seed, step).random::<f64>() < probability
}

pub fn warp<B: Backend>(rgb: Tensor<B, 4>, parameters: &[Warp]) -> Tensor<B, 4> {
    let [b, _, h, w] = rgb.dims();
    assert_eq!(parameters.len(), b);
    let d = rgb.device();
    let (x, y) = pixel_grid(b, h, w, &d);
    let x = x - (w - 1) as f32 / 2.;
    let y = y - (h - 1) as f32 / 2.;
    let coefficient = |i: usize| {
        Tensor::<B, 4>::from_data(
            TensorData::new(
                parameters.iter().map(|p| p.coefficients[i]).collect(),
                [b, 1, 1, 1],
            ),
            &d,
        )
    };
    let denom = coefficient(6) * x.clone() + coefficient(7) * y.clone() + 1.;
    let xx = (coefficient(0) * x.clone() + coefficient(1) * y.clone() + coefficient(4))
        / denom.clone()
        + (w - 1) as f32 / 2.;
    let yy =
        (coefficient(2) * x + coefficient(3) * y + coefficient(5)) / denom + (h - 1) as f32 / 2.;
    sample_pixels(rgb, xx, yy)
}

pub fn references<B: Backend>(
    rgb: Tensor<B, 4>,
    count: usize,
    seed: u64,
    step: usize,
) -> (Vec<Tensor<B, 4>>, Vec<Vec<Warp>>) {
    let [b, _, h, w] = rgb.dims();
    let mut random = rng(seed.wrapping_add(1), step);
    let parameters: Vec<Vec<Warp>> = (0..count)
        .map(|_| {
            (0..b)
                .map(|_| {
                    let angle: f32 = random.random_range(-0.18..0.18);
                    let scale: f32 = random.random_range(0.88..1.12);
                    let shear: f32 = random.random_range(-0.04..0.04);
                    Warp {
                        coefficients: [
                            scale * angle.cos(),
                            -scale * angle.sin() + shear,
                            scale * angle.sin(),
                            scale * angle.cos(),
                            random.random_range(-0.14..0.14) * w as f32,
                            random.random_range(-0.14..0.14) * h as f32,
                            random.random_range(-0.04..0.04) / w as f32,
                            random.random_range(-0.04..0.04) / h as f32,
                        ],
                    }
                })
                .collect()
        })
        .collect();
    (
        parameters.iter().map(|p| warp(rgb.clone(), p)).collect(),
        parameters,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    type B = burn::backend::NdArray<f32>;
    #[test]
    fn projective_rgb_augmentation_replays_and_preserves_sample_identity() {
        let d = Default::default();
        let data: Vec<f32> = (0..2 * 3 * 32 * 32).map(|i| i as f32 / 32.).collect();
        let rgb = Tensor::<B, 4>::from_data(TensorData::new(data, [2, 3, 32, 32]), &d);
        let identity = Warp {
            coefficients: [1., 0., 0., 1., 0., 0., 0., 0.],
        };
        let exact = warp(rgb.clone(), &[identity.clone(), identity]);
        assert_eq!(
            crate::tensor::scalar((exact - rgb.clone()).abs().max()).unwrap(),
            0.
        );
        let (a, pa) = references(rgb.clone(), 2, 19, 41);
        let (b, pb) = references(rgb.clone(), 2, 19, 41);
        assert_eq!(
            serde_json::to_string(&pa).unwrap(),
            serde_json::to_string(&pb).unwrap()
        );
        for (x, y) in a.iter().zip(b) {
            assert_eq!(
                crate::tensor::scalar((x.clone() - y).abs().max()).unwrap(),
                0.
            );
        }
        let independent = warp(rgb.slice_dim(0, 1..2), &pa[0][1..]);
        assert!(
            crate::tensor::scalar((a[0].clone().slice_dim(0, 1..2) - independent).abs().max())
                .unwrap()
                < 1e-4
        );
        assert!(!use_synthetic(19, 41, 0.) && use_synthetic(19, 41, 1.));
        assert_eq!(use_synthetic(19, 41, 0.5), use_synthetic(19, 41, 0.5));
    }
}
