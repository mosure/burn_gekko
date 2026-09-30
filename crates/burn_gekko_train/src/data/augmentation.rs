//! Deterministic RGB-only correspondence batches. Absolute update indices make
//! augmentation replay independent of checkpoint boundaries and logging cadence.
use anyhow::{Result, bail, ensure};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn_gekko_data::{
    RgbScene,
    image_transform::{Homography, grid_targets, warp_rgb},
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EquivarianceConfig {
    pub weight: f64,
    pub temperature: f64,
    /// Shared-weight self-conditioned descriptors are trained as the control.
    pub self_weight: f64,
    pub max_rotation_degrees: f64,
    /// Fraction of image width/height in each direction.
    pub max_translation: f64,
    pub min_scale: f64,
    pub max_scale: f64,
    pub max_perspective: f64,
    pub brightness: f64,
    pub contrast: f64,
    pub min_valid_fraction: f64,
}
impl Default for EquivarianceConfig {
    fn default() -> Self {
        Self {
            weight: 0.1,
            temperature: 0.07,
            self_weight: 1.,
            max_rotation_degrees: 20.,
            max_translation: 0.125,
            min_scale: 0.9,
            max_scale: 1.1,
            max_perspective: 0.05,
            brightness: 0.1,
            contrast: 0.15,
            min_valid_fraction: 0.4,
        }
    }
}
impl EquivarianceConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.weight > 0.
                && self.weight <= 1.
                && (0.01..=1.).contains(&self.temperature)
                && (0. ..=1.).contains(&self.self_weight),
            "invalid equivariance loss settings"
        );
        ensure!(
            (0. ..=45.).contains(&self.max_rotation_degrees)
                && (0. ..=0.3).contains(&self.max_translation)
                && (0.75..=1.).contains(&self.min_scale)
                && (1. ..=1.33).contains(&self.max_scale)
                && (0. ..=0.1).contains(&self.max_perspective),
            "invalid homography range"
        );
        ensure!(
            (0. ..=0.5).contains(&self.brightness)
                && (0. ..=0.5).contains(&self.contrast)
                && (0.1..=1.).contains(&self.min_valid_fraction),
            "invalid appearance or overlap setting"
        );
        Ok(())
    }
}

pub struct WarpBatch<B: Backend> {
    pub rgb: Tensor<B, 4>,
    pub forward: Tensor<B, 3>,
    pub backward: Tensor<B, 3>,
    pub valid_fraction: f64,
}

/// Host-resident RGB is warped before upload. No CPU readback of GPU features.
pub fn batch<B: Backend>(
    scenes: &[RgbScene],
    samples: &[(usize, usize)],
    config: &EquivarianceConfig,
    seed: u64,
    step: usize,
    patch: usize,
    device: &B::Device,
) -> Result<WarpBatch<B>> {
    config.validate()?;
    ensure!(
        !scenes.is_empty() && !samples.is_empty(),
        "empty augmentation batch"
    );
    let (w, h) = (scenes[0].width, scenes[0].height);
    ensure!(
        patch > 0 && w.is_multiple_of(patch) && h.is_multiple_of(patch),
        "invalid augmentation grid"
    );
    let n = w / patch * (h / patch);
    let mut rng = ChaCha8Rng::seed_from_u64(
        seed ^ 0x93e8_43fb_8197_aa21 ^ (step as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
    );
    let mut pixels = Vec::with_capacity(samples.len() * w * h * 3);
    let mut forward = Vec::with_capacity(samples.len() * n * n);
    let mut backward = Vec::with_capacity(samples.len() * n * n);
    let mut valid = 0;
    for &(s, v) in samples {
        let scene = scenes
            .get(s)
            .ok_or_else(|| anyhow::anyhow!("invalid scene index"))?;
        ensure!(
            (scene.width, scene.height) == (w, h),
            "mixed augmentation sizes"
        );
        let rgb = scene
            .views
            .get(v)
            .ok_or_else(|| anyhow::anyhow!("invalid view index"))?;
        let mut accepted = None;
        for _ in 0..32 {
            let mut signed = |m: f64| rng.random_range(-1. ..=1.) * m;
            let angle = signed(config.max_rotation_degrees).to_radians();
            let tx = signed(config.max_translation) * 2.;
            let ty = signed(config.max_translation) * 2.;
            let px = signed(config.max_perspective);
            let py = signed(config.max_perspective);
            let scale = rng.random_range(config.min_scale..=config.max_scale);
            let (sin, cos) = angle.sin_cos();
            let h = Homography::from_normalized(
                [
                    [scale * cos, -scale * sin, tx],
                    [scale * sin, scale * cos, ty],
                    [px, py, 1.],
                ],
                w,
                h,
            )?;
            let f = grid_targets(h, w, scene.height, patch)?;
            let b = grid_targets(h.inverse()?, w, scene.height, patch)?;
            if (f.valid_queries.min(b.valid_queries) as f64) < config.min_valid_fraction * n as f64
            {
                continue;
            }
            accepted = Some((h, f, b));
            break;
        }
        let Some((transform, f, b)) = accepted else {
            bail!("could not sample the required overlap in 32 attempts");
        };
        let mut warped = warp_rgb(rgb, w, h, transform)?;
        let brightness = rng.random_range(-1. ..=1.) * config.brightness;
        let contrast = 1. + rng.random_range(-1. ..=1.) * config.contrast;
        for x in &mut warped {
            *x = (((*x as f64 - 0.5) * contrast + 0.5) + brightness).clamp(0., 1.) as f32;
        }
        pixels.extend(warped);
        forward.extend(f.probabilities);
        backward.extend(b.probabilities);
        valid += f.valid_queries + b.valid_queries;
    }
    let shape = [samples.len(), n, n];
    Ok(WarpBatch {
        rgb: crate::encoder::upload_rgb(pixels, [samples.len(), h, w], device),
        forward: Tensor::from_data(TensorData::new(forward, shape), device),
        backward: Tensor::from_data(TensorData::new(backward, shape), device),
        valid_fraction: valid as f64 / (2 * samples.len() * n) as f64,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn augmentation_replays_by_update_with_finite_normalized_labels() {
        let scene = RgbScene {
            seed: 11,
            width: 64,
            height: 64,
            views: vec![(0..64 * 64 * 3).map(|i| (i % 251) as f32 / 251.).collect()],
        };
        let c = EquivarianceConfig::default();
        let d = Default::default();
        let run = |step| {
            batch::<burn::backend::NdArray<f32>>(
                std::slice::from_ref(&scene),
                &[(0, 0)],
                &c,
                19,
                step,
                16,
                &d,
            )
            .unwrap()
        };
        let (a, b, next) = (run(17), run(17), run(18));
        let values =
            |x: Tensor<burn::backend::NdArray<f32>, 3>| x.into_data().to_vec::<f32>().unwrap();
        assert_eq!(values(a.forward.clone()), values(b.forward));
        assert_eq!(a.rgb.clone().into_data(), b.rgb.into_data());
        assert_ne!(a.rgb.into_data(), next.rgb.into_data());
        for labels in [a.forward, a.backward] {
            for row in values(labels).as_chunks::<16>().0 {
                let sum: f32 = row.iter().sum();
                assert!(sum == 0. || (sum - 1.).abs() < 1e-6);
                assert!(row.iter().all(|x| x.is_finite() && *x >= 0.));
            }
        }
        assert!(a.valid_fraction >= c.min_valid_fraction);
    }
}
