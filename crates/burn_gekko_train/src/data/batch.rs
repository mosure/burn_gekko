//! Small, bounded resident RGB sets with optional full-view features. Masked targets are always re-encoded.
use crate::encoder::{FrozenViews, ImageFeatures, image_features, image_tensor, normalize};
use anyhow::{Result, ensure};
use burn::tensor::{
    Tensor,
    backend::{AutodiffBackend, Backend},
};
use burn_gekko_data::RgbScene;
use burn_vjepa::{SparseTokenMask, VJepaConfig, VJepaEncoder};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;

/// Reproducible room/target permutation per epoch, reconstructed from the absolute sample index.
pub struct SampleSchedule {
    rooms: usize,
    views: usize,
    seed: u64,
    shuffle: bool,
    epoch: Option<usize>,
    order: Vec<(usize, usize)>,
}
impl SampleSchedule {
    pub fn new(rooms: usize, views: usize, seed: u64, shuffle: bool) -> Self {
        Self {
            rooms,
            views,
            seed,
            shuffle,
            epoch: None,
            order: Vec::new(),
        }
    }
    pub fn sample(&mut self, index: usize, fixed: bool) -> (usize, usize) {
        if !self.shuffle || fixed {
            return sample_indices(index, self.rooms, self.views, fixed);
        }
        let total = self.rooms * self.views;
        let epoch = index / total;
        if self.epoch != Some(epoch) {
            self.order = (0..total)
                .map(|i| sample_indices(i, self.rooms, self.views, false))
                .collect();
            self.order.shuffle(&mut ChaCha8Rng::seed_from_u64(
                self.seed
                    .wrapping_add((epoch as u64).wrapping_mul(0x9e3779b97f4a7c15)),
            ));
            self.epoch = Some(epoch);
        }
        self.order[index % total]
    }
}

pub struct ResidentScene<B: Backend> {
    pub seed: u64,
    pub rgb: Vec<Tensor<B, 4>>,
    pub normalized: Vec<Tensor<B, 4>>,
    pub full: Option<Vec<Tensor<B, 3>>>,
}
impl<B: Backend> ResidentScene<B> {
    pub fn new(
        scene: &RgbScene,
        encoder: &VJepaEncoder<B>,
        config: &VJepaConfig,
        cache: bool,
        device: &B::Device,
    ) -> Self {
        let rgb: Vec<_> = (0..scene.views.len())
            .map(|v| image_tensor(scene, v, device))
            .collect();
        let normalized: Vec<_> = rgb.iter().map(|v| normalize(v.clone(), config)).collect();
        let full = cache.then(|| {
            normalized
                .iter()
                .map(|v| encoder.forward_image(v.clone(), None).tokens)
                .collect()
        });
        Self {
            seed: scene.seed,
            rgb,
            normalized,
            full,
        }
    }
}

/// A full epoch visits every room/target pair; room count and camera count cannot alias.
pub fn sample_indices(index: usize, rooms: usize, views: usize, fixed: bool) -> (usize, usize) {
    assert!(rooms > 0 && views > 0);
    if fixed {
        (0, 0)
    } else {
        (index % rooms, (index / rooms) % views)
    }
}

pub fn encode_batch<B: AutodiffBackend>(
    encoder: &VJepaEncoder<B::InnerBackend>,
    scenes: &[ResidentScene<B::InnerBackend>],
    samples: &[(usize, usize)],
    references: usize,
    mask: &SparseTokenMask,
    features: ImageFeatures,
) -> Result<FrozenViews<B>> {
    ensure!(!samples.is_empty(), "empty batch");
    ensure!(
        samples.iter().all(|&(s, v)| s < scenes.len()
            && v < scenes[s].rgb.len()
            && references < scenes[s].rgb.len()),
        "invalid sample/reference selection"
    );
    let target_rgb = Tensor::cat(
        samples
            .iter()
            .map(|&(s, v)| scenes[s].rgb[v].clone())
            .collect(),
        0,
    );
    let target_images = Tensor::cat(
        samples
            .iter()
            .map(|&(s, v)| scenes[s].normalized[v].clone())
            .collect(),
        0,
    );
    let masked = image_features(
        encoder
            .forward_image(target_images.clone(), Some(mask))
            .tokens,
        target_rgb.clone(),
        Some(mask),
        features,
    );
    let full = if scenes.iter().all(|s| s.full.is_some()) {
        Tensor::cat(
            samples
                .iter()
                .map(|&(s, v)| scenes[s].full.as_ref().unwrap()[v].clone())
                .collect(),
            0,
        )
    } else {
        encoder.forward_image(target_images, None).tokens
    };
    let full = image_features(full, target_rgb.clone(), None, features);
    let refs = (1..=references)
        .map(|offset| {
            if scenes.iter().all(|s| s.full.is_some()) {
                Tensor::cat(
                    samples
                        .iter()
                        .map(|&(s, v)| {
                            scenes[s].full.as_ref().unwrap()[(v + offset) % scenes[s].rgb.len()]
                                .clone()
                        })
                        .collect(),
                    0,
                )
            } else {
                let images = Tensor::cat(
                    samples
                        .iter()
                        .map(|&(s, v)| {
                            scenes[s].normalized[(v + offset) % scenes[s].rgb.len()].clone()
                        })
                        .collect(),
                    0,
                );
                encoder.forward_image(images, None).tokens
            }
        })
        .enumerate()
        .map(|(i, tokens)| {
            let rgb = Tensor::cat(
                samples
                    .iter()
                    .map(|&(s, v)| scenes[s].rgb[(v + i + 1) % scenes[s].rgb.len()].clone())
                    .collect(),
                0,
            );
            Tensor::from_inner(image_features(tokens, rgb, None, features))
        })
        .collect();
    Ok(FrozenViews {
        masked: Tensor::from_inner(masked),
        full: Tensor::from_inner(full),
        references: refs,
        target_rgb: Tensor::from_inner(target_rgb),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::{Autodiff, NdArray};
    #[test]
    fn shuffled_schedule_is_bijective_and_replays_across_epoch_boundaries() {
        let mut full = SampleSchedule::new(7, 3, 42, true);
        let samples: Vec<_> = (0..45).map(|i| full.sample(i, false)).collect();
        assert_eq!(
            samples[..21]
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            21
        );
        assert_eq!(
            samples[21..42]
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            21
        );
        assert_ne!(&samples[..21], &samples[21..42]);
        let mut resumed = SampleSchedule::new(7, 3, 42, true);
        assert_eq!(
            (19..45)
                .map(|i| resumed.sample(i, false))
                .collect::<Vec<_>>(),
            samples[19..]
        );
    }
    #[test]
    fn cycle_visits_every_target_for_every_room() {
        let pairs: std::collections::BTreeSet<_> =
            (0..36).map(|i| sample_indices(i, 12, 3, false)).collect();
        assert_eq!(pairs.len(), 36);
        assert_eq!(sample_indices(36, 12, 3, false), (0, 0));
    }
    #[test]
    fn full_feature_cache_matches_online_batched_encoding_without_reusing_masked_tokens() {
        type B = Autodiff<NdArray<f32>>;
        let config = VJepaConfig::tiny_for_tests();
        let device = Default::default();
        let encoder = VJepaEncoder::new(&config, &device);
        let rgb = RgbScene {
            seed: 7,
            width: 32,
            height: 32,
            views: vec![vec![0.2; 32 * 32 * 3], vec![0.7; 32 * 32 * 3]],
        };
        let cached = vec![ResidentScene::new(&rgb, &encoder, &config, true, &device)];
        let online = vec![ResidentScene::new(&rgb, &encoder, &config, false, &device)];
        let mask = SparseTokenMask::new(vec![0, 3], 4).unwrap();
        let a = encode_batch::<B>(
            &encoder,
            &cached,
            &[(0, 0), (0, 1)],
            1,
            &mask,
            ImageFeatures::SemanticRgb,
        )
        .unwrap();
        let b = encode_batch::<B>(
            &encoder,
            &online,
            &[(0, 0), (0, 1)],
            1,
            &mask,
            ImageFeatures::SemanticRgb,
        )
        .unwrap();
        for (x, y) in [
            (a.masked, b.masked),
            (a.full, b.full),
            (a.references[0].clone(), b.references[0].clone()),
        ] {
            assert!(crate::train::scalar((x - y).abs().max()).unwrap() < 1e-5);
        }
    }
}
