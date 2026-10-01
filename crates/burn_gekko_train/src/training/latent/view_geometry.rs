//! Training-only renderer labels; no camera or geometry is passed to the model.
use crate::{latent::LatentModel, latent_pilot::LatentConfig};
use anyhow::{Result, ensure};
use burn::tensor::{Tensor, TensorData, backend::AutodiffBackend};
use burn_gekko_data::{
    SceneEntry, Split,
    view_targets::{Cache, labels},
};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewGeometryConfig {
    pub cache: PathBuf,
    pub weight: f64,
    pub temperature: f64,
}
impl ViewGeometryConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.weight.is_finite()
                && self.weight > 0.
                && self.weight <= 1.
                && (0.01..=1.).contains(&self.temperature),
            "invalid geometry objective settings"
        );
        Ok(())
    }
}

pub(super) struct Targets {
    cache: Cache,
    rooms: Vec<usize>,
}
impl Targets {
    pub fn load(c: &LatentConfig, entries: &[SceneEntry]) -> Result<Option<Self>> {
        let Some(config) = &c.view_geometry else {
            return Ok(None);
        };
        let cache = Cache::load(&config.cache, &c.dataset)?;
        ensure!(
            cache.manifest.patch == 16,
            "geometry objective requires 16px patches"
        );
        let rooms = entries
            .iter()
            .map(|e| cache.room_index(e.seed, Split::Train))
            .collect::<Result<_>>()?;
        Ok(Some(Self { cache, rooms }))
    }
    pub fn loss<B: AutodiffBackend>(
        &self,
        model: &LatentModel<B>,
        target: Tensor<B, 3>,
        reference: Tensor<B, 3>,
        samples: &[(usize, usize)],
        reference_offset: usize,
        config: &ViewGeometryConfig,
    ) -> Result<(Tensor<B, 1>, f64)> {
        let m = &self.cache.manifest;
        let n = m.grid[0] * m.grid[1];
        let mut forward = Vec::with_capacity(samples.len() * n * n);
        let mut backward = Vec::with_capacity(samples.len() * n * n);
        let mut valid = 0;
        for &(s, t) in samples {
            let r = (t + reference_offset) % m.views;
            for (a, b, output) in [(t, r, &mut forward), (r, t, &mut backward)] {
                let (p, count) = labels(&self.cache.pair(self.rooms[s], a, b)?, m.grid, m.patch)?;
                output.extend(p);
                valid += count;
            }
        }
        let device = target.device();
        let descriptor = |a: Tensor<B, 3>, b| -> Result<Tensor<B, 3>> {
            model
                .spatial_descriptor(a.clone(), model.fusion.decoder.pair_features(a, b, m.grid)?)
                .ok_or_else(|| anyhow::anyhow!("geometry objective requires spatial head"))
        };
        let a = descriptor(target.clone(), reference.clone())?;
        let b = descriptor(reference, target)?;
        let shape = [samples.len(), n, n];
        let loss = burn_gekko::objectives::correspondence::bidirectional_nll(
            a,
            b,
            Tensor::from_data(TensorData::new(forward, shape), &device),
            Tensor::from_data(TensorData::new(backward, shape), &device),
            config.temperature,
        );
        Ok((loss, valid as f64 / (2 * samples.len() * n) as f64))
    }
}
