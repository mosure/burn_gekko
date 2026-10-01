use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadTrainConfig {
    pub cache: PathBuf,
    pub cache_sha256: String,
    pub checkpoint_sha256: String,
    pub seed: u64,
    pub steps: usize,
    pub warmup_steps: usize,
    pub width: usize,
    pub rgb_batch: usize,
    pub camera_batch: usize,
    pub learning_rate: f64,
    pub weight_decay: f32,
    pub max_seconds: u64,
    pub eval_every: usize,
}
impl HeadTrainConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.steps > 0
                && self.warmup_steps > 0
                && self.warmup_steps <= self.steps
                && self.width > 0
                && self.width <= 512
                && self.rgb_batch > 0
                && self.camera_batch > 0
                && self.eval_every > 0
                && self.max_seconds > 0,
            "invalid head training sizes"
        );
        ensure!(
            self.learning_rate.is_finite()
                && self.learning_rate > 0.
                && self.learning_rate <= 0.01
                && self.weight_decay.is_finite()
                && self.weight_decay >= 0.,
            "invalid head optimizer"
        );
        ensure!(
            self.cache_sha256.len() == 64 && self.checkpoint_sha256.len() == 64,
            "missing head source pins"
        );
        Ok(())
    }
    pub fn identity(&self) -> Result<String> {
        let mut c = self.clone();
        c.max_seconds = 1;
        burn_gekko_data::fingerprint(&(c, crate::provenance::identity()?))
    }
}
