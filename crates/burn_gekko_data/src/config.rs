use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

pub const SCHEMA: u32 = 1;
pub const GENERATOR: &str = "bevy_zeroverse=0.25.0;bevy_zeroverse_burn=0.8.0;gekko-capture=4";
pub const PILOT05_GENERATOR: &str =
    "bevy_zeroverse=0.23.0;bevy_zeroverse_burn=0.6.0;gekko-capture=3";
pub const PILOT02_GENERATOR: &str =
    "bevy_zeroverse=0.22.0;bevy_zeroverse_burn=0.5.0;gekko-capture=2";
pub const LEGACY_GENERATOR: &str =
    "bevy_zeroverse=0.21.0;bevy_zeroverse_burn=0.4.0;gekko-capture=1";

/// Bounded workstation capture configuration. Defaults remain fixture-sized.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CaptureConfig {
    pub schema: u32,
    pub seed: u64,
    /// Counts of independent rooms, in train/validation/test order.
    pub split_scenes: [usize; 3],
    pub width: usize,
    pub height: usize,
    pub cameras: usize,
    pub density: f32,
    pub timeout_secs: u64,
    /// Latest generator shared-target camera rig. Omitted fields preserve legacy cache identities.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub multiview: Option<bool>,
    /// Published continuous camera spacing control. Mutually exclusive with
    /// disabled grouping; stored in the immutable capture identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera_baseline: Option<f32>,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            schema: SCHEMA,
            seed: 20260927,
            split_scenes: [2, 1, 1],
            width: 64,
            height: 64,
            cameras: 2,
            density: 0.35,
            timeout_secs: 240,
            multiview: None,
            camera_baseline: None,
        }
    }
}

impl CaptureConfig {
    pub fn scenes(&self) -> usize {
        self.split_scenes.iter().sum()
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(self.schema == SCHEMA, "unsupported capture schema");
        if let Some(baseline) = self.camera_baseline {
            ensure!(
                baseline.is_finite() && (0.0..=1.0).contains(&baseline),
                "invalid camera baseline"
            );
            ensure!(
                self.multiview != Some(false),
                "camera baseline requires grouped views"
            );
        }
        ensure!(
            self.split_scenes.iter().all(|&n| (1..=32768).contains(&n)) && self.scenes() <= 32768,
            "capture requires nonempty splits and at most 32768 rooms in total"
        );
        ensure!(
            (32..=512).contains(&self.width)
                && (32..=512).contains(&self.height)
                && self.width.is_multiple_of(16)
                && self.height.is_multiple_of(16),
            "capture images must be 32..=512 and divisible by 16"
        );
        ensure!(
            (2..=4).contains(&self.cameras),
            "preflight supports 2..=4 cameras"
        );
        ensure!(
            self.density.is_finite() && (0.0..=1.0).contains(&self.density),
            "invalid density"
        );
        ensure!(
            (1..=14400).contains(&self.timeout_secs),
            "capture timeout must be 1..=14400 seconds"
        );
        ensure!(
            self.seed.checked_add(self.scenes() as u64).is_some(),
            "seed overflow"
        );
        Ok(())
    }
    pub fn split(&self, index: usize) -> Result<Split> {
        ensure!(index < self.scenes(), "scene index out of range");
        Ok(if index < self.split_scenes[0] {
            Split::Train
        } else if index < self.split_scenes[0] + self.split_scenes[1] {
            Split::Validation
        } else {
            Split::Test
        })
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Split {
    Train,
    Validation,
    Test,
}
