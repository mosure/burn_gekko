//! Checked, checkpoint-bound features for fitting small output heads independently.
use crate::{Split, sha256_file};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs,
    path::{Component, Path},
};

pub use burn_gekko_metrics::calibration::{CAMERA_FRAME, CameraTarget, camera_target};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ArrayFile {
    pub file: String,
    pub sha256: String,
    pub values: usize,
}
impl ArrayFile {
    pub fn load(&self, root: &Path) -> Result<Vec<f32>> {
        ensure!(
            Path::new(&self.file)
                .components()
                .all(|c| matches!(c, Component::Normal(_))),
            "unsafe feature-cache path"
        );
        let path = root.join(&self.file);
        ensure!(
            sha256_file(&path)? == self.sha256,
            "cached feature checksum mismatch"
        );
        let bytes = fs::read(path)?;
        ensure!(
            bytes.len() == self.values * 4,
            "feature-cache shape mismatch"
        );
        let values: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|x| f32::from_le_bytes(*x))
            .collect();
        ensure!(
            values.iter().all(|v| v.is_finite()),
            "nonfinite cached feature"
        );
        Ok(values)
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadSample {
    pub room_seed: u64,
    pub split: Split,
    pub target_view: usize,
    pub reference_view: usize,
    pub hidden_tokens: Vec<usize>,
    pub completion: ArrayFile,
    pub monocular: ArrayFile,
    pub camera_features: ArrayFile,
    pub rgb: ArrayFile,
    pub camera: CameraTarget,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadCache {
    pub schema: u32,
    pub checkpoint_sha256: String,
    pub dataset_id: String,
    pub dataset_manifest_sha256: String,
    pub config_sha256: String,
    pub source_sha256: String,
    pub grid: [usize; 2],
    pub latent_width: usize,
    pub camera_width: usize,
    pub camera_frame: String,
    pub input_contract: String,
    pub samples: Vec<HeadSample>,
}
impl HeadCache {
    pub fn load(root: &Path, expected: &str) -> Result<Self> {
        ensure!(
            sha256_file(&root.join("manifest.json"))? == expected,
            "head cache manifest changed"
        );
        let cache: Self = serde_json::from_slice(&fs::read(root.join("manifest.json"))?)?;
        cache.validate()?;
        Ok(cache)
    }
    pub fn validate(&self) -> Result<()> {
        let tokens = self.grid.iter().product::<usize>();
        ensure!(
            self.schema == 1
                && tokens > 0
                && self.latent_width > 0
                && self.camera_width > 0
                && self.camera_frame == CAMERA_FRAME,
            "invalid head-cache schema"
        );
        let mut identities = BTreeSet::new();
        let mut train = BTreeSet::new();
        let mut validation = BTreeSet::new();
        for row in &self.samples {
            ensure!(
                identities.insert((row.room_seed, row.target_view)),
                "duplicate cached target"
            );
            ensure!(
                row.target_view != row.reference_view
                    && !row.hidden_tokens.is_empty()
                    && row.hidden_tokens.len() < tokens,
                "invalid cached target/mask"
            );
            let ids: BTreeSet<_> = row.hidden_tokens.iter().copied().collect();
            ensure!(
                ids.len() == row.hidden_tokens.len() && ids.iter().all(|&x| x < tokens),
                "invalid hidden tokens"
            );
            ensure!(
                row.completion.values == tokens * self.latent_width
                    && row.monocular.values == row.completion.values
                    && row.rgb.values == tokens * 768
                    && row.camera_features.values == 32 * self.camera_width,
                "invalid cached array shape"
            );
            ensure!(
                row.camera.focal.iter().all(|v| v.is_finite() && *v > 0.)
                    && row.camera.regression().iter().all(|v| v.is_finite()),
                "invalid camera targets"
            );
            match row.split {
                Split::Train => {
                    train.insert(row.room_seed);
                }
                Split::Validation => {
                    validation.insert(row.room_seed);
                }
                Split::Test => anyhow::bail!("head fitting cache must not contain test rooms"),
            }
        }
        ensure!(
            !train.is_empty() && !validation.is_empty() && train.is_disjoint(&validation),
            "head train/validation split overlap or empty split"
        );
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn anchor_coordinates_preserve_signed_baseline_and_non_square_intrinsics() {
        let i = [
            1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        ];
        let mut target = i;
        target[12] = 2.;
        let c = camera_target(&target, &i, std::f32::consts::FRAC_PI_2, 640, 320).unwrap();
        assert_eq!(c.translation, [2., 0., 0.]);
        assert_eq!(&c.regression()[..9], &[1., 0., 0., 0., 1., 0., 1., 0., 0.]);
        assert!((c.focal[0] - 0.25).abs() < 1e-6 && (c.focal[1] - 0.5).abs() < 1e-6);
        let anchor = [
            0., 1., 0., 0., -1., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
        ];
        let c = camera_target(&target, &anchor, 1., 256, 256).unwrap();
        assert_eq!(c.translation, [0., -2., 0.]);
        assert_eq!(c.rotation, [[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]]);
    }
}
