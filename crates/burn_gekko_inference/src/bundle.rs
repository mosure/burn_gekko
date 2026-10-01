use anyhow::{Result, ensure};
use burn_gekko::{heads::spatial::SpatialDescriptorConfig, model::DecoderConfig};
use burn_vjepa::VJepaConfig;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelFile {
    pub name: String,
    pub bytes: usize,
    pub sha256: String,
}
impl ModelFile {
    pub fn verify(&self, bytes: &[u8]) -> Result<()> {
        ensure!(
            bytes.len() == self.bytes && digest(bytes) == self.sha256,
            "model checksum or size mismatch: {}",
            self.name
        );
        Ok(())
    }
}
pub fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Runtime architecture and weight identities, not an implicit randomly initialized fallback.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Bundle {
    pub schema: u32,
    pub id: String,
    pub foundation_sha256: String,
    pub camera_sha256: String,
    pub rgb_sha256: String,
    pub precision: String,
    pub image_size: usize,
    pub encoder: VJepaConfig,
    pub decoder: DecoderConfig,
    pub spatial_descriptor: SpatialDescriptorConfig,
    pub spatial_input_layer: usize,
    pub head_width: usize,
    pub mask_ratio: f32,
    pub mask_seed: u64,
    pub foundation: Vec<ModelFile>,
    pub camera: ModelFile,
    pub rgb: ModelFile,
    pub license: String,
    pub qualification: String,
}
impl Bundle {
    pub fn parse(text: &str) -> Result<Self> {
        let v: Self = toml::from_str(text)?;
        v.validate()?;
        Ok(v)
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.schema == 1 && self.precision == "f32",
            "unsupported inference bundle"
        );
        ensure!(
            (64..=256).contains(&self.image_size) && self.image_size.is_multiple_of(64),
            "unsupported demo size"
        );
        self.decoder.validate()?;
        self.spatial_descriptor.validate()?;
        ensure!(
            self.decoder.encoder_dim == self.encoder.encoder.embed_dim
                && self.encoder.patch_size == 16
                && self.head_width > 0
                && self.head_width <= 256
                && self.spatial_input_layer > 0,
            "invalid runtime architecture"
        );
        ensure!(
            self.mask_ratio > 0. && self.mask_ratio < 1.,
            "invalid completion mask"
        );
        let hash = |s: &str| s.len() == 64 && s.bytes().all(|c| c.is_ascii_hexdigit());
        ensure!(
            [
                &self.foundation_sha256,
                &self.camera_sha256,
                &self.rgb_sha256
            ]
            .into_iter()
            .all(|s| hash(s)),
            "invalid weight identity"
        );
        ensure!(
            !self.foundation.is_empty() && self.foundation.len() <= 64,
            "invalid weight parts"
        );
        let mut names = std::collections::BTreeSet::new();
        for file in self.foundation.iter().chain([&self.camera, &self.rgb]) {
            ensure!(
                !file.name.is_empty()
                    && file
                        .name
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || b"._-".contains(&c))
                    && !file.name.contains("..")
                    && names.insert(&file.name)
                    && file.bytes > 0
                    && file.bytes <= 64 * 1024 * 1024
                    && hash(&file.sha256),
                "invalid or duplicate model file"
            );
        }
        Ok(())
    }
}
