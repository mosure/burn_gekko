//! Resolved schedule, validation and audited checkpoint ancestry.
use super::*;
pub(crate) const REVIEWED_VJEPA_ID: &str =
    "c408f68dd18a38824d0fa1d615e6f9f9f04f111d71a6f7c846f41dc187a8795f";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Initialization {
    Scratch {
        width: usize,
        depth: usize,
        heads: usize,
    },
    Vjepa21 {
        directory: PathBuf,
        expected_encoder_id: String,
    },
}
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RgbCache {
    #[default]
    Device,
    Host,
}
fn default_ri_weight() -> f64 {
    0.1
}
fn default_cache_mib() -> usize {
    32768
}
fn default_probe_rooms() -> usize {
    4
}
fn default_transport_displacement() -> f32 {
    64.
}
fn default_transport_weight() -> f64 {
    5.
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct E2eConfig {
    pub dataset: PathBuf,
    pub initialization: Initialization,
    pub seed: u64,
    pub train_rooms: usize,
    pub batch_size: usize,
    pub steps: usize,
    pub decay_steps: usize,
    pub warmup_steps: usize,
    pub max_seconds: u64,
    pub eval_every: usize,
    pub decoder_width: usize,
    pub decoder_depth: usize,
    pub decoder_heads: usize,
    #[serde(default)]
    pub decoder_qk_norm: bool,
    #[serde(default)]
    pub decoder_stable_attention: bool,
    pub learning_rate: f64,
    pub encoder_lr_ratio: f64,
    pub weight_decay: f32,
    pub mask_ratio: f32,
    pub references: usize,
    pub appearance_bypass: bool,
    pub edge_loss_weight: f64,
    pub gradient_energy_weight: f64,
    pub unfreeze: bool,
    pub unfreeze_min_steps: usize,
    pub unfreeze_min_improvement: f64,
    #[serde(default)]
    pub fixed_example: bool,
    #[serde(default)]
    pub rgb_head: RgbHead,
    #[serde(default)]
    pub rgb_cache: RgbCache,
    #[serde(default = "default_cache_mib")]
    pub rgb_cache_max_mib: usize,
    #[serde(default)]
    pub checkpoint_every: usize,
    #[serde(default)]
    pub ri_start_step: usize,
    #[serde(default = "default_ri_weight")]
    pub ri_weight: f64,
    #[serde(default = "default_probe_rooms")]
    pub probe_rooms: usize,
    #[serde(default)]
    pub appearance_transport: bool,
    #[serde(default = "default_transport_displacement")]
    pub transport_max_displacement: f32,
    #[serde(default = "default_transport_weight")]
    pub transport_loss_weight: f64,
    #[serde(default)]
    pub transport_pyramid_loss: bool,
    #[serde(default)]
    pub transport_coarse_smoothness_weight: f64,
    /// Randomly initialized shared RGB encoder and explicit feature matching.
    #[serde(default)]
    pub transport_matching: bool,
    /// Chance of a batch using independent projective RGB augmentations of each
    /// training target as references. All validation/test views remain captured.
    #[serde(default)]
    pub synthetic_reference_probability: f64,
}
impl E2eConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (0.0..=1.0).contains(&self.synthetic_reference_probability),
            "invalid synthetic reference probability"
        );
        ensure!(
            (1..=32768).contains(&self.train_rooms) && (1..=32).contains(&self.batch_size),
            "invalid bounded training size"
        );
        ensure!(
            self.steps > 0
                && self.steps <= self.decay_steps
                && self.decay_steps <= 500_000
                && self.warmup_steps < self.decay_steps,
            "invalid schedule"
        );
        ensure!(
            (1..=43200).contains(&self.max_seconds) && self.eval_every > 0,
            "invalid wall/probe limit"
        );
        ensure!(
            (1..=65536).contains(&self.rgb_cache_max_mib),
            "invalid RGB cache ceiling"
        );
        ensure!(
            (1..=128).contains(&self.probe_rooms),
            "invalid probe room count"
        );
        ensure!((0.0..=1.0).contains(&self.ri_weight), "invalid RI weight");
        ensure!(
            !self.appearance_transport || self.rgb_head == RgbHead::Direct,
            "appearance transport requires direct RGB"
        );
        ensure!(
            !self.transport_pyramid_loss || self.appearance_transport,
            "image-pyramid transport loss requires the appearance head"
        );
        ensure!(
            !self.transport_matching
                || (self.appearance_transport && self.transport_max_displacement >= 8.),
            "RGB matching requires appearance transport and at least eight pixels of search range"
        );
        ensure!(
            (0.0..=1.0).contains(&self.transport_coarse_smoothness_weight)
                && (self.transport_coarse_smoothness_weight == 0. || self.appearance_transport),
            "invalid coarse flow smoothness setting"
        );
        ensure!(
            (1.0..=128.0).contains(&self.transport_max_displacement)
                && (0.0..=20.0).contains(&self.transport_loss_weight),
            "invalid appearance transport settings"
        );
        ensure!(
            self.rgb_head != RgbHead::Direct || self.gradient_energy_weight == 0.,
            "direct RGB objective does not use an energy reward"
        );
        ensure!(
            self.learning_rate.is_finite()
                && (0.0..=0.003).contains(&self.learning_rate)
                && self.learning_rate > 0.
                && self.encoder_lr_ratio > 0.
                && self.encoder_lr_ratio <= 1.,
            "invalid learning rate"
        );
        ensure!(
            self.mask_ratio > 0. && self.mask_ratio < 1. && (1..=3).contains(&self.references),
            "invalid mask/references"
        );
        ensure!(
            (0.0..=0.2).contains(&self.weight_decay)
                && (0.0..=8.).contains(&self.edge_loss_weight)
                && (0.0..=0.1).contains(&self.gradient_energy_weight),
            "invalid loss/decay"
        );
        ensure!(
            self.unfreeze_min_steps > 0 && (0.0..1.).contains(&self.unfreeze_min_improvement),
            "invalid unfreeze gate"
        );
        ensure!(
            !self.fixed_example || self.batch_size == 1,
            "fixed example requires batch one"
        );
        DecoderConfig {
            encoder_dim: 32,
            width: self.decoder_width,
            depth: self.decoder_depth,
            heads: self.decoder_heads,
            patch: 16,
        }
        .validate()?;
        ensure!(
            (self.decoder_width / self.decoder_heads).is_multiple_of(4),
            "invalid rotary head width"
        );
        match &self.initialization {
            Initialization::Scratch {
                width,
                depth,
                heads,
            } => ensure!(
                (32..=768).contains(width)
                    && (1..=12).contains(depth)
                    && *heads > 0
                    && width.is_multiple_of(*heads)
                    && (width / heads).is_multiple_of(4),
                "invalid scratch encoder dimensions"
            ),
            // Explicitly pin the package whose official EMA tensor provenance
            // and MIT release have been independently checked in this workspace.
            Initialization::Vjepa21 {
                expected_encoder_id,
                ..
            } => ensure!(
                expected_encoder_id == REVIEWED_VJEPA_ID,
                "unreviewed encoder package: add a provenance audit before enabling"
            ),
        }
        Ok(())
    }
    pub(super) fn identity(&self) -> Result<String> {
        let mut c = self.clone();
        c.steps = 1;
        c.max_seconds = 1;
        c.eval_every = 1;
        c.checkpoint_every = 0;
        c.rgb_cache = RgbCache::Device;
        c.rgb_cache_max_mib = default_cache_mib();
        fingerprint(&(c, crate::provenance::identity()?))
    }
}
