//! Resolved schedule, validation and audited checkpoint ancestry.
use super::*;
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LatentConfig {
    pub dataset: PathBuf,
    pub teacher: EncoderSource,
    pub seed: u64,
    pub train_rooms: usize,
    pub validation_rooms: usize,
    pub export_rooms: usize,
    pub batch_size: usize,
    pub steps: usize,
    pub decay_steps: usize,
    pub warmup_steps: usize,
    pub max_seconds: u64,
    pub eval_every: usize,
    pub checkpoint_every: usize,
    pub decoder_width: usize,
    pub decoder_depth: usize,
    pub decoder_heads: usize,
    pub learning_rate: f64,
    pub encoder_lr_ratio: f64,
    pub weight_decay: f32,
    pub mask_ratio: f32,
    #[serde(default)]
    pub train_mask: MaskPattern,
    #[serde(default)]
    pub eval_mask: MaskPattern,
    /// Independent probe mask ratio permits comparisons with identical targets.
    #[serde(default)]
    pub eval_mask_ratio: Option<f32>,
    #[serde(default)]
    pub stable_attention: bool,
    #[serde(default = "default_cross_view_rope")]
    pub cross_view_rope: bool,
    #[serde(default)]
    pub fusion_auxiliary: crate::fusion_objective::FusionAuxiliary,
    /// Append a trained intermediate encoder level to the fusion input.
    /// The projection extension starts at zero in a new weights-only phase.
    #[serde(default)]
    pub spatial_input_layer: Option<usize>,
    /// Zero disables the additional features while retaining projection layout.
    #[serde(
        default = "default_spatial_input_scale",
        skip_serializing_if = "is_default_spatial_input_scale"
    )]
    pub spatial_input_scale: f64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spatial_descriptor: Option<burn_gekko::heads::spatial::SpatialDescriptorConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub equivariance: Option<crate::data::augmentation::EquivarianceConfig>,
    /// Training-only final-feature anchor, with its own audited immutable source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub encoder_preservation: Option<super::preservation::EncoderPreservationConfig>,
    /// Explicit renderer-supervised auxiliary; geometry never enters RGB inference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub view_geometry: Option<super::view_geometry::ViewGeometryConfig>,
    /// Audited weights-only new phase; both optimizers and the gate restart.
    #[serde(default)]
    pub warm_start: Option<WeightAncestor>,
    pub references: usize,
    pub visible_loss_weight: f64,
    pub ri_weight: f64,
    pub ri_start_step: usize,
    pub unfreeze: bool,
    /// Limit progressive unfreezing for controlled frozen/partial/full ablations.
    #[serde(default = "default_encoder_stage_cap")]
    pub encoder_stage_cap: usize,
    /// Explicit starting stage for an audited weights-only phase. Exact resume
    /// restores the checkpoint gate instead. Zero retains progressive unfreezing.
    #[serde(default, skip_serializing_if = "is_zero_stage")]
    pub initial_encoder_stage: usize,
    pub unfreeze_min_steps: usize,
    pub unfreeze_min_improvement: f64,
}
fn default_encoder_stage_cap() -> usize {
    2
}
fn is_zero_stage(value: &usize) -> bool {
    *value == 0
}
fn default_cross_view_rope() -> bool {
    true
}
fn default_spatial_input_scale() -> f64 {
    1.0
}
fn is_default_spatial_input_scale(value: &f64) -> bool {
    *value == 1.0
}
impl LatentConfig {
    pub fn validate(&self) -> Result<()> {
        if let Some(config) = &self.view_geometry {
            config.validate()?;
            ensure!(
                self.spatial_descriptor.is_some(),
                "geometry objective requires a spatial descriptor"
            );
        }
        if let Some(preservation) = &self.encoder_preservation {
            preservation.validate()?;
            ensure!(
                self.unfreeze && self.encoder_stage_cap > 0,
                "preservation requires encoder adaptation"
            );
        }
        self.fusion_auxiliary.validate()?;
        if let Some(head) = &self.spatial_descriptor {
            head.validate()?;
            ensure!(
                self.spatial_input_layer.is_some() && self.spatial_input_scale == 1.0,
                "spatial descriptor requires the full spatial route"
            );
            ensure!(
                self.fusion_auxiliary.descriptor_weight > 0.
                    || self.equivariance.is_some()
                    || self.view_geometry.is_some(),
                "spatial head requires descriptor training"
            );
        }
        if let Some(config) = &self.equivariance {
            config.validate()?;
            ensure!(
                self.spatial_descriptor.is_some(),
                "equivariance requires a spatial descriptor head"
            );
            ensure!(
                !self.fusion_auxiliary.enabled(),
                "register semantic affinity and known-transform studies separately"
            );
        }
        ensure!(
            self.spatial_input_layer.is_none_or(|x| x > 0),
            "spatial input uses a one-based encoder layer"
        );
        ensure!(
            (0.0..=1.0).contains(&self.spatial_input_scale)
                && (self.spatial_input_layer.is_some() || self.spatial_input_scale == 1.0),
            "invalid spatial feature scale"
        );
        ensure!(
            !self.fusion_auxiliary.anchor_warm_start || self.warm_start.is_some(),
            "student affinity anchor requires an audited warm start"
        );
        ensure!(self.encoder_stage_cap <= 2, "invalid encoder stage cap");
        ensure!(
            self.initial_encoder_stage <= self.encoder_stage_cap,
            "initial encoder stage exceeds the stage cap"
        );
        ensure!(
            self.initial_encoder_stage == 0 || (self.unfreeze && self.warm_start.is_some()),
            "a nonzero initial encoder stage requires unfreezing and an audited warm start"
        );
        ensure!(
            (1..=8192).contains(&self.train_rooms) && (1..=128).contains(&self.validation_rooms),
            "invalid room limits"
        );
        ensure!(
            (1..=32).contains(&self.batch_size) && self.export_rooms <= self.validation_rooms,
            "invalid batch/export limit"
        );
        ensure!(
            self.steps > 0
                && self.steps <= self.decay_steps
                && self.decay_steps <= 100_000
                && self.warmup_steps < self.decay_steps,
            "invalid step schedule"
        );
        ensure!(
            (1..=43200).contains(&self.max_seconds) && self.eval_every > 0,
            "invalid wall/probe limit"
        );
        ensure!(
            self.learning_rate > 0.
                && self.learning_rate <= 0.003
                && self.encoder_lr_ratio > 0.
                && self.encoder_lr_ratio <= 1.,
            "invalid learning rates"
        );
        ensure!(
            (0.0..=0.2).contains(&self.weight_decay)
                && self.mask_ratio > 0.
                && self.mask_ratio < 1.,
            "invalid decay/mask"
        );
        ensure!(
            (1..=3).contains(&self.references)
                && (0.0..=1.).contains(&self.visible_loss_weight)
                && (0.0..=1.).contains(&self.ri_weight),
            "invalid reference/loss settings"
        );
        ensure!(
            self.eval_mask_ratio.is_none_or(|r| r > 0. && r < 1.),
            "invalid evaluation mask ratio"
        );
        ensure!(
            self.unfreeze_min_steps > 0 && (0.0..1.).contains(&self.unfreeze_min_improvement),
            "invalid unfreeze gate"
        );
        ensure!(
            !matches!(self.teacher, EncoderSource::Checkpoint { .. }),
            "use the audited MIT Burnpack or a diagnostic tiny teacher"
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
        Ok(())
    }
    pub(super) fn identity(&self) -> Result<String> {
        let mut c = self.clone();
        c.steps = 1;
        c.max_seconds = 1;
        c.checkpoint_every = 0;
        // Probe cadence is part of the unfreezing decision and cannot change on resume.
        if let Some(config) = &self.view_geometry {
            fingerprint(&(
                c,
                crate::provenance::identity()?,
                sha256_file(&config.cache.join("manifest.json"))?,
            ))
        } else {
            fingerprint(&(c, crate::provenance::identity()?))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeightAncestor {
    pub checkpoint: PathBuf,
    pub model_sha256: String,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Snapshot {
    pub(super) identity: String,
    pub(super) dataset_id: String,
    pub(crate) teacher_id: String,
    pub(super) completed_steps: usize,
    pub(super) gate: UnfreezeGate,
    pub(super) backend: String,
    pub(crate) model_sha256: String,
    pub(super) encoder_optimizer_sha256: String,
    pub(super) fusion_optimizer_sha256: String,
    pub(super) noncommercial_weight_dependencies: Vec<String>,
    #[serde(default)]
    pub(super) weight_ancestors: Vec<WeightAncestor>,
}

/// Verify our checkpoint and its recorded weights-only ancestry, without
/// interpreting an identity mismatch as permission for an exact resume.
pub(crate) fn audited_checkpoint(a: &WeightAncestor, teacher_id: &str) -> Result<Snapshot> {
    fn visit(a: &WeightAncestor, teacher_id: &str, stack: &mut Vec<PathBuf>) -> Result<Snapshot> {
        ensure!(stack.len() < 32, "checkpoint ancestry too deep");
        let path = fs::canonicalize(&a.checkpoint)?;
        ensure!(!stack.contains(&path), "checkpoint ancestry cycle");
        ensure!(
            path.starts_with(fs::canonicalize(".data")?),
            "checkpoint outside .data"
        );
        stack.push(path.clone());
        let m: Snapshot = serde_json::from_slice(&fs::read(path.join("metadata.json"))?)?;
        ensure!(
            m.teacher_id == teacher_id && m.noncommercial_weight_dependencies.is_empty(),
            "unreviewed/noncommercial checkpoint teacher or ancestry"
        );
        ensure!(
            m.model_sha256 == a.model_sha256
                && sha256_file(&path.join("model.mpk"))? == a.model_sha256,
            "checkpoint checksum mismatch"
        );
        let provenance: serde_json::Value =
            serde_json::from_slice(&fs::read(path.parent().unwrap().join("provenance.json"))?)?;
        ensure!(
            provenance["task"] == "fixed_vjepa21_latent_prediction"
                && provenance["teacher_id"] == teacher_id
                && provenance["noncommercial_weight_dependencies"]
                    .as_array()
                    .is_some_and(Vec::is_empty),
            "missing or incompatible latent provenance"
        );
        // Older exact resumes name their parent in provenance, not metadata.
        if let Some(parent) = provenance["resume"].as_str() {
            let parent_meta: Snapshot =
                serde_json::from_slice(&fs::read(Path::new(parent).join("metadata.json"))?)?;
            visit(
                &WeightAncestor {
                    checkpoint: parent.into(),
                    model_sha256: parent_meta.model_sha256,
                },
                teacher_id,
                stack,
            )?;
        }
        for parent in &m.weight_ancestors {
            visit(parent, teacher_id, stack)?;
        }
        stack.pop();
        Ok(m)
    }
    visit(a, teacher_id, &mut Vec::new())
}
