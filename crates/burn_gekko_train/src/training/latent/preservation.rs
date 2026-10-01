//! A training-only frozen encoder. Its targets never enter the fusion forward pass.
use super::*;
use burn::module::{ModuleVisitor, Param};
use burn::tensor::backend::Backend;
use burn_gekko::objectives::preservation::relative_feature_mse;
use burn_vjepa::{SparseTokenMask, VJepaConfig};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EncoderPreservationConfig {
    pub weight: f64,
    /// Fixed independently of the student's weights-only or optimizer continuation.
    pub anchor: WeightAncestor,
}

impl EncoderPreservationConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.weight > 0. && self.weight <= 16.,
            "encoder preservation weight must be finite and in (0, 16]"
        );
        ensure!(
            !self.anchor.checkpoint.as_os_str().is_empty()
                && self.anchor.model_sha256.len() == 64
                && self
                    .anchor
                    .model_sha256
                    .bytes()
                    .all(|c| c.is_ascii_hexdigit()),
            "encoder preservation requires a pinned own checkpoint"
        );
        Ok(())
    }
}

pub(super) struct FrozenEncoder<B: Backend> {
    encoder: VJepaEncoder<B>,
    config: VJepaConfig,
    first: Tensor<B, 2>,
    last: Tensor<B, 2>,
}

pub(super) struct Targets<B: Backend> {
    sparse: Tensor<B, 3>,
    dense: Tensor<B, 3>,
    references: Vec<Tensor<B, 3>>,
}

impl<B: Backend> FrozenEncoder<B> {
    /// Exact parameter comparison before the first update of a common-parent run.
    /// A feature-level CUDA mismatch is not evidence of different loaded weights.
    pub fn initial_parameter_check(&self, student: &VJepaEncoder<B>) -> Result<serde_json::Value> {
        struct Parameters<B: Backend>(Vec<Tensor<B, 1>>);
        impl<B: Backend> ModuleVisitor<B> for Parameters<B> {
            fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<B, D>>) {
                self.0.push(p.val().flatten(0, D - 1));
            }
        }
        let mut student_values = Parameters(Vec::new());
        let mut anchor_values = Parameters(Vec::new());
        student.visit(&mut student_values);
        self.encoder.visit(&mut anchor_values);
        ensure!(
            student_values.0.len() == anchor_values.0.len(),
            "anchor parameter count mismatch"
        );
        let count = student_values.0.len();
        let mut elements = 0;
        let differences = student_values
            .0
            .into_iter()
            .zip(anchor_values.0)
            .map(|(s, a)| {
                ensure!(s.dims() == a.dims(), "anchor parameter shape mismatch");
                elements += s.dims()[0];
                Ok((s - a).abs().max())
            })
            .collect::<Result<Vec<_>>>()?;
        let maximum = scalar(Tensor::cat(differences, 0).max())?;
        ensure!(
            maximum == 0.,
            "common-parent encoder parameters differ before training: {maximum}"
        );
        Ok(
            serde_json::json!({"parameter_tensors":count,"parameter_elements":elements,"max_abs_difference":maximum,"scope":"Every encoder floating parameter compared before the first optimizer update; this does not assert bitwise forward equivalence across autodiff and inference."}),
        )
    }

    pub fn load(
        c: &EncoderPreservationConfig,
        teacher_id: &str,
        device: &B::Device,
    ) -> Result<Self> {
        c.validate()?;
        let assessed = crate::latent_assess::load_assessed_model::<B>(&c.anchor, device)?;
        ensure!(
            assessed.teacher_id == teacher_id,
            "preservation teacher lineage mismatch"
        );
        let encoder = assessed.model.encoder.no_grad();
        let last = encoder.blocks.last().unwrap().attn.qkv.weight.val();
        let first = encoder.blocks[0].attn.qkv.weight.val();
        Ok(Self {
            encoder,
            config: assessed.model.encoder_config,
            first,
            last,
        })
    }

    fn encode(&self, rgb: Tensor<B, 4>, mask: Option<&SparseTokenMask>) -> Tensor<B, 3> {
        self.encoder
            .forward_image_capture_layers(normalize(rgb, &self.config), mask, &[])
            .tokens
    }

    pub fn targets(
        &self,
        rgb: Tensor<B, 4>,
        references: &[Tensor<B, 4>],
        mask: &SparseTokenMask,
    ) -> Targets<B> {
        let b = rgb.dims()[0];
        let sparse = self.encode(rgb.clone(), Some(mask));
        let mut views = vec![rgb];
        views.extend_from_slice(references);
        let dense = self.encode(Tensor::cat(views, 0), None);
        Targets {
            sparse,
            dense: dense.clone().slice_dim(0, 0..b),
            references: (0..references.len())
                .map(|i| dense.clone().slice_dim(0, (i + 1) * b..(i + 2) * b))
                .collect(),
        }
    }

    pub fn probe_deltas(&self) -> Result<[f64; 2]> {
        Ok([
            scalar(
                (self.encoder.blocks[0].attn.qkv.weight.val() - self.first.clone())
                    .abs()
                    .max(),
            )?,
            scalar(
                (self.encoder.blocks.last().unwrap().attn.qkv.weight.val() - self.last.clone())
                    .abs()
                    .max(),
            )?,
        ])
    }
}

/// Equal route weight: sparse target, full target, then each full reference.
/// Only final features have preservation targets; early blocks can receive their gradients.
pub(super) fn route_losses<B: AutodiffBackend>(
    model: &LatentModel<B>,
    sparse: Tensor<B, 3>,
    dense: Tensor<B, 3>,
    references: &[Tensor<B, 3>],
    targets: Targets<B::InnerBackend>,
) -> Tensor<B, 1> {
    assert_eq!(references.len(), targets.references.len());
    let mut student = vec![sparse, dense];
    student.extend_from_slice(references);
    let mut anchor = vec![targets.sparse, targets.dense];
    anchor.extend(targets.references);
    let losses = student
        .into_iter()
        .zip(anchor)
        .map(|(s, a)| relative_feature_mse(model.final_encoder_features(s), Tensor::from_inner(a)))
        .collect();
    Tensor::cat(losses, 0)
}
