#[cfg(test)]
use super::attention::Attention;
use super::attention::Block;
use anyhow::{Result, ensure};
use burn::{
    module::{Module, Param},
    nn::{Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Int, Tensor, TensorData, activation, backend::Backend},
};
use burn_vjepa::{SparseTokenMask, get_2d_sincos_pos_embed};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DecoderPosition {
    #[default]
    Absolute,
    Rope2d,
}

/// Read-only dense-pair diagnostics. Content maps remove RoPE only from the
/// readout; they do not change the hidden states or the actual attention update.
pub struct AttentionTrace<B: Backend> {
    pub probability: Tensor<B, 3>,
    pub logits: Tensor<B, 3>,
    pub content: Tensor<B, 3>,
    pub centered_content: Tensor<B, 3>,
}

pub struct PairTrace<B: Backend> {
    pub features: Tensor<B, 3>,
    pub layers: Vec<AttentionTrace<B>>,
}

pub struct PairTraining<B: Backend> {
    pub features: Tensor<B, 3>,
    pub logits: Vec<Tensor<B, 3>>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct DecoderConfig {
    pub encoder_dim: usize,
    pub width: usize,
    pub depth: usize,
    pub heads: usize,
    pub patch: usize,
}
impl DecoderConfig {
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.encoder_dim > 0 && (8..=512).contains(&self.width) && self.width.is_multiple_of(4),
            "invalid decoder width"
        );
        ensure!(
            (1..=8).contains(&self.depth)
                && self.heads > 0
                && self.width.is_multiple_of(self.heads),
            "invalid decoder attention dimensions"
        );
        ensure!(self.patch == 16, "preflight requires V-JEPA patch size 16");
        Ok(())
    }
}

/// Joint attention over a set of references. Reference order has no semantic meaning.
/// This extends the published pairwise Gekko conditioning; it is not a pretrained Gekko checkpoint.
#[derive(Module, Debug)]
pub struct GekkoDecoder<B: Backend> {
    projection: Linear<B>,
    blocks: Vec<Block<B>>,
    norm: LayerNorm<B>,
    pub cross_rgb: Linear<B>,
    pub mae_rgb: Linear<B>,
    pub ri: Linear<B>,
    cross_mask: Param<Tensor<B, 3>>,
    mae_mask: Param<Tensor<B, 3>>,
    reference_role: Param<Tensor<B, 3>>,
    width: usize,
    #[module(skip)]
    position_encoding: DecoderPosition,
    #[module(skip)]
    head_dim: usize,
    #[module(skip)]
    mae_context_before_self: bool,
    #[module(skip)]
    cross_view_rope: bool,
    pub appearance_head: Option<crate::appearance::AppearanceHead<B>>,
}

pub struct Predictions<B: Backend> {
    pub cross_rgb: Tensor<B, 3>,
    pub mae_rgb: Tensor<B, 3>,
    pub ri: Tensor<B, 3>,
}

impl<B: Backend> GekkoDecoder<B> {
    pub fn new(c: &DecoderConfig, device: &B::Device) -> Result<Self> {
        Self::with_position(c, DecoderPosition::Absolute, device)
    }
    pub fn with_position(
        c: &DecoderConfig,
        position_encoding: DecoderPosition,
        device: &B::Device,
    ) -> Result<Self> {
        Self::with_reconstruction(c, position_encoding, false, device)
    }
    pub fn with_reconstruction(
        c: &DecoderConfig,
        position_encoding: DecoderPosition,
        predict_patch_stats: bool,
        device: &B::Device,
    ) -> Result<Self> {
        c.validate()?;
        ensure!(
            position_encoding == DecoderPosition::Absolute || (c.width / c.heads).is_multiple_of(4),
            "2D rotary attention requires head width divisible by four"
        );
        let token = || {
            Initializer::Normal {
                mean: 0.0,
                std: 0.02,
            }
            .init([1, 1, c.width], device)
        };
        Ok(Self {
            projection: LinearConfig::new(c.encoder_dim, c.width).init(device),
            blocks: (0..c.depth)
                .map(|_| Block::new(c.width, c.heads, device))
                .collect(),
            norm: LayerNormConfig::new(c.width).init(device),
            cross_rgb: LinearConfig::new(
                c.width,
                c.patch * c.patch * 3 + usize::from(predict_patch_stats) * 2,
            )
            .init(device),
            mae_rgb: LinearConfig::new(
                c.width,
                c.patch * c.patch * 3 + usize::from(predict_patch_stats) * 2,
            )
            .init(device),
            ri: LinearConfig::new(c.width, c.patch * c.patch).init(device),
            cross_mask: token(),
            mae_mask: token(),
            reference_role: token(),
            width: c.width,
            position_encoding,
            head_dim: c.width / c.heads,
            mae_context_before_self: false,
            cross_view_rope: true,
            appearance_head: None,
        })
    }
    /// Preserve legacy checkpoint behavior unless the corrected Gekko MAE context is requested.
    pub fn with_mae_context_before_self(mut self, enabled: bool) -> Self {
        self.mae_context_before_self = enabled;
        self
    }
    pub(crate) fn encoder_input_width(&self) -> usize {
        self.projection.weight.val().dims()[0]
    }
    /// Weights-only architecture extension: preserve the existing projection
    /// and initially ignore appended encoder features. Never use on exact resume.
    pub(crate) fn extend_encoder_input_zero(mut self, additional: usize) -> Self {
        assert!(additional > 0);
        self.projection.weight = self.projection.weight.map(|weight| {
            let required = weight.is_require_grad();
            let extra = Tensor::zeros([additional, weight.dims()[1]], &weight.device());
            Tensor::cat(vec![weight, extra], 0)
                .detach()
                .set_require_grad(required)
        });
        self
    }
    /// Keep image RoPE in self/MAE attention while independently controlling
    /// the same-coordinate prior between distinct views.
    pub fn with_cross_view_rope(mut self, enabled: bool) -> Self {
        self.cross_view_rope = enabled;
        self
    }
    /// Optional learned per-head query/key normalization, initialized from
    /// scratch in both attention slots. Released Gekko self-attention motivates
    /// this ablation; applying it to cross-attention is an explicit extension.
    /// Enabling it does not load or transfer any released parameters.
    /// F64 softmax/value accumulation for strict set-order numerical checks.
    /// Encoder and linear projections retain the backend's configured precision.
    pub fn with_stable_attention(mut self, enabled: bool) -> Self {
        for block in &mut self.blocks {
            block.self_attn.stable_attention = enabled;
            block.cross_attn.stable_attention = enabled;
        }
        self
    }

    pub fn with_attention_normalization(mut self, enabled: bool, device: &B::Device) -> Self {
        if enabled {
            for block in &mut self.blocks {
                for attention in [&mut block.self_attn, &mut block.cross_attn] {
                    attention.q_norm = Some(
                        LayerNormConfig::new(self.head_dim)
                            .with_epsilon(1e-6)
                            .init(device),
                    );
                    attention.k_norm = Some(
                        LayerNormConfig::new(self.head_dim)
                            .with_epsilon(1e-6)
                            .init(device),
                    );
                }
            }
        }
        self
    }

    fn position(&self, grid: [usize; 2], device: &B::Device) -> Tensor<B, 3> {
        if self.position_encoding == DecoderPosition::Rope2d {
            return Tensor::zeros([1, grid[0] * grid[1], self.width], device);
        }
        Tensor::from_data(
            TensorData::new(
                get_2d_sincos_pos_embed(self.width, grid[0], grid[1]),
                [1, grid[0] * grid[1], self.width],
            ),
            device,
        )
    }
    fn target(
        &self,
        visible: Tensor<B, 3>,
        mask: &SparseTokenMask,
        grid: [usize; 2],
        mae: bool,
    ) -> Tensor<B, 3> {
        let [b, k, _] = visible.dims();
        let device = visible.device();
        let token = if mae {
            self.mae_mask.val()
        } else {
            self.cross_mask.val()
        }
        .expand([b, 1, self.width]);
        let bank = Tensor::cat(vec![self.projection.forward(visible), token], 1);
        let mut indices = vec![k as i64; grid[0] * grid[1]];
        for (i, &index) in mask.indices().iter().enumerate() {
            indices[index] = i as i64;
        }
        let indices =
            Tensor::<B, 1, Int>::from_data(TensorData::new(indices, [grid[0] * grid[1]]), &device);
        bank.select(1, indices) + self.position(grid, &device)
    }
    fn decode(
        &self,
        mut target: Tensor<B, 3>,
        references: Option<Tensor<B, 3>>,
        rotary: Option<&crate::rotary::Rotary2d<B>>,
    ) -> Tensor<B, 3> {
        for block in &self.blocks {
            target = block.forward(
                target,
                references.clone(),
                rotary,
                self.mae_context_before_self,
                self.cross_view_rope,
            );
        }
        self.norm.forward(target)
    }
    /// Masked-target features for an explicit appearance-transport head. The dense
    /// target is deliberately absent from this API: it cannot leak hidden RGB.
    pub fn completion_features(
        &self,
        masked_target: Tensor<B, 3>,
        reference: Tensor<B, 3>,
        mask: &SparseTokenMask,
        grid: [usize; 2],
    ) -> Result<Tensor<B, 3>> {
        ensure!(
            reference.dims()[1] == grid[0] * grid[1],
            "reference grid mismatch"
        );
        ensure!(
            masked_target.dims()[1] == mask.len(),
            "sparse target mismatch"
        );
        let device = masked_target.device();
        let rotary = (self.position_encoding == DecoderPosition::Rope2d)
            .then(|| crate::rotary::Rotary2d::new(grid, self.head_dim, &device));
        let context = self.projection.forward(reference)
            + self.position(grid, &device)
            + self.reference_role.val();
        Ok(self.decode(
            self.target(masked_target, mask, grid, false),
            Some(context),
            rotary.as_ref(),
        ))
    }
    /// RGB completion accepts only sparse target features and reference features.
    /// Dense target features are reserved for the separate RI operation.
    pub fn reconstruct(
        &self,
        masked_target: Tensor<B, 3>,
        references: Vec<Tensor<B, 3>>,
        mask: &SparseTokenMask,
        grid: [usize; 2],
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>)> {
        let (cross, mae) = self.reconstruction_features(masked_target, references, mask, grid)?;
        Ok((self.cross_rgb.forward(cross), self.mae_rgb.forward(mae)))
    }
    /// Shared fusion features for alternative prediction spaces. The target
    /// input is sparse; dense teacher features are deliberately not accepted.
    pub fn reconstruction_features(
        &self,
        masked_target: Tensor<B, 3>,
        references: Vec<Tensor<B, 3>>,
        mask: &SparseTokenMask,
        grid: [usize; 2],
    ) -> Result<(Tensor<B, 3>, Tensor<B, 3>)> {
        let n = grid[0] * grid[1];
        ensure!(
            !references.is_empty() && references.len() <= 3,
            "expected 1..=3 references"
        );
        ensure!(
            mask.dense_len() == n && !mask.is_empty() && mask.len() < n,
            "invalid visible mask"
        );
        ensure!(
            masked_target.dims()[1] == mask.len(),
            "sparse target mismatch"
        );
        let device = masked_target.device();
        let rotary = (self.position_encoding == DecoderPosition::Rope2d)
            .then(|| crate::rotary::Rotary2d::new(grid, self.head_dim, &device));
        let context = self.reference_context(references, grid)?;
        let cross = self.decode(
            self.target(masked_target.clone(), mask, grid, false),
            Some(context),
            rotary.as_ref(),
        );
        let mae = self.decode(
            self.target(masked_target, mask, grid, true),
            None,
            rotary.as_ref(),
        );
        Ok((cross, mae))
    }
    fn reference_context(
        &self,
        references: Vec<Tensor<B, 3>>,
        grid: [usize; 2],
    ) -> Result<Tensor<B, 3>> {
        ensure!(
            !references.is_empty() && references.len() <= 3,
            "expected 1..=3 references"
        );
        let mut projected = Vec::new();
        for r in references {
            ensure!(r.dims()[1] == grid[0] * grid[1], "reference grid mismatch");
            let device = r.device();
            projected.push(
                self.projection.forward(r)
                    + self.position(grid, &device)
                    + self.reference_role.val(),
            );
        }
        Ok(Tensor::cat(projected, 1))
    }
    pub fn relative_improvement(
        &self,
        full_target: Tensor<B, 3>,
        references: Vec<Tensor<B, 3>>,
        grid: [usize; 2],
    ) -> Result<Tensor<B, 3>> {
        Ok(self
            .ri
            .forward(self.relative_improvement_features(full_target, references, grid)?))
    }
    /// Separate dense student branch for a learned error-improvement predictor.
    /// This must never feed back into masked completion.
    pub fn relative_improvement_features(
        &self,
        full_target: Tensor<B, 3>,
        references: Vec<Tensor<B, 3>>,
        grid: [usize; 2],
    ) -> Result<Tensor<B, 3>> {
        ensure!(
            full_target.dims()[1] == grid[0] * grid[1],
            "dense target grid mismatch"
        );
        let device = full_target.device();
        let rotary = (self.position_encoding == DecoderPosition::Rope2d)
            .then(|| crate::rotary::Rotary2d::new(grid, self.head_dim, &device));
        let full = self.projection.forward(full_target) + self.position(grid, &device);
        Ok(self.decode(
            full,
            Some(self.reference_context(references, grid)?),
            rotary.as_ref(),
        ))
    }
    /// Dense pair diagnostic: mean cross-attention across all heads/layers.
    /// Explicit local readout, not ZeroCo's full refinement protocol.
    pub fn correspondence_attention(
        &self,
        target: Tensor<B, 3>,
        reference: Tensor<B, 3>,
        grid: [usize; 2],
    ) -> Result<Tensor<B, 3>> {
        let n = grid[0] * grid[1];
        ensure!(
            target.dims()[1] == n && reference.dims()[1] == n,
            "dense pair grid mismatch"
        );
        let device = target.device();
        let rotary = (self.position_encoding == DecoderPosition::Rope2d)
            .then(|| crate::rotary::Rotary2d::new(grid, self.head_dim, &device));
        let context = self.reference_context(vec![reference], grid)?;
        let cross_rotary = if self.cross_view_rope {
            rotary.as_ref()
        } else {
            None
        };
        let mut x = self.projection.forward(target) + self.position(grid, &device);
        let mut sum = Tensor::zeros([x.dims()[0], n, n], &device);
        for block in &self.blocks {
            let q = block.n1.forward(x.clone());
            x = x + block.self_attn.forward(q.clone(), q, rotary.as_ref());
            let q = block.n2.forward(x.clone());
            let k = block.context_norm.forward(context.clone());
            sum = sum
                + block
                    .cross_attn
                    .probability_map(q.clone(), k.clone(), cross_rotary);
            x = x + block.cross_attn.forward(q, k, cross_rotary);
            let mlp = block.down.forward(activation::gelu(
                block.up.forward(block.n3.forward(x.clone())),
            ));
            x = x + mlp;
        }
        Ok(sum / self.blocks.len() as f64)
    }

    /// Dense auxiliary branch, separate from masked completion. Its full target
    /// features and teacher labels never enter the completion forward path.
    pub fn pair_training(
        &self,
        target: Tensor<B, 3>,
        reference: Tensor<B, 3>,
        grid: [usize; 2],
    ) -> Result<PairTraining<B>> {
        self.pair_forward(target, reference, grid, true)
    }

    /// Dense descriptors without allocating the unused attention diagnostics.
    pub fn pair_features(
        &self,
        target: Tensor<B, 3>,
        reference: Tensor<B, 3>,
        grid: [usize; 2],
    ) -> Result<Tensor<B, 3>> {
        Ok(self.pair_forward(target, reference, grid, false)?.features)
    }

    fn pair_forward(
        &self,
        target: Tensor<B, 3>,
        reference: Tensor<B, 3>,
        grid: [usize; 2],
        capture_logits: bool,
    ) -> Result<PairTraining<B>> {
        let n = grid[0] * grid[1];
        ensure!(
            target.dims()[1] == n && reference.dims()[1] == n,
            "dense pair grid mismatch"
        );
        let device = target.device();
        let rotary = (self.position_encoding == DecoderPosition::Rope2d)
            .then(|| crate::rotary::Rotary2d::new(grid, self.head_dim, &device));
        let cross_rotary = if self.cross_view_rope {
            rotary.as_ref()
        } else {
            None
        };
        let context = self.reference_context(vec![reference], grid)?;
        let mut x = self.projection.forward(target) + self.position(grid, &device);
        let mut logits = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let q = block.n1.forward(x.clone());
            x = x + block.self_attn.forward(q.clone(), q, rotary.as_ref());
            let q = block.n2.forward(x.clone());
            let k = block.context_norm.forward(context.clone());
            if capture_logits {
                logits.push(
                    block
                        .cross_attn
                        .logits_map(q.clone(), k.clone(), cross_rotary),
                );
            }
            x = x + block.cross_attn.forward(q, k, cross_rotary);
            x = x.clone()
                + block
                    .down
                    .forward(activation::gelu(block.up.forward(block.n3.forward(x))));
        }
        Ok(PairTraining {
            features: self.norm.forward(x),
            logits,
        })
    }

    /// Frozen-checkpoint causal diagnostic. Setting `cross_rope` false changes
    /// cross-view attention only; self-attention keeps its original image grid.
    /// No camera, correspondence, or other evaluation label is accepted here.
    pub fn pair_trace(
        &self,
        target: Tensor<B, 3>,
        reference: Tensor<B, 3>,
        grid: [usize; 2],
        cross_rope: bool,
    ) -> Result<PairTrace<B>> {
        let n = grid[0] * grid[1];
        ensure!(
            target.dims()[1] == n && reference.dims()[1] == n,
            "dense pair grid mismatch"
        );
        let device = target.device();
        let rotary = (self.position_encoding == DecoderPosition::Rope2d)
            .then(|| crate::rotary::Rotary2d::new(grid, self.head_dim, &device));
        let cross_position = if cross_rope { rotary.as_ref() } else { None };
        let context = self.reference_context(vec![reference], grid)?;
        let mut x = self.projection.forward(target) + self.position(grid, &device);
        let mut layers = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let q = block.n1.forward(x.clone());
            x = x + block.self_attn.forward(q.clone(), q, rotary.as_ref());
            let q = block.n2.forward(x.clone());
            let k = block.context_norm.forward(context.clone());
            layers.push(block.cross_attn.trace(q.clone(), k.clone(), cross_position));
            x = x + block.cross_attn.forward(q, k, cross_position);
            let mlp = block.down.forward(activation::gelu(
                block.up.forward(block.n3.forward(x.clone())),
            ));
            x = x + mlp;
        }
        Ok(PairTrace {
            features: self.norm.forward(x),
            layers,
        })
    }
    pub fn forward(
        &self,
        masked_target: Tensor<B, 3>,
        full_target: Tensor<B, 3>,
        references: Vec<Tensor<B, 3>>,
        mask: &SparseTokenMask,
        grid: [usize; 2],
    ) -> Result<Predictions<B>> {
        let n = grid[0] * grid[1];
        ensure!(
            !references.is_empty() && references.len() <= 3,
            "expected 1..=3 references"
        );
        ensure!(
            mask.dense_len() == n && !mask.is_empty() && mask.len() < n,
            "invalid visible mask"
        );
        ensure!(
            masked_target.dims()[1] == mask.len() && full_target.dims()[1] == n,
            "target token count mismatch"
        );
        let device = masked_target.device();
        let rotary = (self.position_encoding == DecoderPosition::Rope2d)
            .then(|| crate::rotary::Rotary2d::new(grid, self.head_dim, &device));
        let mut projected = Vec::new();
        for r in references {
            ensure!(
                r.dims() == full_target.dims(),
                "reference token shape mismatch"
            );
            projected.push(
                self.projection.forward(r)
                    + self.position(grid, &device)
                    + self.reference_role.val(),
            );
        }
        let context = Tensor::cat(projected, 1);
        let cross = self.decode(
            self.target(masked_target.clone(), mask, grid, false),
            Some(context.clone()),
            rotary.as_ref(),
        );
        let mae = self.decode(
            self.target(masked_target, mask, grid, true),
            None,
            rotary.as_ref(),
        );
        let full = self.projection.forward(full_target) + self.position(grid, &device);
        let ri = self.decode(full, Some(context), rotary.as_ref());
        Ok(Predictions {
            cross_rgb: self.cross_rgb.forward(cross),
            mae_rgb: self.mae_rgb.forward(mae),
            ri: self.ri.forward(ri),
        })
    }
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
