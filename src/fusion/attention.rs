//! Attention kernels and transformer blocks shared by pair/set decoding.
use super::decoder::AttentionTrace;
use burn::{
    module::Module,
    nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig},
    tensor::{Tensor, activation, backend::Backend},
};
#[derive(Module, Debug)]
pub(super) struct Attention<B: Backend> {
    pub(super) q: Linear<B>,
    pub(super) k: Linear<B>,
    pub(super) v: Linear<B>,
    pub(super) out: Linear<B>,
    pub(super) q_norm: Option<LayerNorm<B>>,
    pub(super) k_norm: Option<LayerNorm<B>>,
    pub(super) heads: usize,
    #[module(skip)]
    pub(super) stable_attention: bool,
}
impl<B: Backend> Attention<B> {
    pub(super) fn new(d: usize, heads: usize, device: &B::Device) -> Self {
        Self {
            q: LinearConfig::new(d, d).init(device),
            k: LinearConfig::new(d, d).init(device),
            v: LinearConfig::new(d, d).init(device),
            out: LinearConfig::new(d, d).init(device),
            q_norm: None,
            k_norm: None,
            heads,
            stable_attention: false,
        }
    }
    pub(super) fn forward(
        &self,
        query: Tensor<B, 3>,
        context: Tensor<B, 3>,
        rotary: Option<&crate::rotary::Rotary2d<B>>,
    ) -> Tensor<B, 3> {
        let [b, n, d] = query.dims();
        let m = context.dims()[1];
        let h = self.heads;
        let dh = d / h;
        let mut q = self.q.forward(query).reshape([b, n, h, dh]).swap_dims(1, 2);
        let mut k = self
            .k
            .forward(context.clone())
            .reshape([b, m, h, dh])
            .swap_dims(1, 2);
        if let Some(norm) = &self.q_norm {
            q = norm.forward(q);
        }
        if let Some(norm) = &self.k_norm {
            k = norm.forward(k);
        }
        if let Some(rotary) = rotary {
            q = rotary.apply(q);
            k = rotary.apply(k);
        }
        let v = self
            .v
            .forward(context)
            .reshape([b, m, h, dh])
            .swap_dims(1, 2);
        let logits = q.matmul(k.swap_dims(2, 3)) / (dh as f64).sqrt();
        let aggregate = if self.stable_attention {
            // Keep softmax and the reference reduction away from TF32 rounding.
            // Projection storage remains unchanged. This is an explicit numeric
            // policy, not a change to learned parameters or reference ordering.
            let dtype = burn::tensor::FloatDType::from(v.dtype());
            activation::softmax(logits.cast(burn::tensor::FloatDType::F64), 3)
                .matmul(v.cast(burn::tensor::FloatDType::F64))
                .cast(dtype)
        } else {
            activation::softmax(logits, 3).matmul(v)
        };
        self.out
            .forward(aggregate.swap_dims(1, 2).reshape([b, n, d]))
    }
    /// Read-only head-mean probability map using forward's queries and keys.
    pub(super) fn probability_map(
        &self,
        query: Tensor<B, 3>,
        context: Tensor<B, 3>,
        rotary: Option<&crate::rotary::Rotary2d<B>>,
    ) -> Tensor<B, 3> {
        let [b, n, d] = query.dims();
        let m = context.dims()[1];
        let h = self.heads;
        let dh = d / h;
        let mut q = self.q.forward(query).reshape([b, n, h, dh]).swap_dims(1, 2);
        let mut k = self
            .k
            .forward(context)
            .reshape([b, m, h, dh])
            .swap_dims(1, 2);
        if let Some(norm) = &self.q_norm {
            q = norm.forward(q);
        }
        if let Some(norm) = &self.k_norm {
            k = norm.forward(k);
        }
        if let Some(rotary) = rotary {
            q = rotary.apply(q);
            k = rotary.apply(k);
        }
        let logits = q.matmul(k.swap_dims(2, 3)) / (dh as f64).sqrt();
        let dtype = burn::tensor::FloatDType::from(logits.dtype());
        let map = if self.stable_attention {
            activation::softmax(logits.cast(burn::tensor::FloatDType::F64), 3).cast(dtype)
        } else {
            activation::softmax(logits, 3)
        };
        map.mean_dim(1).reshape([b, n, m])
    }

    pub(super) fn trace(
        &self,
        query: Tensor<B, 3>,
        context: Tensor<B, 3>,
        rotary: Option<&crate::rotary::Rotary2d<B>>,
    ) -> AttentionTrace<B> {
        let [b, n, d] = query.dims();
        let m = context.dims()[1];
        let (h, dh) = (self.heads, d / self.heads);
        let mut q = self.q.forward(query).reshape([b, n, h, dh]).swap_dims(1, 2);
        let mut k = self
            .k
            .forward(context)
            .reshape([b, m, h, dh])
            .swap_dims(1, 2);
        if let Some(norm) = &self.q_norm {
            q = norm.forward(q);
        }
        if let Some(norm) = &self.k_norm {
            k = norm.forward(k);
        }
        let content = q.clone().matmul(k.clone().swap_dims(2, 3)) / (dh as f64).sqrt();
        let centered = (q.clone() - q.clone().mean_dim(2))
            .matmul((k.clone() - k.clone().mean_dim(2)).swap_dims(2, 3))
            / (dh as f64).sqrt();
        if let Some(rotary) = rotary {
            q = rotary.apply(q);
            k = rotary.apply(k);
        }
        let logits = q.matmul(k.swap_dims(2, 3)) / (dh as f64).sqrt();
        let probability = if self.stable_attention {
            let dtype = burn::tensor::FloatDType::from(logits.dtype());
            activation::softmax(logits.clone().cast(burn::tensor::FloatDType::F64), 3).cast(dtype)
        } else {
            activation::softmax(logits.clone(), 3)
        };
        let head_mean = |x: Tensor<B, 4>| x.mean_dim(1).reshape([b, n, m]);
        AttentionTrace {
            probability: head_mean(probability),
            logits: head_mean(logits),
            content: head_mean(content),
            centered_content: head_mean(centered),
        }
    }

    // Training builds only the requested score branch. Constructing unused
    // trace alternatives here would retain disconnected autodiff graphs.
    pub(super) fn logits_map(
        &self,
        query: Tensor<B, 3>,
        context: Tensor<B, 3>,
        rotary: Option<&crate::rotary::Rotary2d<B>>,
    ) -> Tensor<B, 3> {
        let [b, n, d] = query.dims();
        let m = context.dims()[1];
        let (h, dh) = (self.heads, d / self.heads);
        let mut q = self.q.forward(query).reshape([b, n, h, dh]).swap_dims(1, 2);
        let mut k = self
            .k
            .forward(context)
            .reshape([b, m, h, dh])
            .swap_dims(1, 2);
        if let Some(norm) = &self.q_norm {
            q = norm.forward(q);
        }
        if let Some(norm) = &self.k_norm {
            k = norm.forward(k);
        }
        if let Some(rotary) = rotary {
            q = rotary.apply(q);
            k = rotary.apply(k);
        }
        (q.matmul(k.swap_dims(2, 3)) / (dh as f64).sqrt())
            .mean_dim(1)
            .reshape([b, n, m])
    }
}

#[derive(Module, Debug)]
pub(super) struct Block<B: Backend> {
    pub(super) self_attn: Attention<B>,
    pub(super) cross_attn: Attention<B>,
    pub(super) n1: LayerNorm<B>,
    pub(super) n2: LayerNorm<B>,
    pub(super) context_norm: LayerNorm<B>,
    pub(super) n3: LayerNorm<B>,
    pub(super) up: Linear<B>,
    pub(super) down: Linear<B>,
}
impl<B: Backend> Block<B> {
    pub(super) fn new(d: usize, heads: usize, device: &B::Device) -> Self {
        Self {
            self_attn: Attention::new(d, heads, device),
            cross_attn: Attention::new(d, heads, device),
            n1: LayerNormConfig::new(d).init(device),
            n2: LayerNormConfig::new(d).init(device),
            context_norm: LayerNormConfig::new(d).init(device),
            n3: LayerNormConfig::new(d).init(device),
            up: LinearConfig::new(d, d * 4).init(device),
            down: LinearConfig::new(d * 4, d).init(device),
        }
    }
    pub(super) fn forward(
        &self,
        mut x: Tensor<B, 3>,
        reference: Option<Tensor<B, 3>>,
        rotary: Option<&crate::rotary::Rotary2d<B>>,
        mae_context_before_self: bool,
        cross_view_rope: bool,
    ) -> Tensor<B, 3> {
        let cross_rotary = if reference.is_some() && !cross_view_rope {
            None
        } else {
            rotary
        };
        let reference = if mae_context_before_self {
            Some(reference.unwrap_or_else(|| x.clone()))
        } else {
            reference
        };
        let q = self.n1.forward(x.clone());
        x = x + self.self_attn.forward(q.clone(), q, rotary);
        // MAE uses target self-attention in the shared cross-attention slot.
        let context = reference.unwrap_or_else(|| x.clone());
        let cross = self.cross_attn.forward(
            self.n2.forward(x.clone()),
            self.context_norm.forward(context),
            cross_rotary,
        );
        x = x + cross;
        let mlp = self.down.forward(activation::gelu(
            self.up.forward(self.n3.forward(x.clone())),
        ));
        x + mlp
    }
}
