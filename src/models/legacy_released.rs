//! Strict Burn implementation of the released Gekko-L architecture, used as a
//! pretrained appearance path and as a parity reference. The downloaded weights
//! retain their CC-BY-NC-SA-4.0 terms; see docs/pilot-04.md for provenance.
use crate::rotary::Rotary2d;
use anyhow::{Result, ensure};
use burn::{
    module::{Module, ModuleMapper, Param},
    nn::{
        LayerNorm, LayerNormConfig, Linear, LinearConfig,
        conv::{Conv2d, Conv2dConfig},
    },
    tensor::{Int, Tensor, TensorData, activation, backend::Backend},
};
use burn_vjepa::SparseTokenMask;
use safetensors::{Dtype, SafeTensors};
use std::{collections::BTreeSet, path::Path};

struct Weights<'a> {
    tensors: SafeTensors<'a>,
    used: BTreeSet<String>,
}
impl<'a> Weights<'a> {
    fn tensor<B: Backend, const D: usize>(
        &mut self,
        name: &str,
        expected: [usize; D],
        device: &B::Device,
    ) -> Result<Tensor<B, D>> {
        ensure!(self.used.insert(name.to_owned()), "duplicate weight {name}");
        let tensor = self.tensors.tensor(name)?;
        ensure!(
            tensor.dtype() == Dtype::F32 && tensor.shape() == expected,
            "unexpected shape/dtype: {name}"
        );
        let data: Vec<f32> = tensor
            .data()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        ensure!(
            data.iter().all(|v| v.is_finite()),
            "nonfinite weight {name}"
        );
        Ok(Tensor::from_data(TensorData::new(data, expected), device))
    }
    fn linear<B: Backend>(
        &mut self,
        name: &str,
        input: usize,
        output: usize,
        d: &B::Device,
    ) -> Result<Linear<B>> {
        let mut linear = LinearConfig::new(input, output).init(d);
        linear.weight = Param::from_tensor(
            self.tensor(&format!("{name}.weight"), [output, input], d)?
                .transpose(),
        );
        linear.bias = Some(Param::from_tensor(self.tensor(
            &format!("{name}.bias"),
            [output],
            d,
        )?));
        Ok(linear)
    }
    fn norm<B: Backend>(
        &mut self,
        name: &str,
        width: usize,
        d: &B::Device,
    ) -> Result<LayerNorm<B>> {
        let mut norm = LayerNormConfig::new(width).with_epsilon(1e-6).init(d);
        norm.gamma = Param::from_tensor(self.tensor(&format!("{name}.weight"), [width], d)?);
        norm.beta = Some(Param::from_tensor(self.tensor(
            &format!("{name}.bias"),
            [width],
            d,
        )?));
        Ok(norm)
    }
}

#[derive(Module, Debug)]
struct Attention<B: Backend> {
    qkv: Linear<B>,
    proj: Linear<B>,
    q_norm: LayerNorm<B>,
    k_norm: LayerNorm<B>,
    heads: usize,
}
impl<B: Backend> Attention<B> {
    fn load(
        w: &mut Weights<'_>,
        p: &str,
        width: usize,
        heads: usize,
        d: &B::Device,
    ) -> Result<Self> {
        Ok(Self {
            qkv: w.linear(&format!("{p}.qkv"), width, width * 3, d)?,
            proj: w.linear(&format!("{p}.proj"), width, width, d)?,
            q_norm: w.norm(&format!("{p}.q_norm"), width / heads, d)?,
            k_norm: w.norm(&format!("{p}.k_norm"), width / heads, d)?,
            heads,
        })
    }
    fn forward(&self, x: Tensor<B, 3>, rope: &Rotary2d<B>) -> Tensor<B, 3> {
        let [b, n, d] = x.dims();
        let h = self.heads;
        let dh = d / h;
        let qkv = self
            .qkv
            .forward(x)
            .reshape([b, n, 3, h, dh])
            .permute([2, 0, 3, 1, 4]);
        let part = |i| qkv.clone().slice_dim(0, i..i + 1).squeeze_dim::<4>(0);
        let q = rope.apply(self.q_norm.forward(part(0)));
        let k = rope.apply(self.k_norm.forward(part(1)));
        let v = part(2);
        let attention = activation::softmax(q.matmul(k.swap_dims(2, 3)) / (dh as f64).sqrt(), 3);
        self.proj
            .forward(attention.matmul(v).swap_dims(1, 2).reshape([b, n, d]))
    }
}
#[derive(Module, Debug)]
struct Mlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}
impl<B: Backend> Mlp<B> {
    fn load(w: &mut Weights<'_>, p: &str, width: usize, d: &B::Device) -> Result<Self> {
        Ok(Self {
            fc1: w.linear(&format!("{p}.fc1"), width, width * 4, d)?,
            fc2: w.linear(&format!("{p}.fc2"), width * 4, width, d)?,
        })
    }
    fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.fc2.forward(activation::gelu(self.fc1.forward(x)))
    }
}
#[derive(Module, Debug)]
struct EncoderBlock<B: Backend> {
    attn: Attention<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    mlp: Mlp<B>,
}
impl<B: Backend> EncoderBlock<B> {
    fn load(w: &mut Weights<'_>, p: &str, d: &B::Device) -> Result<Self> {
        Ok(Self {
            attn: Attention::load(w, &format!("{p}.attn"), 1024, 16, d)?,
            norm1: w.norm(&format!("{p}.norm1"), 1024, d)?,
            norm2: w.norm(&format!("{p}.norm2"), 1024, d)?,
            mlp: Mlp::load(w, &format!("{p}.mlp"), 1024, d)?,
        })
    }
    fn forward(&self, x: Tensor<B, 3>, rope: &Rotary2d<B>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.norm1.forward(x), rope);
        x.clone() + self.mlp.forward(self.norm2.forward(x))
    }
}
#[derive(Module, Debug)]
struct CrossAttention<B: Backend> {
    q: Linear<B>,
    kv: Linear<B>,
    proj: Linear<B>,
}
impl<B: Backend> CrossAttention<B> {
    fn load(w: &mut Weights<'_>, p: &str, d: &B::Device) -> Result<Self> {
        Ok(Self {
            q: w.linear(&format!("{p}.q"), 768, 768, d)?,
            kv: w.linear(&format!("{p}.kv"), 768, 1536, d)?,
            proj: w.linear(&format!("{p}.proj"), 768, 768, d)?,
        })
    }
    fn forward(&self, x: Tensor<B, 3>, y: Tensor<B, 3>, rope: &Rotary2d<B>) -> Tensor<B, 3> {
        let [b, n, _] = x.dims();
        let m = y.dims()[1];
        let q = rope.apply(self.q.forward(x).reshape([b, n, 12, 64]).swap_dims(1, 2));
        let kv = self
            .kv
            .forward(y)
            .reshape([b, m, 2, 12, 64])
            .permute([2, 0, 3, 1, 4]);
        let k = rope.apply(kv.clone().slice_dim(0, 0..1).squeeze_dim::<4>(0));
        let v = kv.slice_dim(0, 1..2).squeeze_dim::<4>(0);
        self.proj.forward(
            activation::softmax(q.matmul(k.swap_dims(2, 3)) / 8., 3)
                .matmul(v)
                .swap_dims(1, 2)
                .reshape([b, n, 768]),
        )
    }
}
#[derive(Module, Debug)]
struct DecoderBlock<B: Backend> {
    attn: Attention<B>,
    cross_attn: CrossAttention<B>,
    norm1: LayerNorm<B>,
    norm2: LayerNorm<B>,
    norm3: LayerNorm<B>,
    norm_y: LayerNorm<B>,
    mlp: Mlp<B>,
}
impl<B: Backend> DecoderBlock<B> {
    fn load(w: &mut Weights<'_>, p: &str, d: &B::Device) -> Result<Self> {
        Ok(Self {
            attn: Attention::load(w, &format!("{p}.attn"), 768, 12, d)?,
            cross_attn: CrossAttention::load(w, &format!("{p}.cross_attn"), d)?,
            norm1: w.norm(&format!("{p}.norm1"), 768, d)?,
            norm2: w.norm(&format!("{p}.norm2"), 768, d)?,
            norm3: w.norm(&format!("{p}.norm3"), 768, d)?,
            norm_y: w.norm(&format!("{p}.norm_y"), 768, d)?,
            mlp: Mlp::load(w, &format!("{p}.mlp"), 768, d)?,
        })
    }
    fn forward(&self, x: Tensor<B, 3>, y: Tensor<B, 3>, rope: &Rotary2d<B>) -> Tensor<B, 3> {
        let x = x.clone() + self.attn.forward(self.norm1.forward(x), rope);
        let x = x.clone()
            + self
                .cross_attn
                .forward(self.norm2.forward(x), self.norm_y.forward(y), rope);
        x.clone() + self.mlp.forward(self.norm3.forward(x))
    }
}

/// Frozen appearance encoder. Nonoverlapping patch embedding precedes sparse
/// selection; masked pixels are removed before every attention operation.
#[derive(Module, Debug)]
pub struct ReleasedEncoder<B: Backend> {
    patch: Conv2d<B>,
    patch_norm: LayerNorm<B>,
    blocks: Vec<EncoderBlock<B>>,
    norm: LayerNorm<B>,
}
impl<B: Backend> ReleasedEncoder<B> {
    pub fn forward(&self, rgb: Tensor<B, 4>, mask: Option<&SparseTokenMask>) -> Tensor<B, 3> {
        let [b, _, h, w] = rgb.dims();
        let d = rgb.device();
        let grid = [h / 16, w / 16];
        let mut rope = Rotary2d::new(grid, 64, &d);
        let mut tokens = self.patch_norm.forward(
            self.patch
                .forward(normalize(rgb))
                .reshape([b, 1024, grid[0] * grid[1]])
                .swap_dims(1, 2),
        );
        if let Some(mask) = mask {
            let ids = mask_indices::<B>(mask, &d);
            tokens = tokens.select(1, ids.clone());
            rope = rope.select(ids);
        }
        for block in &self.blocks {
            tokens = block.forward(tokens, &rope);
        }
        self.norm.forward(tokens)
    }
}
pub fn normalize<B: Backend>(rgb: Tensor<B, 4>) -> Tensor<B, 4> {
    let d = rgb.device();
    let mean = Tensor::from_data(TensorData::new(vec![0.485, 0.456, 0.406], [1, 3, 1, 1]), &d);
    let std = Tensor::from_data(TensorData::new(vec![0.229, 0.224, 0.225], [1, 3, 1, 1]), &d);
    (rgb - mean) / std
}
fn mask_indices<B: Backend>(mask: &SparseTokenMask, d: &B::Device) -> Tensor<B, 1, Int> {
    Tensor::from_data(
        TensorData::new(
            mask.indices().iter().map(|&v| v as i64).collect(),
            [mask.len()],
        ),
        d,
    )
}

#[derive(Module, Debug)]
pub struct ReleasedDecoder<B: Backend> {
    projection: Linear<B>,
    blocks: Vec<DecoderBlock<B>>,
    norm: LayerNorm<B>,
    pub cross_head: Linear<B>,
    pub mae_head: Linear<B>,
    cross_mask: Param<Tensor<B, 3>>,
    mae_mask: Param<Tensor<B, 3>>,
}
impl<B: Backend> ReleasedDecoder<B> {
    /// Fine-tune only the final decoder blocks and final normalization. The
    /// pretrained encoder and RGB/RI heads remain frozen.
    pub fn train_tail(mut self, count: usize) -> Self {
        assert!(count <= self.blocks.len());
        struct Trainable;
        impl<B: Backend> ModuleMapper<B> for Trainable {
            fn map_float<const D: usize>(
                &mut self,
                param: Param<Tensor<B, D>>,
            ) -> Param<Tensor<B, D>> {
                let (id, tensor, mapper) = param.consume();
                Param::from_mapped_value(id, tensor.set_require_grad(true), mapper)
            }
        }
        let first = self.blocks.len() - count;
        self.blocks = self
            .blocks
            .into_iter()
            .enumerate()
            .map(|(i, b)| {
                if i >= first {
                    b.map(&mut Trainable)
                } else {
                    b.no_grad()
                }
            })
            .collect();
        if count > 0 {
            self.norm = self.norm.map(&mut Trainable);
        }
        self
    }
    pub fn optimization_markers(&self) -> [Tensor<B, 2>; 2] {
        [
            self.blocks[0].attn.qkv.weight.val().detach(),
            self.blocks.last().unwrap().attn.qkv.weight.val().detach(),
        ]
    }
    pub fn features(
        &self,
        target: Tensor<B, 3>,
        reference: Option<Tensor<B, 3>>,
        mask: Option<&SparseTokenMask>,
        grid: [usize; 2],
        mae: bool,
    ) -> Tensor<B, 3> {
        let d = target.device();
        let b = target.dims()[0];
        let n = grid[0] * grid[1];
        let rope = Rotary2d::new(grid, 64, &d);
        let mut x = self.projection.forward(target);
        if let Some(mask) = mask {
            let k = mask.len();
            let token = if mae {
                self.mae_mask.val()
            } else {
                self.cross_mask.val()
            }
            .expand([b, 1, 768]);
            let mut indices = vec![k as i64; n];
            for (i, &j) in mask.indices().iter().enumerate() {
                indices[j] = i as i64;
            }
            x = Tensor::cat(vec![x, token], 1).select(
                1,
                Tensor::<B, 1, Int>::from_data(TensorData::new(indices, [n]), &d),
            );
        }
        let context = reference.map(|r| self.projection.forward(r));
        for block in &self.blocks {
            x = block.forward(
                x.clone(),
                context.clone().unwrap_or_else(|| x.clone()),
                &rope,
            );
        }
        self.norm.forward(x)
    }
}

/// Require exactly the 712 tensors in the published ViT-L package. Each tensor
/// is shape checked and consumed once; linear weights are explicitly transposed.
pub fn load<B: Backend>(
    path: &Path,
    d: &B::Device,
) -> Result<(ReleasedEncoder<B>, ReleasedDecoder<B>)> {
    let bytes = std::fs::read(path)?;
    let mut w = Weights {
        tensors: SafeTensors::deserialize(&bytes)?,
        used: BTreeSet::new(),
    };
    ensure!(
        w.tensors.len() == 712,
        "expected released Gekko-L package (712 tensors)"
    );
    let mut patch = Conv2dConfig::new([3, 1024], [16, 16])
        .with_stride([16, 16])
        .init(d);
    patch.weight = Param::from_tensor(w.tensor("patch_embed.proj.weight", [1024, 3, 16, 16], d)?);
    patch.bias = Some(Param::from_tensor(w.tensor(
        "patch_embed.proj.bias",
        [1024],
        d,
    )?));
    let encoder = ReleasedEncoder {
        patch,
        patch_norm: w.norm("patch_embed.norm", 1024, d)?,
        blocks: (0..24)
            .map(|i| EncoderBlock::load(&mut w, &format!("enc_blocks.{i}"), d))
            .collect::<Result<_>>()?,
        norm: w.norm("enc_norm", 1024, d)?,
    };
    let decoder = ReleasedDecoder {
        projection: w.linear("decoder_embed", 1024, 768, d)?,
        blocks: (0..12)
            .map(|i| DecoderBlock::load(&mut w, &format!("dec_blocks.{i}"), d))
            .collect::<Result<_>>()?,
        norm: w.norm("dec_norm", 768, d)?,
        cross_head: w.linear("croco_head", 768, 1024, d)?,
        mae_head: w.linear("mae_head", 768, 768, d)?,
        cross_mask: Param::from_tensor(w.tensor("croco_mask_token", [1, 1, 768], d)?),
        mae_mask: Param::from_tensor(w.tensor("mae_mask_token", [1, 1, 768], d)?),
    };
    ensure!(w.used.len() == 712, "unconsumed released tensors");
    Ok((encoder, decoder))
}
