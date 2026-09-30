//! Export the exact loaded encoder and dense/sparse image outputs for independent parity.
use anyhow::{Result, ensure};
use burn::{
    nn::{LayerNorm, Linear},
    tensor::{Tensor, TensorData, backend::Backend},
};
use burn_gekko_train::{
    encoder::normalize,
    train::{EncoderSource, load_encoder},
};
use burn_vjepa::SparseTokenMask;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

struct Export {
    root: PathBuf,
    shapes: BTreeMap<String, Vec<usize>>,
}
impl Export {
    fn tensor<B: Backend, const D: usize>(&mut self, name: &str, x: Tensor<B, D>) -> Result<()> {
        self.shapes.insert(name.into(), x.dims().to_vec());
        let values = x
            .into_data()
            .to_vec::<f32>()
            .map_err(|e| anyhow::anyhow!("{e:?}"))?;
        let bytes: Vec<_> = values.iter().flat_map(|x| x.to_le_bytes()).collect();
        fs::write(self.root.join(format!("{name}.f32")), bytes)?;
        Ok(())
    }
    fn linear<B: Backend>(&mut self, name: &str, layer: &Linear<B>) -> Result<()> {
        // PyTorch Linear uses [out, in], Burn uses [in, out].
        self.tensor(&format!("{name}.weight"), layer.weight.val().transpose())?;
        if let Some(bias) = &layer.bias {
            self.tensor(&format!("{name}.bias"), bias.val())?;
        }
        Ok(())
    }
    fn norm<B: Backend>(&mut self, name: &str, layer: &LayerNorm<B>) -> Result<()> {
        self.tensor(&format!("{name}.weight"), layer.gamma.val())?;
        if let Some(beta) = &layer.beta {
            self.tensor(&format!("{name}.bias"), beta.val())?;
        }
        Ok(())
    }
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 3 || args.len() == 4,
        "usage: encoder_audit PACKAGE OUTPUT [cpu]"
    );
    if args.get(3).is_some_and(|s| s == "cpu") {
        return run::<burn::backend::NdArray<f32>>(&args, &[32, 256]);
    }
    #[cfg(feature = "cuda")]
    return run::<burn::backend::Cuda<f32, i32>>(&args, &[32, 256, 384]);
    #[cfg(not(feature = "cuda"))]
    anyhow::bail!("CUDA feature required, or select cpu")
}
fn run<B: Backend>(args: &[String], sizes: &[usize]) -> Result<()> {
    let root = Path::new(&args[2]);
    ensure!(
        root.starts_with(".data") && !root.exists(),
        "choose new .data output"
    );
    fs::create_dir_all(root)?;
    let device = Default::default();
    let (encoder, config, identity) = load_encoder::<B>(
        &EncoderSource::Burnpack {
            directory: args[1].clone().into(),
        },
        29,
        &device,
    )?;
    let mut export = Export {
        root: root.into(),
        shapes: BTreeMap::new(),
    };
    export.tensor(
        "patch_embed.proj.weight",
        encoder.patch_embed.proj.weight.val(),
    )?;
    export.tensor(
        "patch_embed.proj.bias",
        encoder.patch_embed.proj.bias.as_ref().unwrap().val(),
    )?;
    export.tensor(
        "patch_embed_img.proj.weight",
        encoder.image_patch_embed.proj.weight.val(),
    )?;
    export.tensor(
        "patch_embed_img.proj.bias",
        encoder.image_patch_embed.proj.bias.as_ref().unwrap().val(),
    )?;
    export.tensor(
        "img_mod_embed",
        encoder.image_mod_embed.val().unsqueeze_dim::<3>(0),
    )?;
    export.tensor(
        "video_mod_embed",
        encoder.video_mod_embed.val().unsqueeze_dim::<3>(0),
    )?;
    for (i, block) in encoder.blocks.iter().enumerate() {
        export.norm(&format!("blocks.{i}.norm1"), &block.norm1)?;
        export.norm(&format!("blocks.{i}.norm2"), &block.norm2)?;
        export.linear(&format!("blocks.{i}.attn.qkv"), &block.attn.qkv)?;
        export.linear(&format!("blocks.{i}.attn.proj"), &block.attn.proj)?;
        export.linear(&format!("blocks.{i}.mlp.fc1"), &block.mlp.fc1)?;
        export.linear(&format!("blocks.{i}.mlp.fc2"), &block.mlp.fc2)?;
    }
    for (i, norm) in encoder.norms_block.iter().enumerate() {
        export.norm(&format!("norms_block.{i}"), norm)?;
    }
    let weights = export.shapes.clone();
    for &size in sizes {
        let values: Vec<f32> = (0..3 * size * size)
            .map(|i| (i % 251) as f32 / 250.0)
            .collect();
        let image = normalize(
            Tensor::<B, 4>::from_data(TensorData::new(values, [1, 3, size, size]), &device),
            &config,
        );
        export.tensor(&format!("input-{size}"), image.clone())?;
        let n = (size / 16) * (size / 16);
        let indices: Vec<_> = (0..n).filter(|i| i % 4 == 0).collect();
        let mask = SparseTokenMask::new(indices.clone(), n)?;
        export.tensor(
            &format!("dense-{size}"),
            encoder.forward_image(image.clone(), None).tokens,
        )?;
        export.tensor(
            &format!("sparse-{size}"),
            encoder.forward_image(image, Some(&mask)).tokens,
        )?;
        burn_gekko_data::write_json(&root.join(format!("mask-{size}.json")), &indices)?;
    }
    burn_gekko_data::write_json(
        &root.join("manifest.json"),
        &serde_json::json!({"encoder_id":identity,"config":config,"weights":weights,"tensors":export.shapes}),
    )?;
    Ok(())
}
