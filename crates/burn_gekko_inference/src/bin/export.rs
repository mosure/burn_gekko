//! Package existing, audited float32 records; no training or numerical conversion.
use anyhow::{Context, Result, ensure};
use burn_gekko_inference::{Bundle, ModelFile, bundle::digest};
use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    checkpoint: PathBuf,
    heads: PathBuf,
    encoder_manifest: PathBuf,
    id: String,
}
fn pin(name: String, bytes: &[u8], root: &Path) -> Result<ModelFile> {
    fs::write(root.join(&name), bytes)?;
    Ok(ModelFile {
        name,
        bytes: bytes.len(),
        sha256: digest(bytes),
    })
}
fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().collect();
    ensure!(
        args.len() == 5 && args[1] == "--config" && args[3] == "--output",
        "usage: gekko-demo-export --config recipe.toml --output directory"
    );
    let c: Config = toml::from_str(&fs::read_to_string(&args[2])?)?;
    let out = Path::new(&args[4]);
    ensure!(!out.exists(), "preserve existing model export");
    let metadata: serde_json::Value =
        serde_json::from_slice(&fs::read(c.checkpoint.join("metadata.json"))?)?;
    let head_meta: serde_json::Value =
        serde_json::from_slice(&fs::read(c.heads.join("metadata.json"))?)?;
    let run: toml::Value = toml::from_str(&fs::read_to_string(
        c.checkpoint
            .parent()
            .context("checkpoint parent")?
            .join("config.toml"),
    )?)?;
    let source: serde_json::Value = serde_json::from_slice(&fs::read(&c.encoder_manifest)?)?;
    let foundation = fs::read(c.checkpoint.join("model.mpk"))?;
    let camera = fs::read(c.heads.join("camera.mpk"))?;
    let rgb = fs::read(c.heads.join("rgb.mpk"))?;
    let foundation_sha256 = digest(&foundation);
    ensure!(
        metadata["model_sha256"] == foundation_sha256
            && metadata["noncommercial_weight_dependencies"] == serde_json::json!([])
            && head_meta["checkpoint_sha256"] == foundation_sha256
            && head_meta["files"]["camera"] == digest(&camera)
            && head_meta["files"]["rgb"] == digest(&rgb),
        "unaudited/mixed inference weights"
    );
    fs::create_dir_all(out)?;
    let parts = foundation
        .chunks(32 * 1024 * 1024)
        .enumerate()
        .map(|(i, b)| pin(format!("foundation-{i:03}.mpk.part"), b, out))
        .collect::<Result<Vec<_>>>()?;
    let integer = |k: &str| -> Result<usize> {
        Ok(run[k].as_integer().context("architecture integer")? as usize)
    };
    let encoder: burn_vjepa::VJepaConfig = serde_json::from_value(source["jepa_config"].clone())?;
    let bundle=Bundle{
        schema:1,id:c.id, foundation_sha256, camera_sha256:digest(&camera),rgb_sha256:digest(&rgb),precision:"f32".into(),image_size:256,
        decoder:burn_gekko::model::DecoderConfig{encoder_dim:encoder.encoder.embed_dim,width:integer("decoder_width")?,depth:integer("decoder_depth")?,heads:integer("decoder_heads")?,patch:16},
        encoder, spatial_descriptor:run["spatial_descriptor"].clone().try_into()?,spatial_input_layer:integer("spatial_input_layer")?,head_width:64,mask_ratio:0.9,mask_seed:857,
        foundation:parts,camera:pin("camera.mpk".into(),&camera,out)?,rgb:pin("rgb.mpk".into(),&rgb,out)?,
        license:"MIT OR Apache-2.0; includes MIT V-JEPA 2.1 encoder ancestry. See LICENSE.vjepa.txt and NOTICE.md.".into(),
        qualification:"Research checkpoint; RGB remains blurry and calibration overfits. Browser and new Zeroverse rendering are interactive demonstrations, not the offline benchmark protocol. Camera input is a dense RGB pair; completion uses sparse target pixels only. No pretrained noncommercial weights.".into(),
    };
    bundle.validate()?;
    fs::write(out.join("manifest.toml"), toml::to_string_pretty(&bundle)?)?;
    for name in ["LICENSE.vjepa.txt", "NOTICE.md"] {
        fs::copy(c.checkpoint.join(name), out.join(name))?;
    }
    fs::write(out.join("LICENSE-MIT"), include_str!("../../LICENSE-MIT"))?;
    fs::write(
        out.join("LICENSE-APACHE"),
        include_str!("../../LICENSE-APACHE"),
    )?;
    fs::copy(
        c.checkpoint.join("metadata.json"),
        out.join("foundation-provenance.json"),
    )?;
    fs::copy(
        c.heads.join("metadata.json"),
        out.join("heads-provenance.json"),
    )?;
    println!(
        "Exported {} exact float32 foundation bytes in {} parts",
        foundation.len(),
        bundle.foundation.len()
    );
    Ok(())
}
