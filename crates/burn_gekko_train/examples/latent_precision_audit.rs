//! Focused evaluation of reference-order numerical sensitivity, not a training run.
use anyhow::{Result, ensure};
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Tensor, backend::Backend},
};
use burn_gekko_data::{DatasetManifest, Split, fingerprint, load_rgb, sha256_file, write_json};
use burn_gekko_train::{
    e2e_pilot::host_batch,
    encoder::{normalize, visible_mask},
    latent::{LatentModel, normalize_teacher, token_mse},
    latent_pilot::LatentConfig,
    model::DecoderConfig,
    train::load_encoder,
};
use clap::Parser;
use std::{fs, path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    checkpoint: PathBuf,
    #[arg(long)]
    output: PathBuf,
}
fn max<B: Backend>(x: Tensor<B, 3>) -> Result<f32> {
    Ok(burn_gekko_train::latent_eval::values(x.abs().max())?[0])
}
fn run<B: Backend>(a: Args) -> Result<()> {
    let c: LatentConfig = burn_gekko_data::read_config(&a.config)?;
    c.validate()?;
    ensure!(!a.output.exists(), "audit output exists");
    let parent = a.output.parent().unwrap();
    fs::create_dir_all(parent)?;
    ensure!(
        fs::canonicalize(parent)?.starts_with(fs::canonicalize(".data")?),
        "output must be under .data"
    );
    let meta: serde_json::Value =
        serde_json::from_slice(&fs::read(a.checkpoint.join("metadata.json"))?)?;
    ensure!(
        meta["noncommercial_weight_dependencies"] == serde_json::json!([]),
        "noncommercial checkpoint prohibited"
    );
    ensure!(
        meta["model_sha256"] == sha256_file(&a.checkpoint.join("model.mpk"))?,
        "checkpoint checksum mismatch"
    );
    let manifest: DatasetManifest =
        serde_json::from_slice(&fs::read(c.dataset.join("manifest.json"))?)?;
    ensure!(
        meta["dataset_id"] == manifest.dataset_id
            && manifest.dataset_id
                == fingerprint(&(
                    manifest.schema,
                    &manifest.generator,
                    &manifest.binary_sha256,
                    &manifest.config
                ))?,
        "dataset identity mismatch"
    );
    let entry = manifest
        .scenes
        .iter()
        .find(|e| e.split == Split::Validation)
        .unwrap();
    ensure!(
        std::path::Path::new(&entry.file).components().count() == 1,
        "invalid shard path"
    );
    let file = c.dataset.join("raw").join(&entry.file);
    ensure!(
        sha256_file(&file)? == entry.sha256,
        "sample checksum mismatch"
    );
    let scenes = vec![load_rgb(&file)?];
    let d = Default::default();
    let (encoder, ec, id) = load_encoder::<B>(&c.teacher, c.seed, &d)?;
    ensure!(meta["teacher_id"] == id, "teacher identity mismatch");
    let teacher = encoder.clone();
    let mut model = LatentModel::new(
        encoder,
        ec.clone(),
        &DecoderConfig {
            encoder_dim: ec.encoder.embed_dim,
            width: c.decoder_width,
            depth: c.decoder_depth,
            heads: c.decoder_heads,
            patch: 16,
        },
        &d,
    )?
    .load_file(
        a.checkpoint.join("model"),
        &NamedMpkFileRecorder::<FullPrecisionSettings>::default(),
        &d,
    )?;
    let mask = visible_mask(
        scenes[0].height / 16 * (scenes[0].width / 16),
        c.mask_ratio,
        c.seed ^ 0x4c4154454e54,
        0,
    )?;
    let (rgb, refs) = host_batch::<B>(&scenes, &[(0, 0)], c.references, &d);
    let target = normalize_teacher(
        teacher
            .forward_image(normalize(rgb.clone(), &ec), None)
            .tokens,
    );
    let mut reversed = refs.clone();
    reversed.reverse();
    let visible =
        burn_gekko_train::matching::visibility(scenes[0].height, scenes[0].width, &mask, &d);
    let altered = rgb.clone() * visible.clone() + (visible.neg() + 1.) * 0.913;
    let mut rows = Vec::new();
    let mut original = None;
    for stable in [false, true] {
        model.fusion.decoder = model.fusion.decoder.with_stable_attention(stable);
        let output = model.predict(rgb.clone(), &refs, &mask)?.cross;
        let permutation = model.predict(rgb.clone(), &reversed, &mask)?.cross;
        let hidden = model.predict(altered.clone(), &refs, &mask)?.cross;
        let delta = output.clone() - permutation;
        let mse = burn_gekko_train::latent_eval::values(
            burn_gekko_train::latent::select_tokens(
                token_mse(output.clone(), target.clone()),
                &mask,
                true,
            )
            .mean(),
        )?[0];
        let policy_delta = original
            .as_ref()
            .map(|x: &Tensor<B, 3>| max(output.clone() - x.clone()))
            .transpose()?;
        let mut times = Vec::new();
        for _ in 0..3 {
            let tick = Instant::now();
            let _ = burn_gekko_train::latent_eval::values(
                model.predict(rgb.clone(), &refs, &mask)?.cross.mean(),
            )?;
            B::sync(&d).map_err(|e| anyhow::anyhow!("{e}"))?;
            times.push(tick.elapsed().as_secs_f64());
        }
        rows.push(serde_json::json!({"stable_attention":stable,"permutation_max_abs":max(delta.clone())?,"permutation_rms":burn_gekko_train::latent_eval::values(delta.powf_scalar(2.).mean().sqrt())?[0],"hidden_rgb_max_abs":max(output.clone()-hidden)?,"hidden_mse":mse,"max_abs_change_from_default_policy":policy_delta,"forward_seconds":times}));
        if original.is_none() {
            original = Some(output);
        }
    }
    write_json(
        &a.output,
        &serde_json::json!({"diagnostic_only":true,"scope":"one validation target; F64 softmax/value accumulation override; weights unchanged","backend":std::any::type_name::<B>(),"checkpoint":a.checkpoint,"checkpoint_sha256":meta["model_sha256"],"room_seed":scenes[0].seed,"rows":rows}),
    )?;
    Ok(())
}
fn main() -> Result<()> {
    #[cfg(feature = "cuda")]
    type B = burn::backend::Cuda<f32, i32>;
    #[cfg(not(feature = "cuda"))]
    type B = burn::backend::NdArray<f32>;
    run::<B>(Args::parse())
}
