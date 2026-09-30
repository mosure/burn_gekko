use anyhow::{Result, ensure};
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Tensor, TensorData, backend::Backend},
};
use burn_gekko_train::{
    encoder::{ImageFeatures, image_features, normalize},
    hybrid::HybridFusion,
    hybrid_pilot::HybridConfig,
    train::{load_encoder, scalar},
};
use burn_vjepa::SparseTokenMask;
use clap::Parser;
use std::{fs, path::PathBuf};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    sample: PathBuf,
    #[arg(long)]
    output: PathBuf,
}
fn run<B: Backend>(args: &Args) -> Result<()> {
    let c: HybridConfig = burn_gekko_data::read_config(&args.config)?;
    let d = Default::default();
    let (jepa, jepa_config, _) = load_encoder::<B>(&c.encoder, c.seed, &d)?;
    let (appearance, decoder) = burn_gekko_train::released::load::<B>(&c.weights, &d)?;
    let model = HybridFusion::new(decoder, 1536, c.normalize_predicted_content, &d).load_file(
        c.checkpoint.as_ref().unwrap(),
        &NamedMpkFileRecorder::<FullPrecisionSettings>::default(),
        &d,
    )?;
    let m: serde_json::Value = serde_json::from_slice(&fs::read(args.sample.join("sample.json"))?)?;
    let h = m["height"].as_u64().unwrap() as usize;
    let w = m["width"].as_u64().unwrap() as usize;
    let ids: Vec<usize> = m["visible_patch_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let mask = SparseTokenMask::new(ids.clone(), h * w / 256)?;
    let read = |name: &str| -> Result<Vec<f32>> {
        let bytes = fs::read(args.sample.join(name))?;
        Ok(bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect())
    };
    let tensor = |values: Vec<f32>| {
        Tensor::<B, 4>::from_data(TensorData::new(values, [1, h, w, 3]), &d).permute([0, 3, 1, 2])
    };
    let rgb = read("target.f32")?;
    let mut changed = rgb.clone();
    for y in 0..h {
        for x in 0..w {
            if !ids.contains(&(y / 16 * (w / 16) + x / 16)) {
                for channel in 0..3 {
                    changed[(y * w + x) * 3 + channel] =
                        ((y * 31 + x * 47 + channel * 59) % 251) as f32 / 250.;
                }
            }
        }
    }
    let references: Vec<_> = (0..c.references)
        .map(|i| read(&format!("reference-{i}.f32")).map(&tensor))
        .collect::<Result<_>>()?;
    let app_refs: Vec<_> = references
        .iter()
        .map(|r| appearance.forward(r.clone(), None))
        .collect();
    let jepa_refs: Vec<_> = references
        .iter()
        .map(|r| {
            image_features(
                jepa.forward_image(normalize(r.clone(), &jepa_config), None)
                    .tokens,
                r.clone(),
                None,
                ImageFeatures::SemanticRgb,
            )
        })
        .collect();
    let forward = |rgb: Vec<f32>, reverse: bool, adapter: bool| {
        let rgb = tensor(rgb);
        let app = appearance.forward(rgb.clone(), Some(&mask));
        let jf = image_features(
            jepa.forward_image(normalize(rgb.clone(), &jepa_config), Some(&mask))
                .tokens,
            rgb,
            Some(&mask),
            ImageFeatures::SemanticRgb,
        );
        let mut ar = app_refs.clone();
        let mut jr = jepa_refs.clone();
        if reverse {
            ar.reverse();
            jr.reverse();
        }
        model
            .forward(app, jf, ar, jr, &mask, [h / 16, w / 16], adapter)
            .rgb
    };
    let original = forward(rgb.clone(), false, true);
    let intervention = forward(changed, false, true);
    let permuted = forward(rgb.clone(), true, true);
    let ablated = forward(rgb, false, false);
    let hidden_delta = scalar((original.clone() - intervention).abs().max())?;
    let permutation_delta = scalar((original.clone() - permuted).abs().max())?;
    let adapter_effect = scalar((original - ablated).powf_scalar(2.).mean())?;
    ensure!(hidden_delta == 0., "hidden pixels changed completion");
    ensure!(
        permutation_delta < 1e-6,
        "reference permutation changed completion"
    );
    burn_gekko_data::write_json(
        &args.output,
        &serde_json::json!({"hidden_target_intervention_max_abs":hidden_delta,"reference_permutation_max_abs":permutation_delta,"adapter_ablation_rgb_mean_squared_difference":adapter_effect,"checkpoint_sha256":burn_gekko_data::sha256_file(c.checkpoint.as_ref().unwrap())?}),
    )?;
    Ok(())
}
fn main() -> Result<()> {
    let args = Args::parse();
    #[cfg(feature = "cuda")]
    run::<burn::backend::Cuda<f32, i32>>(&args)?;
    #[cfg(not(feature = "cuda"))]
    run::<burn::backend::NdArray<f32>>(&args)?;
    Ok(())
}
