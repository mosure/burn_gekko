//! Evaluate the pretrained RI channel separately from masked completion.
use anyhow::{Result, ensure};
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::backend::Backend,
};
use burn_gekko_data::{Split, Visibility};
use burn_gekko_train::{
    encoder::{ImageFeatures, image_features, image_tensor, normalize},
    eval::ranking_metrics,
    hybrid::HybridFusion,
    hybrid_pilot::HybridConfig,
    train::load_encoder,
};
use clap::Parser;
use std::{fs, path::PathBuf};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long,default_value="validation",value_parser=["validation","test"])]
    split: String,
    #[arg(long)]
    adapter: bool,
}
fn run<B: Backend>(args: &Args) -> Result<()> {
    ensure!(!args.output.exists(), "choose new output path");
    fs::create_dir_all(&args.output)?;
    let c: HybridConfig = burn_gekko_data::read_config(&args.config)?;
    let d = Default::default();
    let (jepa, jepa_config, encoder_id) = load_encoder::<B>(&c.encoder, c.seed, &d)?;
    let (appearance, decoder) = burn_gekko_train::released::load::<B>(&c.weights, &d)?;
    let model = HybridFusion::new(decoder, 1536, c.normalize_predicted_content, &d).load_file(
        c.checkpoint.as_ref().unwrap(),
        &NamedMpkFileRecorder::<FullPrecisionSettings>::default(),
        &d,
    )?;
    let manifest = burn_gekko_data::open_dataset(&c.dataset)?;
    let split = if args.split == "test" {
        Split::Test
    } else {
        Split::Validation
    };
    let mut scores = Vec::new();
    let mut rows = Vec::new();
    for (index, entry) in manifest
        .scenes
        .iter()
        .filter(|e| e.split == split)
        .enumerate()
    {
        let path = c.dataset.join("raw").join(&entry.file);
        let rgb = burn_gekko_data::load_rgb(&path)?;
        let grid = [rgb.height / 16, rgb.width / 16];
        let images: Vec<_> = (0..rgb.views.len())
            .map(|v| image_tensor::<B>(&rgb, v, &d))
            .collect();
        let app: Vec<_> = images
            .iter()
            .map(|r| appearance.forward(r.clone(), None))
            .collect();
        let jf: Vec<_> = images
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
        for target in 0..images.len() {
            let refs: Vec<_> = (1..=c.references)
                .map(|i| (target + i) % images.len())
                .collect();
            let output = model.relative_improvement(
                app[target].clone(),
                jf[target].clone(),
                refs.iter().map(|&i| app[i].clone()).collect(),
                refs.iter().map(|&i| jf[i].clone()).collect(),
                grid,
                args.adapter,
            );
            let prediction = output.into_data().convert::<f32>().to_vec::<f32>()?;
            // Annotation decoding occurs after the model forward, and cannot affect it.
            let geometry = burn_gekko_data::load_geometry(&path)?;
            let start = scores.len();
            let mut labels = Vec::new();
            let mut heatmap = Vec::new();
            for y in 0..rgb.height {
                for x in 0..rgb.width {
                    let score =
                        prediction[(y / 16 * grid[1] + x / 16) * 256 + (y % 16) * 16 + x % 16];
                    ensure!(score.is_finite(), "nonfinite RI");
                    heatmap.push(score);
                    let visibility: Vec<_> = refs
                        .iter()
                        .map(|&r| {
                            burn_gekko_data::visibility(&geometry, target, r, y * rgb.width + x)
                        })
                        .collect();
                    if visibility.contains(&Visibility::Visible) {
                        scores.push((score, true));
                        labels.push(1u8);
                    } else if visibility.contains(&Visibility::Unknown) {
                        labels.push(255u8);
                    } else {
                        scores.push((score, false));
                        labels.push(0u8);
                    }
                }
            }
            rows.push(serde_json::json!({"room_seed":entry.seed,"target_view":target,"covisibility":ranking_metrics(&scores[start..])?}));
            if c.export_all_views || (index < 16 && target == 0) {
                let dir = args
                    .output
                    .join(format!("room-{}-view-{target}", entry.seed));
                fs::create_dir_all(&dir)?;
                fs::write(
                    dir.join("ri.f32"),
                    heatmap
                        .iter()
                        .flat_map(|x| x.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
                fs::write(dir.join("visibility.u8"), labels)?;
            }
        }
    }
    burn_gekko_data::write_json(
        &args.output.join("report.json"),
        &serde_json::json!({"dataset_id":manifest.dataset_id,"encoder_id":encoder_id,"split":split,"adapter_enabled":args.adapter,"aggregate":"max over pairwise pretrained RI outputs","covisibility":ranking_metrics(&scores)?,"targets":rows,"checkpoint_sha256":burn_gekko_data::sha256_file(c.checkpoint.as_ref().unwrap())?}),
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
