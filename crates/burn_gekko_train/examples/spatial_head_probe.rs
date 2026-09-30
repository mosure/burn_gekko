//! Native compatibility and neutral-initialization check on real cached RGB.
use anyhow::{Result, ensure};
use burn_gekko::heads::spatial::SpatialDescriptorConfig;
use burn_gekko_data::{Split, load_rgb, open_dataset, read_config, write_json};
use burn_gekko_train::{
    encoder::image_tensor, latent_assess::load_assessed_model, latent_pilot::WeightAncestor,
};
use clap::Parser;
use serde::Deserialize;
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    output: PathBuf,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    dataset: PathBuf,
    weights: WeightAncestor,
    rooms: usize,
    residual_radius: f64,
}
fn main() -> Result<()> {
    let a = Args::parse();
    let c: Config = read_config(&a.config)?;
    ensure!(
        !a.output.exists() && (1..=4).contains(&c.rooms),
        "invalid probe output/size"
    );
    #[cfg(feature = "cuda")]
    type B = burn::backend::Cuda<f32, i32>;
    #[cfg(not(feature = "cuda"))]
    type B = burn::backend::NdArray<f32>;
    let d = Default::default();
    let loaded = load_assessed_model::<B>(&c.weights, &d)?;
    ensure!(
        loaded.model.fusion.spatial_descriptor.is_none(),
        "probe expects a legacy parent without this head"
    );
    let model = loaded
        .model
        .prepare_spatial_descriptor(Some(&SpatialDescriptorConfig {
            residual_radius: c.residual_radius,
        }))?;
    model.validate_spatial_descriptor()?;
    let manifest = open_dataset(&c.dataset)?;
    let mut max_delta = 0f64;
    let mut matching_changes = 0;
    let mut queries = 0;
    for e in manifest
        .scenes
        .iter()
        .filter(|e| e.split == Split::Validation)
        .take(c.rooms)
    {
        let rgb = load_rgb(&c.dataset.join("raw").join(&e.file))?;
        let views = [
            image_tensor::<B>(&rgb, 0, &d),
            image_tensor::<B>(&rgb, 1, &d),
        ];
        let encoded = model.encode_references(&views);
        let n = [rgb.height / 16, rgb.width / 16];
        let pair = model
            .fusion
            .decoder
            .pair_training(encoded[0].clone(), encoded[1].clone(), n)?;
        let spatial = model
            .spatial_descriptor(encoded[0].clone(), pair.features)
            .unwrap();
        let width = model.encoder_config.encoder.embed_dim;
        let base = encoded[0].clone().slice_dim(2, width..2 * width);
        let centered = base.clone() - base.mean_dim(1);
        max_delta = max_delta.max(burn_gekko::tensor::scalar(
            (spatial - centered).abs().max(),
        )?);
        let teacher = burn_gekko_train::correspondence::fixed_views(
            &loaded.teacher,
            &model.encoder_config,
            &views,
        );
        let mut readouts = burn_gekko_train::correspondence::standard_readouts(
            &model,
            encoded[0].clone(),
            encoded[1].clone(),
            teacher[0].clone(),
            teacher[1].clone(),
            n,
        )?;
        let layer = loaded.config.spatial_input_layer.unwrap();
        let control = burn_gekko_train::encoder_audit::capture_view_layers(
            &model.encoder,
            &model.encoder_config,
            &views,
            &[layer - 1],
        )?;
        readouts.extend(burn_gekko_train::encoder_audit::readouts(
            "student", &control, 0, 1,
        )?);
        for (new, baseline) in [
            (
                "spatial_residual".to_string(),
                format!("student_l{layer:02}_centered"),
            ),
            (
                "spatial_residual_conditional".into(),
                format!("student_l{layer:02}_centered_conditional"),
            ),
        ] {
            let indices = |name: &str| &readouts.iter().find(|(k, _)| k == name).unwrap().1.0;
            matching_changes += indices(&new)
                .iter()
                .zip(indices(&baseline))
                .filter(|(a, b)| a != b)
                .count();
            queries += indices(&new).len();
        }
    }
    ensure!(
        queries > 0 && max_delta == 0. && matching_changes == 0,
        "zero-head parity failed: delta {max_delta}, changes {matching_changes}/{queries}"
    );
    write_json(
        &a.output,
        &serde_json::json!({"status":"passed","checkpoint_sha256":c.weights.model_sha256,"zero_head_max_abs_feature_delta":max_delta,"matching_changes":matching_changes,"queries":queries,"geometry_loaded":false,"backend":std::any::type_name::<B>()}),
    )?;
    Ok(())
}
