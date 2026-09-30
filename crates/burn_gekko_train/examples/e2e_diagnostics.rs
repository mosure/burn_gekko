//! Evaluate the learned end-to-end RI channel and RGB input interventions.
use anyhow::{Result, ensure};
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::backend::Backend,
};
use burn_gekko_data::{Split, Visibility};
use burn_gekko_train::{
    e2e_pilot::{E2eConfig, Scene, evaluate, initialize},
    encoder::{image_tensor, visible_mask},
    eval::ranking_metrics,
    train::scalar,
};
use clap::Parser;
use std::{fs, path::PathBuf};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long,default_value="validation",value_parser=["train","validation","test"])]
    split: String,
    #[arg(long)]
    rooms: Option<usize>,
    #[arg(long)]
    checkpoint: PathBuf,
    /// Export RGB and monocular predictions for every requested room/view.
    #[arg(long)]
    rgb_all: bool,
    /// Evaluate on a new immutable capture after verifying every room seed is
    /// disjoint from the checkpoint dataset. No model or optimizer is changed.
    #[arg(long)]
    evaluation_dataset: Option<PathBuf>,
    /// Export an unrelated-reference control alongside the related predictions.
    #[arg(long, requires = "rgb_all")]
    unrelated: bool,
    /// Explicit diagnostic override of softmax/value accumulation precision.
    #[arg(long)]
    stable_attention: bool,
}
fn run<B: Backend>(args: &Args) -> Result<()> {
    ensure!(!args.output.exists(), "choose new output path");
    fs::create_dir_all(&args.output)?;
    let mut c: E2eConfig = burn_gekko_data::read_config(&args.config)?;
    if args.stable_attention {
        c.decoder_stable_attention = true;
    }
    // This diagnostic permits v1/v2 source compatibility checks, but never a
    // silent change to skipped architecture fields when loading model records.
    let recorded: E2eConfig = burn_gekko_data::read_config(
        &args
            .checkpoint
            .parent()
            .ok_or_else(|| anyhow::anyhow!("checkpoint has no run directory"))?
            .join("config.toml"),
    )?;
    ensure!(
        c.decoder_width == recorded.decoder_width
            && c.decoder_depth == recorded.decoder_depth
            && c.decoder_heads == recorded.decoder_heads
            && c.decoder_qk_norm == recorded.decoder_qk_norm
            && (c.decoder_stable_attention == recorded.decoder_stable_attention
                || args.stable_attention)
            && c.rgb_head == recorded.rgb_head
            && c.appearance_transport == recorded.appearance_transport
            && c.transport_matching == recorded.transport_matching
            && c.transport_max_displacement == recorded.transport_max_displacement
            && c.appearance_bypass == recorded.appearance_bypass,
        "checkpoint decoder architecture mismatch"
    );

    let d = Default::default();
    let (model, encoder_id) = initialize::<B>(&c, &d)?;
    let meta: serde_json::Value =
        serde_json::from_slice(&fs::read(args.checkpoint.join("metadata.json"))?)?;
    ensure!(
        meta["noncommercial_weight_dependencies"]
            .as_array()
            .is_some_and(|a| a.is_empty()),
        "NC checkpoint rejected"
    );
    ensure!(
        meta["encoder_id"] == encoder_id,
        "encoder identity mismatch"
    );
    ensure!(
        meta["model_sha256"] == burn_gekko_data::sha256_file(&args.checkpoint.join("model.mpk"))?,
        "model hash mismatch"
    );
    let model = model.load_file(
        args.checkpoint.join("model"),
        &NamedMpkFileRecorder::<FullPrecisionSettings>::default(),
        &d,
    )?;
    let original = burn_gekko_data::open_dataset(&c.dataset)?;
    ensure!(
        meta["dataset_id"] == original.dataset_id,
        "dataset mismatch"
    );
    let manifest = if let Some(path) = &args.evaluation_dataset {
        ensure!(
            args.split != "train",
            "fresh evaluation cannot be labeled training"
        );
        let fresh = burn_gekko_data::open_dataset(path)?;
        validate_fresh_capture(&original, &fresh)?;
        c.dataset = path.clone();
        fresh
    } else {
        original.clone()
    };
    burn_gekko_data::write_config(&args.output.join("config.toml"), &c)?;
    let mut audit = None;
    let split = match args.split.as_str() {
        "test" => Split::Test,
        "train" => Split::Train,
        _ => Split::Validation,
    };
    let mut scores = Vec::new();
    let mut rows = Vec::new();
    let mut rgb_scenes = Vec::new();
    let mut batch_audit_images = Vec::new();
    for (index, entry) in manifest
        .scenes
        .iter()
        .filter(|e| e.split == split)
        .take(args.rooms.unwrap_or(usize::MAX))
        .enumerate()
    {
        let path = c.dataset.join("raw").join(&entry.file);
        let rgb = burn_gekko_data::load_rgb(&path)?;
        let grid = [rgb.height / 16, rgb.width / 16];
        let images: Vec<_> = (0..rgb.views.len())
            .map(|v| image_tensor::<B>(&rgb, v, &d))
            .collect();
        if index < c.batch_size {
            batch_audit_images.push(images.clone());
        }
        if args.rgb_all {
            rgb_scenes.push(Scene {
                seed: entry.seed,
                rgb: images.clone(),
            });
        }
        let features: Vec<_> = images
            .iter()
            .map(|r| model.encode(r.clone(), None))
            .collect();
        if index == 0 {
            let mask = visible_mask(grid[0] * grid[1], c.mask_ratio, 79, 0)?;
            let refs: Vec<_> = (1..=c.references).map(|v| images[v].clone()).collect();
            let normal = model.complete(images[0].clone(), &refs, &mask)?;
            let dir = args
                .output
                .join("completion")
                .join(format!("room-{}-view-0", entry.seed));
            fs::create_dir_all(&dir)?;
            let export = |name: &str, x: burn::tensor::Tensor<B, 4>| -> Result<()> {
                let data = x
                    .permute([0, 2, 3, 1])
                    .into_data()
                    .convert::<f32>()
                    .to_vec::<f32>()?;
                fs::write(
                    dir.join(name),
                    data.into_iter()
                        .flat_map(f32::to_le_bytes)
                        .collect::<Vec<_>>(),
                )?;
                Ok(())
            };
            export("target.f32", images[0].clone())?;
            export("prediction.f32", normal.rgb.clone())?;
            export("monocular.f32", normal.monocular.clone())?;
            for (i, r) in refs.iter().enumerate() {
                export(&format!("reference-{i}.f32"), r.clone())?;
            }
            let ids: Vec<i64> = (0..grid[0] * grid[1])
                .filter(|i| !mask.indices().contains(i))
                .map(|i| i as i64)
                .collect();
            let count = ids.len();
            let ids = burn::tensor::Tensor::<B, 1, burn::tensor::Int>::from_data(
                burn::tensor::TensorData::new(ids, [count]),
                &d,
            );
            let error = burn_gekko_train::loss::rgb_patches(
                normal.rgb.clone() - images[0].clone(),
                16,
                false,
            );
            let mse = scalar(error.powf_scalar(2.).select(1, ids).mean())?;
            burn_gekko_data::write_json(
                &dir.join("sample.json"),
                &serde_json::json!({"room_seed":entry.seed,"target_view":0,"height":rgb.height,"width":rgb.width,"visible_patch_ids":mask.indices(),"hidden_rgb_mse":mse,"reference_count":c.references,"oracle_statistics":false,"split":split,"diagnostic_only":split==Split::Train}),
            )?;
            let mut pixels = images[0]
                .clone()
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()?;
            for channel in 0..3 {
                for y in 0..rgb.height {
                    for x in 0..rgb.width {
                        if !mask.indices().contains(&(y / 16 * grid[1] + x / 16)) {
                            pixels[(channel * rgb.height + y) * rgb.width + x] = 1.7;
                        }
                    }
                }
            }
            let changed = burn::tensor::Tensor::from_data(
                burn::tensor::TensorData::new(pixels, [1, 3, rgb.height, rgb.width]),
                &d,
            );
            let intervened = model.complete(changed, &refs, &mask)?;
            let reversed: Vec<_> = refs.iter().rev().cloned().collect();
            let permuted = model.complete(images[0].clone(), &reversed, &mask)?;
            let features_normal = model.encode_references(&refs);
            let features_reversed = model.encode_references(&reversed);
            let mut encoder_permutation = 0f64;
            for (normal, reversed) in features_normal
                .into_iter()
                .zip(features_reversed.into_iter().rev())
            {
                encoder_permutation =
                    encoder_permutation.max(scalar((normal - reversed).abs().max())?);
            }
            let permutation_difference = normal.rgb.clone() - permuted.rgb;
            audit = Some(
                serde_json::json!({"hidden_target_intervention_max_abs":scalar((normal.rgb.clone()-intervened.rgb).abs().max())?,"reference_permutation_max_abs":scalar(permutation_difference.clone().abs().max())?,"reference_permutation_rms":scalar(permutation_difference.powf_scalar(2.).mean())?.sqrt(),"reference_encoder_permutation_max_abs":encoder_permutation,"monocular_reference_permutation_max_abs":scalar((normal.monocular-permuted.monocular).abs().max())?}),
            );
        }
        for target in 0..images.len() {
            let refs: Vec<_> = (1..=c.references)
                .map(|i| (target + i) % images.len())
                .collect();
            let output = model.decoder.relative_improvement(
                features[target].clone(),
                refs.iter().map(|&v| features[v].clone()).collect(),
                grid,
            )?;
            if index == 0 && target == 0 {
                let reversed = model.decoder.relative_improvement(
                    features[target].clone(),
                    refs.iter().rev().map(|&v| features[v].clone()).collect(),
                    grid,
                )?;
                audit.as_mut().unwrap()["ri_reference_permutation_max_abs"] =
                    serde_json::json!(scalar((output.clone() - reversed).abs().max())?);
            }
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
            {
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
    if batch_audit_images.len() == c.batch_size {
        use burn::tensor::{Int, Tensor, TensorData};
        let [_, _, h, w] = batch_audit_images[0][0].dims();
        let mask = visible_mask(h / 16 * (w / 16), c.mask_ratio, 79, 0)?;
        // Use distinct rooms and varied target-view indices at the configured
        // training batch size. Per-view encoders and layer norms must preserve
        // sample independence, including the native GPU execution path.
        let targets: Vec<_> = (0..c.batch_size)
            .map(|i| batch_audit_images[i][i % batch_audit_images[i].len()].clone())
            .collect();
        let references: Vec<Vec<_>> = (0..c.batch_size)
            .map(|i| {
                (1..=c.references)
                    .map(|offset| {
                        batch_audit_images[i][(i + offset) % batch_audit_images[i].len()].clone()
                    })
                    .collect()
            })
            .collect();
        let independent = (0..c.batch_size)
            .map(|i| model.complete(targets[i].clone(), &references[i], &mask))
            .collect::<Result<Vec<_>>>()?;
        let target_batch = Tensor::cat(targets, 0);
        let reference_batches = (0..c.references)
            .map(|v| {
                Tensor::cat(
                    (0..c.batch_size)
                        .map(|i| references[i][v].clone())
                        .collect(),
                    0,
                )
            })
            .collect::<Vec<_>>();
        let batched = model.complete(target_batch, &reference_batches, &mask)?;
        let expected = Tensor::cat(
            (0..c.batch_size)
                .map(|i| independent[i].rgb.clone())
                .collect(),
            0,
        );
        let expected_mono = Tensor::cat(
            (0..c.batch_size)
                .map(|i| independent[i].monocular.clone())
                .collect(),
            0,
        );
        let difference = batched.rgb - expected;
        let hidden: Vec<i64> = (0..h / 16 * (w / 16))
            .filter(|i| !mask.indices().contains(i))
            .map(|i| i as i64)
            .collect();
        let len = hidden.len();
        let ids = Tensor::<B, 1, Int>::from_data(TensorData::new(hidden, [len]), &d);
        let report = audit.as_mut().unwrap();
        report["batched_targets"] = serde_json::json!(c.batch_size);
        report["batch_audit_unique_rooms"] = serde_json::json!(c.batch_size);
        report["batched_rgb_max_abs"] = serde_json::json!(scalar(difference.clone().abs().max())?);
        report["batched_rgb_hidden_rms"] = serde_json::json!(
            scalar(
                burn_gekko_train::loss::rgb_patches(difference, 16, false)
                    .select(1, ids)
                    .powf_scalar(2.)
                    .mean()
            )?
            .sqrt()
        );
        report["batched_monocular_max_abs"] =
            serde_json::json!(scalar((batched.monocular - expected_mono).abs().max())?);
    }
    if args.rgb_all {
        ensure!(!rgb_scenes.is_empty(), "no RGB scenes selected");
        let directory = args.output.join("rgb-all");
        evaluate(
            &model,
            &rgb_scenes,
            &c,
            &directory,
            rgb_scenes.len(),
            false,
            true,
        )?;
        if args.unrelated {
            evaluate(
                &model,
                &rgb_scenes,
                &c,
                &args.output.join("rgb-unrelated"),
                rgb_scenes.len(),
                true,
                true,
            )?;
        }
        // Preserve the split on every standalone RGB export. Training examples
        // must never be mistaken for held-out reconstruction evidence.
        let directories = std::iter::once(directory)
            .chain(args.unrelated.then(|| args.output.join("rgb-unrelated")));
        for directory in directories {
            for entry in fs::read_dir(&directory)? {
                let path = entry?.path();
                let path = if path.is_dir() {
                    path.join("sample.json")
                } else {
                    path
                };
                if path.extension().is_some_and(|e| e == "json") {
                    let mut value: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
                    value["split"] = serde_json::to_value(split)?;
                    value["diagnostic_only"] = serde_json::json!(split == Split::Train);
                    burn_gekko_data::write_json(&path, &value)?;
                }
            }
        }
    }
    burn_gekko_data::write_json(
        &args.output.join("report.json"),
        &serde_json::json!({"dataset_id":manifest.dataset_id,"checkpoint_dataset_id":original.dataset_id,"evaluation_dataset_override":args.evaluation_dataset,"stable_attention":c.decoder_stable_attention,"attention_precision_override":args.stable_attention,"encoder_id":encoder_id,"split":split,"aggregate":"joint set-attention RI head trained from scratch","covisibility":ranking_metrics(&scores)?,"targets":rows,"input_audit":audit,"checkpoint_sha256":burn_gekko_data::sha256_file(&args.checkpoint.join("model.mpk"))?}),
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

/// Keep fresh test captures separate from every split of the source dataset.
fn validate_fresh_capture(
    original: &burn_gekko_data::DatasetManifest,
    fresh: &burn_gekko_data::DatasetManifest,
) -> Result<()> {
    ensure!(
        original.config.width == fresh.config.width
            && original.config.height == fresh.config.height
            && original.config.cameras == fresh.config.cameras,
        "fresh evaluation image dimensions or camera count differ"
    );
    let known: std::collections::BTreeSet<_> = original.scenes.iter().map(|s| s.seed).collect();
    ensure!(
        fresh.scenes.iter().all(|s| !known.contains(&s.seed)),
        "fresh evaluation room seeds overlap the checkpoint dataset"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fresh_evaluation_rejects_overlap_in_any_split_and_shape_changes() {
        let make = |seed| burn_gekko_data::DatasetManifest {
            schema: 1,
            dataset_id: "fixture".into(),
            generator: burn_gekko_data::GENERATOR.into(),
            binary_sha256: "fixture".into(),
            config: Default::default(),
            scenes: vec![burn_gekko_data::SceneEntry {
                file: "fixture".into(),
                sha256: "fixture".into(),
                seed,
                split: burn_gekko_data::Split::Test,
            }],
        };
        let original = make(19);
        assert!(validate_fresh_capture(&original, &make(19)).is_err());
        let mut fresh = make(20);
        assert!(validate_fresh_capture(&original, &fresh).is_ok());
        fresh.config.width *= 2;
        assert!(validate_fresh_capture(&original, &fresh).is_err());
    }
}
