use crate::{
    batch::ResidentScene,
    encoder::{image_features, image_tensor, visible_mask},
    loss::{calibrated_rgb, content_prediction, reconstruction_loss, rgb_patches},
    model::{DecoderConfig, GekkoDecoder},
    train::{CheckpointMetadata, TrainConfig, TrainReport, load_encoder},
};
use anyhow::{Context, Result, ensure};
use burn::{
    module::Module,
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
    tensor::{Tensor, backend::Backend},
};
use burn_gekko_data::{
    Split, Visibility, load_geometry, load_rgb, open_dataset, read_config_snapshot, sha256_file,
    visibility,
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvaluationOptions {
    #[serde(default)]
    pub unrelated_references: bool,
    /// One predetermined target-zero sample per evenly spaced room.
    #[serde(default)]
    pub export_rooms: usize,
    pub export_directory: Option<PathBuf>,
    /// Explicit opt-in for overfit diagnostics; never a held-out quality result.
    #[serde(default)]
    pub diagnostic_training: bool,
    pub max_rooms: Option<usize>,
    pub target_view: Option<usize>,
    pub mask_step: Option<usize>,
    /// Explicit training-cache provenance for evaluation on a fresh, disjoint
    /// dataset. Omitting this keeps the original strict dataset identity check.
    pub training_dataset: Option<PathBuf>,
}

pub use burn_gekko_eval::ranking::{RankingMetrics, ranking_metrics};

#[derive(Debug, Serialize)]
pub struct TargetMetrics {
    pub room_seed: u64,
    pub target_view: usize,
    pub covisibility: RankingMetrics,
    pub unknown_pixels: usize,
    pub total_loss: f64,
    pub cross_mse: f64,
    pub mae_mse: f64,
    pub ri_loss: f64,
    pub masked_rgb_mse: f64,
    pub masked_rgb_psnr: f64,
}

#[derive(Debug, Serialize)]
pub struct EvalReport {
    pub artifact_kind: String,
    pub dataset_id: String,
    pub training_dataset_id: String,
    pub encoder_id: String,
    pub split: Split,
    pub completed_training_steps: usize,
    pub evaluated_target_views: usize,
    pub unknown_pixels: usize,
    pub min_raw_ri: f32,
    pub max_raw_ri: f32,
    pub covisibility: RankingMetrics,
    pub targets: Vec<TargetMetrics>,
    pub mean_total_loss: f64,
    pub mean_cross_mse: f64,
    pub mean_mae_mse: f64,
    pub mean_ri_loss: f64,
    pub relative_cross_improvement: f64,
    pub reference_mode: String,
    pub samples: Vec<PathBuf>,
    pub caveat: String,
    pub rgb_uses_target_statistics: bool,
    pub mean_masked_rgb_mse: f64,
}

/// Geometry-derived labels are loaded only here, after the model input/forward API is fixed to RGB.
pub fn evaluate<B: Backend>(
    dataset: &Path,
    run: &Path,
    split: Split,
    device: &B::Device,
    checkpoint_step: Option<usize>,
) -> Result<EvalReport> {
    evaluate_with_options::<B>(
        dataset,
        run,
        split,
        device,
        checkpoint_step,
        &EvaluationOptions::default(),
    )
}

pub fn evaluate_with_options<B: Backend>(
    dataset: &Path,
    run: &Path,
    split: Split,
    device: &B::Device,
    checkpoint_step: Option<usize>,
    options: &EvaluationOptions,
) -> Result<EvalReport> {
    ensure!(
        split != Split::Train || options.diagnostic_training,
        "training split requires explicit diagnostic_training=true"
    );
    let manifest = open_dataset(dataset)?;
    ensure!(options.max_rooms != Some(0), "max_rooms must be positive");
    ensure!(
        options
            .target_view
            .is_none_or(|v| v < manifest.config.cameras),
        "invalid target_view"
    );
    ensure!(
        options.export_rooms <= 16,
        "at most 16 annotated rooms per evaluation"
    );
    ensure!(
        options.export_rooms == 0 || options.export_directory.is_some(),
        "sample export requires a directory"
    );
    if let Some(path) = &options.export_directory {
        ensure!(!path.exists(), "sample export directory already exists");
        let absolute = std::path::absolute(path)?;
        ensure!(
            !absolute
                .components()
                .any(|c| c == std::path::Component::ParentDir),
            "sample export path must not contain .."
        );
        let ancestor = absolute
            .ancestors()
            .find(|p| p.exists())
            .context("sample export parent")?;
        ensure!(
            fs::canonicalize(ancestor)?.starts_with(fs::canonicalize(".data")?),
            "sample exports must stay inside .data"
        );
        fs::create_dir_all(path)?;
        ensure!(
            fs::canonicalize(path)?.starts_with(fs::canonicalize(".data")?),
            "sample exports must stay inside .data"
        );
    }
    let config: TrainConfig = read_config_snapshot(run, "config")?;
    config.validate()?;
    let trained: TrainReport = serde_json::from_slice(&fs::read(run.join("report.json"))?)?;
    let selected_step = checkpoint_step.unwrap_or(trained.completed_steps);
    ensure!(
        selected_step <= trained.completed_steps,
        "checkpoint step exceeds completed training"
    );
    let checkpoint = run.join(format!("checkpoint-{selected_step:06}"));
    let meta: CheckpointMetadata =
        serde_json::from_slice(&fs::read(checkpoint.join("metadata.json"))?)?;
    ensure!(
        meta.schema == 1
            && meta.completed_steps == selected_step
            && meta.dataset_id == trained.dataset_id,
        "training report/checkpoint mismatch"
    );
    if manifest.dataset_id != meta.dataset_id {
        let source = options.training_dataset.as_ref().context(
            "evaluation dataset/checkpoint mismatch; cross-dataset evaluation requires training_dataset",
        )?;
        ensure!(
            split != Split::Train,
            "cross-dataset evaluation must be held out"
        );
        ensure!(
            fs::canonicalize(source)?.starts_with(fs::canonicalize(".data")?),
            "training cache must stay inside .data"
        );
        let training_manifest = open_dataset(source)?;
        ensure!(
            training_manifest.dataset_id == meta.dataset_id,
            "wrong training cache"
        );
        let old_seeds: std::collections::HashSet<_> =
            training_manifest.scenes.iter().map(|s| s.seed).collect();
        ensure!(
            manifest.scenes.iter().all(|s| !old_seeds.contains(&s.seed)),
            "fresh evaluation rooms overlap the original dataset"
        );
    }
    ensure!(
        sha256_file(&checkpoint.join("decoder.mpk"))? == meta.model_sha256,
        "decoder checkpoint checksum mismatch"
    );
    let (encoder, enc_config, encoder_id) =
        load_encoder::<B>(&config.encoder, config.seed, device)?;
    ensure!(
        encoder_id == meta.encoder_id && encoder_id == trained.encoder_id,
        "encoder changed since training"
    );
    let decoder = GekkoDecoder::<B>::with_reconstruction(
        &DecoderConfig {
            encoder_dim: config.image_features.width(enc_config.encoder.embed_dim),
            width: config.decoder_width,
            depth: config.decoder_depth,
            heads: config.decoder_heads,
            patch: enc_config.patch_size,
        },
        config.decoder_position,
        config.predict_patch_stats,
        device,
    )?
    .with_mae_context_before_self(config.mae_context_before_self)
    .load_file(
        checkpoint.join("decoder"),
        &NamedMpkFileRecorder::<FullPrecisionSettings>::default(),
        device,
    )?;
    let mut scores = Vec::new();
    let mut unknown = 0;
    let mut targets = 0;
    let mut target_metrics = Vec::new();
    let mut loss_sums = [0.0f64; 4];
    let mut rgb_mse_sum = 0.0;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let entries: Vec<_> = manifest
        .scenes
        .iter()
        .filter(|s| s.split == split)
        .take(options.max_rooms.unwrap_or(usize::MAX))
        .collect();
    ensure!(
        !options.unrelated_references || entries.len() >= 2,
        "unrelated control requires two held-out rooms"
    );
    let export_ids: std::collections::BTreeSet<_> = (0..options.export_rooms.min(entries.len()))
        .map(|i| i * entries.len() / options.export_rooms)
        .collect();
    let mut samples = Vec::new();
    for (entry_index, entry) in entries.iter().enumerate() {
        let path = dataset.join("raw").join(&entry.file);
        let rgb = load_rgb(&path)?;
        ensure!(
            config.references < rgb.views.len(),
            "reference count exceeds dataset views"
        );
        let geometry = load_geometry(&path)?;
        let prepared = ResidentScene::<B>::new(&rgb, &encoder, &enc_config, true, device);
        let unrelated = if options.unrelated_references {
            let other = load_rgb(
                &dataset
                    .join("raw")
                    .join(&entries[(entry_index + 1) % entries.len()].file),
            )?;
            Some(ResidentScene::<B>::new(
                &other,
                &encoder,
                &enc_config,
                true,
                device,
            ))
        } else {
            None
        };
        let patch = enc_config.patch_size;
        let grid = [rgb.height / patch, rgb.width / patch];
        let mask = visible_mask(
            grid[0] * grid[1],
            config.mask_ratio,
            config.seed,
            options.mask_step.unwrap_or(usize::MAX),
        )?;
        for target in 0..rgb.views.len() {
            if options.target_view.is_some_and(|v| target != v) {
                continue;
            }
            let score_start = scores.len();
            let unknown_start = unknown;
            let image = prepared.normalized[target].clone();
            let masked = image_features(
                encoder.forward_image(image.clone(), Some(&mask)).tokens,
                prepared.rgb[target].clone(),
                Some(&mask),
                config.image_features,
            );
            let full = image_features(
                prepared.full.as_ref().unwrap()[target].clone(),
                prepared.rgb[target].clone(),
                None,
                config.image_features,
            );
            let reference_ids: Vec<_> = (1..=config.references)
                .map(|i| (target + i) % rgb.views.len())
                .collect();
            let references = reference_ids
                .iter()
                .map(|&r| {
                    let source = unrelated.as_ref().unwrap_or(&prepared);
                    image_features(
                        source.full.as_ref().unwrap()[r].clone(),
                        source.rgb[r].clone(),
                        None,
                        config.image_features,
                    )
                })
                .collect();
            let output = decoder.forward(masked, full, references, &mask, grid)?;
            let raw_target = rgb_patches(prepared.rgb[target].clone(), patch, false);
            let raw_prediction = if config.predict_patch_stats {
                calibrated_rgb(output.cross_rgb.clone(), patch * patch * 3)
            } else if config.normalize_targets {
                let mean = raw_target.clone().mean_dim(2);
                let centered = raw_target.clone() - mean.clone();
                let variance =
                    centered.powf_scalar(2.0).sum_dim(2) / (patch * patch * 3 - 1) as f32;
                output.cross_rgb.clone() * (variance + 1e-6).sqrt() + mean
            } else {
                output.cross_rgb.clone()
            };
            let hidden: Vec<i64> = (0..grid[0] * grid[1])
                .filter(|i| mask.indices().binary_search(i).is_err())
                .map(|i| i as i64)
                .collect();
            let hidden_len = hidden.len();
            let hidden = burn::tensor::Tensor::<B, 1, burn::tensor::Int>::from_data(
                burn::tensor::TensorData::new(hidden, [hidden_len]),
                device,
            );
            let rgb_mse = crate::train::scalar(
                (raw_prediction - raw_target)
                    .powf_scalar(2.0)
                    .select(1, hidden)
                    .mean(),
            )?;
            rgb_mse_sum += rgb_mse;
            let ri = output.ri.clone().into_data().to_vec::<f32>()?;
            let export = target == 0 && export_ids.contains(&entry_index);
            let predictions = if export {
                Some((
                    content_prediction(output.cross_rgb.clone(), patch * patch * 3)
                        .into_data()
                        .to_vec::<f32>()?,
                    content_prediction(output.mae_rgb.clone(), patch * patch * 3)
                        .into_data()
                        .to_vec::<f32>()?,
                    if config.predict_patch_stats {
                        output
                            .cross_rgb
                            .clone()
                            .slice_dim(2, patch * patch * 3..patch * patch * 3 + 2)
                            .into_data()
                            .to_vec::<f32>()?
                    } else {
                        Vec::new()
                    },
                    if config.predict_patch_stats {
                        output
                            .mae_rgb
                            .clone()
                            .slice_dim(2, patch * patch * 3..patch * patch * 3 + 2)
                            .into_data()
                            .to_vec::<f32>()?
                    } else {
                        Vec::new()
                    },
                ))
            } else {
                None
            };
            let losses = reconstruction_loss(
                output,
                rgb_patches(image_tensor::<B>(&rgb, target, device), patch, false),
                &mask,
                config.normalize_targets,
                config.predict_patch_stats,
            );
            let losses = Tensor::cat(vec![losses.total, losses.cross, losses.mae, losses.ri], 0)
                .into_data()
                .convert::<f32>()
                .to_vec::<f32>()?;
            ensure!(
                losses.iter().all(|v| v.is_finite()),
                "nonfinite evaluation loss"
            );
            for (sum, &value) in loss_sums.iter_mut().zip(&losses) {
                *sum += value as f64;
            }
            ensure!(ri.iter().all(|v| v.is_finite()), "nonfinite RI prediction");
            let mut label_map = if export {
                Vec::with_capacity(rgb.width * rgb.height)
            } else {
                Vec::new()
            };
            for y in 0..rgb.height {
                for x in 0..rgb.width {
                    let score = ri[((y / patch) * grid[1] + x / patch) * patch * patch
                        + (y % patch) * patch
                        + x % patch];
                    min = min.min(score);
                    max = max.max(score);
                    let labels: Vec<_> = reference_ids
                        .iter()
                        .map(|&r| visibility(&geometry, target, r, y * rgb.width + x))
                        .collect();
                    if labels.contains(&Visibility::Visible) {
                        scores.push((score, true));
                        if export {
                            label_map.push(1);
                        }
                    } else if labels.contains(&Visibility::Unknown) {
                        unknown += 1;
                        if export {
                            label_map.push(255);
                        }
                    } else {
                        scores.push((score, false));
                        if export {
                            label_map.push(0);
                        }
                    }
                }
            }
            target_metrics.push(TargetMetrics {
                room_seed: entry.seed,
                target_view: target,
                covisibility: ranking_metrics(&scores[score_start..])?,
                unknown_pixels: unknown - unknown_start,
                total_loss: losses[0] as f64,
                cross_mse: losses[1] as f64,
                mae_mse: losses[2] as f64,
                ri_loss: losses[3] as f64,
                masked_rgb_mse: rgb_mse,
                masked_rgb_psnr: -10.0 * rgb_mse.max(1e-12).log10(),
            });
            if let Some((cross, mae, cross_stats, mae_stats)) = predictions {
                let dir = options
                    .export_directory
                    .as_ref()
                    .unwrap()
                    .join(format!("room-{}-view-{target}", entry.seed));
                fs::create_dir(&dir)?;
                write_floats(&dir.join("target.f32"), &rgb.views[target])?;
                for (i, &reference) in reference_ids.iter().enumerate() {
                    // Annotation exports are intended for correct references; do not silently label controls as originals.
                    ensure!(
                        !options.unrelated_references,
                        "annotated exports require correct references"
                    );
                    write_floats(
                        &dir.join(format!("reference-{i}.f32")),
                        &rgb.views[reference],
                    )?;
                }
                write_floats(&dir.join("cross.f32"), &cross)?;
                write_floats(&dir.join("mae.f32"), &mae)?;
                if config.predict_patch_stats {
                    write_floats(&dir.join("cross-statistics.f32"), &cross_stats)?;
                    write_floats(&dir.join("mae-statistics.f32"), &mae_stats)?;
                }
                write_floats(&dir.join("ri.f32"), &ri)?;
                fs::write(dir.join("visibility.u8"), label_map)?;
                burn_gekko_data::write_json(
                    &dir.join("sample.json"),
                    &serde_json::json!({
                        "room_seed": entry.seed, "target_view":target, "reference_views":reference_ids,
                        "width":rgb.width, "height":rgb.height, "patch_size":patch,
                        "visible_patch_ids":mask.indices(), "normalize_targets":config.normalize_targets,
                        "predicted_patch_statistics":config.predict_patch_stats,
                        "statistics_layout":"per patch: predicted mean, predicted log standard deviation; float32 little-endian",
                        "prediction_layout":"patch-major, pixel-major, RGB; float32 little-endian",
                        "rgb_layout":"HWC sRGB float32 little-endian", "visibility_labels":{"0":"not visible in any reference","1":"visible in at least one reference","255":"unknown"},
                        "visualization_caveat":if config.normalize_targets && !config.predict_patch_stats { "Oracle display: target patch statistics required." } else { "Standalone RGB prediction; no hidden target statistics used. Visible patches can be composited from input." },
                        "metrics":target_metrics.last().unwrap()
                    }),
                )?;
                samples.push(dir);
            }
            targets += 1;
        }
        if (entry_index + 1).is_multiple_of(8) {
            eprintln!(
                "evaluated {}/{} held-out rooms",
                entry_index + 1,
                entries.len()
            );
        }
    }
    ensure!(targets > 0, "empty evaluation split");
    Ok(EvalReport {
        artifact_kind: if split == Split::Train {
            "training_overfit_diagnostic"
        } else {
            "bounded_diagnostic_not_a_quality_benchmark"
        }
        .into(),
        dataset_id: manifest.dataset_id,
        training_dataset_id: trained.dataset_id,
        encoder_id,
        split,
        completed_training_steps: selected_step,
        evaluated_target_views: targets,
        unknown_pixels: unknown,
        min_raw_ri: min,
        max_raw_ri: max,
        covisibility: ranking_metrics(&scores)?,
        targets: target_metrics,
        mean_total_loss: loss_sums[0] / targets as f64,
        mean_cross_mse: loss_sums[1] / targets as f64,
        mean_mae_mse: loss_sums[2] / targets as f64,
        mean_ri_loss: loss_sums[3] / targets as f64,
        relative_cross_improvement: 1.0 - loss_sums[1] / loss_sums[2].max(1e-12),
        reference_mode: if options.unrelated_references {
            "unrelated_room"
        } else {
            "correct_room"
        }
        .into(),
        samples,
        rgb_uses_target_statistics: config.normalize_targets && !config.predict_patch_stats,
        mean_masked_rgb_mse: rgb_mse_sum / targets as f64,
        caveat: format!(
            "Raw RI ranked against correct-reference geometric visibility; labels held fixed for reference interventions. Unknown pixels excluded. Reconstruction loss uses {} RGB targets and one fixed mask. Training-split diagnostics do not measure generalization. Single-seed bounded diagnostics do not establish general co-visibility capability.",
            if config.normalize_targets {
                "patch-normalized"
            } else {
                "raw sRGB"
            }
        ),
    })
}

fn write_floats(path: &Path, values: &[f32]) -> Result<()> {
    ensure!(
        values.iter().all(|v| v.is_finite()),
        "nonfinite annotation export"
    );
    fs::write(
        path,
        values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ranking_oracles_include_ties_and_degenerate_classes() {
        let perfect = ranking_metrics(&[(2., true), (1., false)]).unwrap();
        assert_eq!(perfect.average_precision, Some(1.));
        assert_eq!(perfect.auroc, Some(1.));
        let reversed = ranking_metrics(&[(1., true), (2., false)]).unwrap();
        assert_eq!(reversed.average_precision, Some(0.5));
        assert_eq!(reversed.auroc, Some(0.));
        let ties = ranking_metrics(&[(1., true), (1., false)]).unwrap();
        assert_eq!(ties.average_precision, Some(0.5));
        assert_eq!(ties.auroc, Some(0.5));
        assert!(
            ranking_metrics(&[(1., false)])
                .unwrap()
                .average_precision
                .is_none()
        );
        assert!(ranking_metrics(&[(f32::NAN, true)]).is_err());
    }
}
