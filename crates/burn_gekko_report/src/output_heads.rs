//! Verified output-head evidence and annotated RGB/camera diagnostics for one run.
use crate::{
    artifact::{Source, number, pinned, record},
    correspondence::CorrespondenceFigure,
    experiment::PinnedFile,
    figures::floats,
};
use anyhow::{Context, Result, ensure};
use burn_gekko_data::head_cache::CameraTarget;
use burn_gekko_eval::schema::{Capability, CapabilityStatus, Metric};
use image::{Rgb, RgbImage};
use serde_json::Value;
use std::{collections::BTreeMap, path::Path};

fn file(v: &Value) -> Result<PinnedFile> {
    Ok(serde_json::from_value(v.clone())?)
}
fn json_file(v: &Value, sources: &mut Vec<Source>) -> Result<Value> {
    pinned(&file(v)?, sources)
}
fn bound(input: &PinnedFile, sources: &mut Vec<Source>) -> Result<Value> {
    let v = pinned(input, sources)?;
    ensure!(v["schema"] == 1, "invalid output-head evidence");
    for value in v["files"]
        .as_object()
        .context("head evidence sources")?
        .values()
    {
        let p = file(value)?;
        ensure!(
            record(&p.path, sources)? == p.sha256,
            "head source checksum mismatch"
        );
    }
    Ok(v)
}
pub fn load(
    input: &PinnedFile,
    checkpoint: &str,
    sources: &mut Vec<Source>,
) -> Result<Vec<Capability>> {
    let mut verified = BTreeMap::new();
    let burn_gekko_eval::heads::audit::AuditedHeads {
        cache,
        report,
        steps,
        scores,
    } = burn_gekko_eval::heads::audit::load(
        &burn_gekko_eval::adaptation::Input {
            path: input.path.clone(),
            sha256: input.sha256.clone(),
        },
        checkpoint,
        &mut verified,
    )?;
    for (path, expected) in verified {
        ensure!(
            record(&path, sources)? == expected,
            "head evidence changed after verification"
        );
    }
    let metric = |id: &str, label: &str, value: f64, unit: &str, lower| {
        Metric{id:id.into(),label:label.into(),value,unit:unit.into(),lower_is_better:lower,samples:scores.targets,aggregation:"equal target views, all declared validation rooms; RGB scores use hidden sRGB pixels only".into()}
    };
    let mut camera = vec![
        metric(
            "rotation",
            "Relative rotation error",
            scores.rotation_mean_degrees,
            "degrees",
            true,
        ),
        metric(
            "focal",
            "Focal length relative error",
            scores.focal_mean_relative_error,
            "fraction",
            true,
        ),
    ];
    if let Some(value) = scores.translation_mean_degrees {
        camera.push(metric(
            "translation",
            "Signed translation-direction error",
            value,
            "degrees",
            true,
        ));
    }
    if let Some(value) = scores.pose_auc_10 {
        camera.push(metric(
            "pose_auc10",
            "Pose AUC at 10 degrees",
            value,
            "fraction",
            false,
        ));
    }
    let constant = &report["constant_camera_validation"];
    let limitations = vec![
        format!(
            "{} development rooms, {} target/reference pairs; one training seed. Dense RGB pair input, independent from sparse completion. Principal point fixed at the image center; translation scale is not predicted. {} invalid rotations; {} focal clamps.",
            scores.rooms, scores.targets, scores.invalid_rotations, scores.focal_clamps
        ),
        format!(
            "Training-label constant control: rotation {:.2} degrees, translation {:.2} degrees, focal error {:.2}%. Stable regression alone does not prove useful camera estimation.",
            number(constant, "rotation_mean_degrees")?,
            number(constant, "translation_mean_degrees")?,
            number(constant, "focal_mean_relative_error")? * 100.
        ),
    ];
    let mut rgb = Vec::new();
    if let Some(value) = scores.rgb_hidden_psnr_db {
        rgb.push(metric("psnr", "Hidden-pixel RGB PSNR", value, "dB", false));
    }
    if let Some(value) = scores.monocular_hidden_psnr_db {
        rgb.push(metric(
            "mono_psnr",
            "References-disabled RGB PSNR",
            value,
            "dB",
            false,
        ));
    }
    rgb.push(metric(
        "mse",
        "Hidden-pixel RGB MSE",
        scores.rgb_hidden_mse,
        "squared sRGB units",
        true,
    ));
    let gates = report["stability_gates"]
        .as_object()
        .context("head stability gates")?;
    let failed: Vec<_> = gates
        .iter()
        .filter(|(_, v)| **v != true)
        .map(|(k, _)| k.clone())
        .collect();
    let status = if failed.is_empty() {
        "PASSED"
    } else {
        "FAILED"
    };
    let gradient_metric = |id: &str, label: &str, key: &str| -> Result<Metric> {
        Ok(Metric {
            id: id.into(),
            label: label.into(),
            value: number(&report, key)?,
            unit: "L2 norm".into(),
            lower_is_better: true,
            samples: steps.len(),
            aggregation: "maximum before per-head global clipping over logged optimizer updates"
                .into(),
        })
    };
    let out=vec![
        Capability{id:"camera".into(),label:"Learned camera calibration head".into(),status:CapabilityStatus::Evaluated,protocol:format!("Head weights trained from scratch on a frozen foundation; {}. Six-dimensional rotation columns, signed unit-baseline regression, log-positive focal lengths. No supplied calibration enters inference.",cache.camera_frame),limitations,metrics:camera},
        Capability{id:"rgb_completion".into(),label:"Learned RGB reconstruction head".into(),status:CapabilityStatus::Evaluated,protocol:"One trained 16x16 patch decoder shared by cross-view and references-disabled latent predictions. sRGB [0,1], data range 1, only originally hidden pixels scored; PSNR computed per view then averaged. Ground-truth pixels are never pasted into scored predictions.".into(),limitations:vec![format!("{} development rooms; fixed foundation and one head-training seed. This small head study does not establish sharp completion or real-scene transfer. Colored latent maps elsewhere remain feature visualizations.",scores.rooms)],metrics:rgb},
        Capability{id:"output_head_stability".into(),label:"Camera and RGB training stability".into(),status:CapabilityStatus::Evaluated,protocol:format!("{status}: fixed schedule, independent optimizers, warmup, global gradient clipping at 1 per head, finite-value checks and saved-weight replay. Failed gates: {failed:?}. Final checkpoint only; no validation selection."),limitations:vec![format!("Frozen-feature CPU phase: {} updates in {:.1} seconds. Initial/final validation RGB PSNR {:.2}/{:.2} dB; camera regression loss {:.4}/{:.4}. This verifies bounded optimization, not full encoder/fusion joint-training stability.",report["completed_steps"],number(&report,"training_seconds")?,number(&report["initial_validation"],"rgb_hidden_psnr_db")?,scores.rgb_hidden_psnr_db.unwrap_or(0.),number(&report["initial_validation"],"camera_regression_loss")?,scores.camera_regression_loss)],metrics:vec![gradient_metric("camera_gradient","Maximum camera gradient norm before clipping","max_camera_gradient_norm")?,gradient_metric("rgb_gradient","Maximum RGB gradient norm before clipping","max_rgb_gradient_norm")?]}
    ];
    for cap in &out {
        cap.validate()?;
    }
    Ok(out)
}

fn image(patches: &[f32], grid: [usize; 2]) -> Result<RgbImage> {
    ensure!(
        patches.len() == grid[0] * grid[1] * 768,
        "head visual shape"
    );
    let mut image = RgbImage::new((grid[1] * 16) as u32, (grid[0] * 16) as u32);
    for y in 0..grid[0] * 16 {
        for x in 0..grid[1] * 16 {
            let i = ((y / 16 * grid[1] + x / 16) * 256 + (y % 16 * 16 + x % 16)) * 3;
            image.put_pixel(
                x as u32,
                y as u32,
                Rgb(std::array::from_fn(|c| {
                    (patches[i + c].clamp(0., 1.) * 255.).round() as u8
                })),
            );
        }
    }
    Ok(image)
}
pub fn figures(
    input: &PinnedFile,
    out: &Path,
    sources: &mut Vec<Source>,
) -> Result<Vec<CorrespondenceFigure>> {
    let bound = bound(input, sources)?;
    let predictions = json_file(&bound["files"]["predictions"], sources)?;
    let grid: [usize; 2] = serde_json::from_value(predictions["grid"].clone())?;
    let samples = predictions["samples"].as_array().context("head examples")?;
    let mut figures = Vec::new();
    for (i, index) in [0, samples.len() / 2, samples.len() - 1]
        .into_iter()
        .enumerate()
    {
        let s = &samples[index];
        let w = grid[1] * 16;
        let h = grid[0] * 16;
        let data = |name: &str, sources: &mut Vec<Source>| -> Result<Vec<f32>> {
            floats(&file(&s["files"][name])?.path, sources)
        };
        let target = data("target", sources)?;
        let prediction = data("prediction", sources)?;
        let mono = data("monocular", sources)?;
        let mut masked = target.clone();
        for token in s["hidden_tokens"].as_array().context("head visual mask")? {
            masked[token.as_u64().unwrap() as usize * 768
                ..(token.as_u64().unwrap() as usize + 1) * 768]
                .fill(0.08);
        }
        let mut canvas = RgbImage::from_pixel((w * 4 + 24) as u32, h as u32, Rgb([235, 240, 244]));
        for (column, values) in [&masked, &target, &prediction, &mono]
            .into_iter()
            .enumerate()
        {
            image::imageops::replace(
                &mut canvas,
                &image(values, grid)?,
                (column * (w + 8)) as i64,
                0,
            );
        }
        let path = format!("media/output-head-rgb-{i}.png");
        canvas.save(out.join(&path))?;
        let metrics = &s["metrics"];
        figures.push(CorrespondenceFigure {title:format!("RGB head / room {}, view {}",s["room_seed"],s["target_view"]),file:path,caption:format!("Left to right: sparse target input, target RGB (evaluation only), predicted RGB, references-disabled RGB. Hidden-pixel PSNR {:.2} / {:.2} dB; the full images shown are generated by the same learned head. These fixed first/middle/last samples are not selected by quality.",number(&metrics["rgb"],"psnr_db")?,number(&metrics["monocular"],"psnr_db")?)});
        let truth: CameraTarget = serde_json::from_value(s["camera_target"].clone())?;
        let p: Vec<f32> = serde_json::from_value(metrics["camera_prediction"].clone())?;
        let mut plot = RgbImage::from_pixel(512, 256, Rgb([244, 247, 250]));
        let arrow = |plot: &mut RgbImage, v: [f64; 3], offset: f64, color| {
            let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-8);
            let xy = [
                offset + 110. + 80. * (v[0] - 0.45 * v[2]) / norm,
                128. - 80. * (v[1] + 0.3 * v[2]) / norm,
            ];
            crate::correspondence::line(plot, [offset + 110., 128.], xy, color);
            crate::correspondence::point(plot, xy[0], xy[1], color);
        };
        arrow(&mut plot, truth.translation, 0., Rgb([30, 160, 120]));
        arrow(
            &mut plot,
            [p[6] as f64, p[7] as f64, p[8] as f64],
            0.,
            Rgb([218, 127, 18]),
        );
        if let Some(rotation) = burn_gekko_eval::heads::rotation_from_six(&p[..6]) {
            for (column, _) in rotation[0].iter().enumerate() {
                arrow(
                    &mut plot,
                    std::array::from_fn(|r| truth.rotation[r][column]),
                    256.,
                    Rgb([30, 160, 120]),
                );
                arrow(
                    &mut plot,
                    std::array::from_fn(|r| rotation[r][column]),
                    256.,
                    Rgb([218, 127, 18]),
                );
            }
        }
        let file = format!("media/output-head-camera-{i}.png");
        plot.save(out.join(&file))?;
        figures.push(CorrespondenceFigure{title:format!("Camera head / room {}, view {}",s["room_seed"],s["target_view"]),file,caption:format!("Green: renderer truth; amber: prediction. Left: signed unit translation; right: target orientation axes in the reference frame, under one fixed oblique projection. This is a dense RGB-pair route. Rotation error {:.2} degrees; translation-direction error {:.2} degrees; focal relative error {:.2}%. Diagram lengths are not metric translation scale.",number(&metrics["camera"],"rotation_degrees")?,number(&metrics["camera"],"translation_degrees")?,100.*number(&metrics["camera"],"focal_relative_error")?)});
    }
    Ok(figures)
}
