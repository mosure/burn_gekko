//! Output-head scores. Training objectives never substitute for camera or RGB accuracy.
use crate::{camera, metrics};
use anyhow::{Result, ensure};
use burn_gekko_data::head_cache::CameraTarget;
use serde::{Deserialize, Serialize};

/// Gram-Schmidt decoding of two predicted columns. Degenerate estimates are failures.
pub fn rotation_from_six(x: &[f32]) -> Option<camera::Rotation> {
    if x.len() != 6 || x.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let normalize = |a: [f64; 3]| {
        let n = a.iter().map(|v| v * v).sum::<f64>().sqrt();
        (n >= 1e-8).then(|| a.map(|v| v / n))
    };
    let a = normalize([x[0] as f64, x[1] as f64, x[2] as f64])?;
    let b = [x[3] as f64, x[4] as f64, x[5] as f64];
    let dot = camera::dot(a, b);
    let b = normalize(std::array::from_fn(|i| b[i] - dot * a[i]))?;
    let c = [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ];
    Some(std::array::from_fn(|i| [a[i], b[i], c[i]]))
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraScore {
    pub rotation_degrees: f64,
    pub translation_degrees: Option<f64>,
    pub focal_relative_error: f64,
    pub rotation_valid: bool,
    pub focal_was_clamped: bool,
    pub regression_loss: f64,
}
pub fn camera_score(pred: &[f32], truth: &CameraTarget) -> Result<CameraScore> {
    ensure!(
        pred.len() == 11 && pred.iter().all(|v| v.is_finite()),
        "invalid camera prediction"
    );
    let rotation = rotation_from_six(&pred[..6]);
    let angle = rotation
        .map(|r| camera::rotation_degrees(r, truth.rotation))
        .transpose()?
        .unwrap_or(180.);
    let focal = [
        (pred[9] as f64).clamp(-4., 4.).exp(),
        (pred[10] as f64).clamp(-4., 4.).exp(),
    ];
    let y = truth.regression();
    let mse = |a: usize, b: usize| {
        (a..b)
            .map(|i| (pred[i] as f64 - y[i] as f64).powi(2))
            .sum::<f64>()
            / (b - a) as f64
    };
    Ok(CameraScore {
        rotation_degrees: angle,
        translation_degrees: camera::translation_degrees(
            [pred[6] as f64, pred[7] as f64, pred[8] as f64],
            truth.translation,
        )?,
        focal_relative_error: camera::focal_relative_error(focal, truth.focal)?,
        rotation_valid: rotation.is_some(),
        focal_was_clamped: pred[9..].iter().any(|v| !(-4. ..=4.).contains(v)),
        regression_loss: mse(0, 6)
            + if truth.translation_valid() {
                mse(6, 9)
            } else {
                0.
            }
            + mse(9, 11),
    })
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RgbScore {
    pub mse: f64,
    pub psnr_db: Option<f64>,
}
pub fn rgb_score(pred: &[f32], truth: &[f32], hidden: &[usize]) -> Result<RgbScore> {
    ensure!(
        pred.iter().all(|v| (0. ..=1.).contains(v))
            && truth.iter().all(|v| (-0.001..=1.001).contains(v)),
        "invalid RGB range"
    );
    let mse = metrics::completion(pred, truth, 768, hidden)?.mse;
    Ok(RgbScore {
        mse,
        psnr_db: metrics::psnr(mse, 1.)?,
    })
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadRow {
    pub room_seed: u64,
    pub target_view: usize,
    pub reference_view: usize,
    pub camera: CameraScore,
    pub rgb: RgbScore,
    pub monocular: RgbScore,
    pub camera_prediction: Vec<f32>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadScores {
    pub targets: usize,
    pub rooms: usize,
    pub camera_regression_loss: f64,
    pub rotation_mean_degrees: f64,
    pub translation_mean_degrees: Option<f64>,
    pub focal_mean_relative_error: f64,
    pub pose_auc_10: Option<f64>,
    pub invalid_rotations: usize,
    pub focal_clamps: usize,
    pub rgb_hidden_mse: f64,
    pub rgb_hidden_psnr_db: Option<f64>,
    pub monocular_hidden_psnr_db: Option<f64>,
    pub rows: Vec<HeadRow>,
}
pub fn summarize(rows: Vec<HeadRow>) -> Result<HeadScores> {
    ensure!(!rows.is_empty(), "empty head evaluation");
    let n = rows.len() as f64;
    let mean = |f: fn(&HeadRow) -> f64| rows.iter().map(f).sum::<f64>() / n;
    let options = |f: fn(&HeadRow) -> Option<f64>| {
        let v: Vec<_> = rows.iter().filter_map(f).collect();
        (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64)
    };
    let pose: Vec<_> = rows
        .iter()
        .filter_map(|r| {
            r.camera
                .translation_degrees
                .map(|t| t.max(r.camera.rotation_degrees))
        })
        .collect();
    Ok(HeadScores {
        targets: rows.len(),
        rooms: rows
            .iter()
            .map(|r| r.room_seed)
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        camera_regression_loss: mean(|r| r.camera.regression_loss),
        rotation_mean_degrees: mean(|r| r.camera.rotation_degrees),
        translation_mean_degrees: options(|r| r.camera.translation_degrees),
        focal_mean_relative_error: mean(|r| r.camera.focal_relative_error),
        pose_auc_10: if pose.is_empty() {
            None
        } else {
            Some(camera::pose_auc(&pose, 10.)?)
        },
        invalid_rotations: rows.iter().filter(|r| !r.camera.rotation_valid).count(),
        focal_clamps: rows.iter().filter(|r| r.camera.focal_was_clamped).count(),
        rgb_hidden_mse: mean(|r| r.rgb.mse),
        rgb_hidden_psnr_db: options(|r| r.rgb.psnr_db),
        monocular_hidden_psnr_db: options(|r| r.monocular.psnr_db),
        rows,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn degenerate_rotations_count_as_failed_and_focal_lengths_stay_positive() {
        let target = CameraTarget {
            rotation: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
            translation: [1., 0., 0.],
            focal: [1., 1.],
        };
        let mut p = [0.; 11];
        let bad = camera_score(&p, &target).unwrap();
        assert_eq!(bad.rotation_degrees, 180.);
        assert!(!bad.rotation_valid);
        p[0] = 1.;
        p[4] = 1.;
        p[6] = 1.;
        assert_eq!(camera_score(&p, &target).unwrap().rotation_degrees, 0.);
        p[9] = 1000.;
        p[10] = -1000.;
        let out = camera_score(&p, &target).unwrap();
        assert!(out.focal_was_clamped && out.focal_relative_error.is_finite());
        assert!(rotation_from_six(&[1., 0., 0., 2., 0., 0.]).is_none());
        let r = rotation_from_six(&[0., 1., 0., -1., 0., 0.]).unwrap();
        assert_eq!(camera::rotation_degrees(r, target.rotation).unwrap(), 90.);
    }
    #[test]
    fn rgb_psnr_uses_only_hidden_pixels_in_srgb_range_one() {
        let target = vec![0.; 1536];
        let mut pred = vec![1.; 1536];
        pred[..768].fill(0.1);
        let m = rgb_score(&pred, &target, &[0]).unwrap();
        assert!((m.psnr_db.unwrap() - 20.).abs() < 1e-5);
        assert!(rgb_score(&pred, &target, &[0, 0]).is_err());
    }
}
