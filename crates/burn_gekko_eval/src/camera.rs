//! Relative pose in an anchor camera frame; metric translation scale is not inferred.
//! Rotations are row-major SO(3); translation is signed anchor-to-target direction.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
pub type Rotation = [[f64; 3]; 3];
pub fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}
fn norm(a: [f64; 3]) -> f64 {
    dot(a, a).sqrt()
}
pub fn rotation_degrees(pred: Rotation, truth: Rotation) -> Result<f64> {
    for r in [pred, truth] {
        ensure!(
            r.iter().flatten().all(|x| x.is_finite()),
            "nonfinite rotation"
        );
        for a in 0..3 {
            for b in 0..3 {
                ensure!(
                    (dot(r[a], r[b]) - f64::from(a == b)).abs() < 1e-4,
                    "rotation is not orthonormal"
                );
            }
        }
        let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
            - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
            + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
        ensure!((det - 1.).abs() < 1e-4, "reflection is not SO(3)");
    }
    let trace = pred
        .iter()
        .flatten()
        .zip(truth.iter().flatten())
        .map(|(a, b)| a * b)
        .sum::<f64>();
    Ok(((trace - 1.) / 2.).clamp(-1., 1.).acos().to_degrees())
}
/// Zero ground-truth baseline is excluded; zero predictions count as failed (180 degrees).
pub fn translation_degrees(pred: [f64; 3], truth: [f64; 3]) -> Result<Option<f64>> {
    ensure!(
        pred.iter().chain(truth.iter()).all(|v| v.is_finite()),
        "nonfinite translation"
    );
    let (p, t) = (norm(pred), norm(truth));
    Ok(if t < 1e-8 {
        None
    } else if p < 1e-8 {
        Some(180.)
    } else {
        Some(
            (dot(pred, truth) / (p * t))
                .clamp(-1., 1.)
                .acos()
                .to_degrees(),
        )
    })
}
pub fn focal_relative_error(pred: [f64; 2], truth: [f64; 2]) -> Result<f64> {
    ensure!(
        pred.iter()
            .chain(truth.iter())
            .all(|v| v.is_finite() && *v > 0.),
        "focal length must be positive"
    );
    Ok(((pred[0] / truth[0] - 1.).abs() + (pred[1] / truth[1] - 1.).abs()) / 2.)
}
/// Normalized trapezoidal area under the empirical pose recall curve; max(R,t) in degrees.
pub fn pose_auc(errors: &[f64], threshold: f64) -> Result<f64> {
    ensure!(
        !errors.is_empty()
            && threshold.is_finite()
            && threshold > 0.
            && errors.iter().all(|x| x.is_finite() && *x >= 0.),
        "invalid pose AUC input"
    );
    let mut e = errors.to_vec();
    e.sort_by(f64::total_cmp);
    let (mut x, mut y, mut area) = (0., 0., 0.);
    for (i, &v) in e.iter().enumerate() {
        if v > threshold {
            break;
        }
        let recall = (i + 1) as f64 / e.len() as f64;
        area += (v - x) * (y + recall) / 2.;
        x = v;
        y = recall;
    }
    Ok((area + (threshold - x) * y) / threshold)
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraRecord {
    pub sample: String,
    pub predicted_rotation: Rotation,
    pub target_rotation: Rotation,
    pub predicted_translation: [f64; 3],
    pub target_translation: [f64; 3],
    /// fx / image width, fy / image height, with centered principal point.
    pub predicted_focal: [f64; 2],
    pub target_focal: [f64; 2],
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraMetrics {
    pub samples: usize,
    pub nonzero_baselines: usize,
    pub rotation_mean_degrees: f64,
    pub translation_mean_degrees: Option<f64>,
    pub focal_mean_relative_error: f64,
    pub pose_auc_5: Option<f64>,
    pub pose_auc_10: Option<f64>,
    pub pose_auc_20: Option<f64>,
}
pub fn evaluate(rows: &[CameraRecord]) -> Result<CameraMetrics> {
    ensure!(!rows.is_empty(), "no camera observations");
    let (mut r, mut t, mut f, mut pose) = (0., Vec::new(), 0., Vec::new());
    for row in rows {
        let angle = rotation_degrees(row.predicted_rotation, row.target_rotation)?;
        r += angle;
        if let Some(v) = translation_degrees(row.predicted_translation, row.target_translation)? {
            t.push(v);
            pose.push(v.max(angle));
        }
        f += focal_relative_error(row.predicted_focal, row.target_focal)?;
    }
    let auc = |threshold| {
        if pose.is_empty() {
            Ok(None)
        } else {
            pose_auc(&pose, threshold).map(Some)
        }
    };
    Ok(CameraMetrics {
        samples: rows.len(),
        nonzero_baselines: t.len(),
        rotation_mean_degrees: r / rows.len() as f64,
        translation_mean_degrees: (!t.is_empty()).then(|| t.iter().sum::<f64>() / t.len() as f64),
        focal_mean_relative_error: f / rows.len() as f64,
        pose_auc_5: auc(5.)?,
        pose_auc_10: auc(10.)?,
        pose_auc_20: auc(20.)?,
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gauge_and_failure_conventions() {
        let i = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
        let r = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]];
        assert_eq!(rotation_degrees(i, i).unwrap(), 0.);
        assert!((rotation_degrees(r, i).unwrap() - 90.).abs() < 1e-10);
        assert_eq!(
            translation_degrees([0., 0., 0.], [1., 0., 0.]).unwrap(),
            Some(180.)
        );
        assert_eq!(
            translation_degrees([1., 0., 0.], [0., 0., 0.]).unwrap(),
            None
        );
        assert_eq!(
            translation_degrees([-2., 0., 0.], [1., 0., 0.]).unwrap(),
            Some(180.)
        );
        assert_eq!(pose_auc(&[0., 0.], 5.).unwrap(), 1.);
        assert_eq!(pose_auc(&[10., 20.], 5.).unwrap(), 0.);
        assert!((pose_auc(&[2., 4.], 5.).unwrap() - 0.6).abs() < 1e-10);
        assert!(rotation_degrees([[-1., 0., 0.], i[1], i[2]], i).is_err());
    }
}
