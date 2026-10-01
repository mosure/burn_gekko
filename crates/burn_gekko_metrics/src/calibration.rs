//! Camera labels shared by offline evaluation and interactive inference.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

pub const CAMERA_FRAME: &str = "Bevy anchor camera: +X right, +Y up, -Z forward; R maps target axes into reference axes; translation is reference-to-target direction in reference axes";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraTarget {
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
    pub focal: [f64; 2],
}
impl CameraTarget {
    pub fn regression(&self) -> [f32; 11] {
        let n = self.translation.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mut y = [0.; 11];
        for c in 0..2 {
            for r in 0..3 {
                y[c * 3 + r] = self.rotation[r][c] as f32;
            }
        }
        if n >= 1e-8 {
            for i in 0..3 {
                y[6 + i] = (self.translation[i] / n) as f32;
            }
        }
        y[9] = self.focal[0].ln() as f32;
        y[10] = self.focal[1].ln() as f32;
        y
    }
    pub fn translation_valid(&self) -> bool {
        self.translation.iter().map(|x| x * x).sum::<f64>() >= 1e-16
    }
}

/// Column-major world-from-view matrices from the renderer. Metric scale is only a label.
pub fn camera_target(
    target: &[f32; 16],
    reference: &[f32; 16],
    fovy: f32,
    width: usize,
    height: usize,
) -> Result<CameraTarget> {
    ensure!(
        width > 0 && height > 0 && fovy > 0. && fovy < std::f32::consts::PI,
        "invalid calibration dimensions"
    );
    ensure!(
        target.iter().chain(reference).all(|v| v.is_finite()),
        "nonfinite camera transform"
    );
    for matrix in [target, reference] {
        for i in 0..3 {
            for j in 0..3 {
                let dot: f64 = (0..3)
                    .map(|k| matrix[i * 4 + k] as f64 * matrix[j * 4 + k] as f64)
                    .sum();
                ensure!(
                    (dot - f64::from(i == j)).abs() < 1e-4,
                    "camera axes are not orthonormal"
                );
            }
        }
    }
    let mut rotation = [[0.; 3]; 3];
    let mut translation = [0.; 3];
    for i in 0..3 {
        for j in 0..3 {
            rotation[i][j] = (0..3)
                .map(|k| reference[i * 4 + k] as f64 * target[j * 4 + k] as f64)
                .sum();
        }
        translation[i] = (0..3)
            .map(|k| reference[i * 4 + k] as f64 * (target[12 + k] - reference[12 + k]) as f64)
            .sum();
    }
    let focal = height as f64 / (2. * (fovy as f64 * 0.5).tan());
    Ok(CameraTarget {
        rotation,
        translation,
        focal: [focal / width as f64, focal / height as f64],
    })
}
