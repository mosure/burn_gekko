//! RGB-only external pairs and separate post-inference camera labels.
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};

pub fn default_image_size() -> usize {
    256
}

/// Image probes use square, 16-pixel patch grids with bounded memory needs.
pub fn descriptor_grid(image_size: usize) -> anyhow::Result<[usize; 2]> {
    anyhow::ensure!(
        (16..=512).contains(&image_size) && image_size.is_multiple_of(16),
        "image size must be a multiple of 16, at most 512"
    );
    Ok([image_size / 16; 2])
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RgbView {
    pub file: PathBuf,
    pub sha256: String,
    pub original_file: PathBuf,
    pub original_sha256: String,
    pub original_hw: [usize; 2],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ViewPair {
    pub id: String,
    pub sequence: String,
    pub interval: usize,
    pub target: String,
    pub reference: String,
}
/// No geometry, calibration or ground truth is accepted by the model exporter.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairImages {
    pub schema: u32,
    pub image_size: usize,
    pub evaluation_use: String,
    pub images: BTreeMap<String, RgbView>,
    pub pairs: Vec<ViewPair>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoseLabel {
    pub id: String,
    /// X_reference = rotation * X_target + translation; row-major rotation.
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
    pub association_seconds: [f64; 2],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PairLabels {
    pub schema: u32,
    pub image_manifest_sha256: String,
    /// fx, fy, cx, cy in the original undistorted image coordinates.
    pub intrinsics: [f64; 4],
    pub coordinate_frame: String,
    pub rows: Vec<PoseLabel>,
}

/// Grid index zero denotes the first patch center. Preserve half-pixel resize geometry.
pub fn original_pixel(xy: [f64; 2], grid: [usize; 2], hw: [usize; 2]) -> [f64; 2] {
    [
        (xy[0] + 0.5) * hw[1] as f64 / grid[1] as f64 - 0.5,
        (xy[1] + 0.5) * hw[0] as f64 / grid[0] as f64 - 0.5,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn patch_centers_preserve_resize_coordinates_and_manifest_rejects_labels() {
        assert_eq!(original_pixel([0., 0.], [16, 16], [480, 640]), [19.5, 14.5]);
        assert_eq!(
            original_pixel([7.5, 7.5], [16, 16], [480, 640]),
            [319.5, 239.5]
        );
        assert_eq!(descriptor_grid(512).unwrap(), [32, 32]);
        assert_eq!(original_pixel([0., 0.], [32, 32], [480, 640]), [9.5, 7.]);
        assert_eq!(
            original_pixel([15.5, 15.5], [32, 32], [480, 640]),
            [319.5, 239.5]
        );
        assert!(descriptor_grid(255).is_err());
        assert!(descriptor_grid(0).is_err());
        assert!(descriptor_grid(528).is_err());
        let value = serde_json::json!({"schema":1,"image_size":256,"evaluation_use":"held_out","images":{},"pairs":[],"intrinsics":[1,1,0,0]});
        assert!(serde_json::from_value::<PairImages>(value).is_err());
    }
}
