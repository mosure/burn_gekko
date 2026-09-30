//! Image-only projective augmentation. Coordinates use pixel edges: pixel
//! centers are (x + 0.5, y + 0.5), patch centers are patch * (column + 0.5).
//! These transforms are generated from RGB; no renderer geometry is required.
use anyhow::{Result, ensure};

#[derive(Clone, Copy, Debug)]
pub struct Homography(pub [[f64; 3]; 3]);

impl Homography {
    pub fn map(self, [x, y]: [f64; 2]) -> Option<[f64; 2]> {
        let q = self.0.map(|r| r[0] * x + r[1] * y + r[2]);
        if !q.iter().all(|v| v.is_finite()) || q[2].abs() <= 1e-10 {
            return None;
        }
        let p = [q[0] / q[2], q[1] / q[2]];
        p.iter().all(|v| v.is_finite()).then_some(p)
    }

    pub fn inverse(self) -> Result<Self> {
        ensure!(
            self.0.iter().flatten().all(|v| v.is_finite()),
            "nonfinite homography"
        );
        let m = self.0;
        let mut c = [[0.; 3]; 3];
        for (i, row) in c.iter_mut().enumerate() {
            for (j, x) in row.iter_mut().enumerate() {
                *x = m[(j + 1) % 3][(i + 1) % 3] * m[(j + 2) % 3][(i + 2) % 3]
                    - m[(j + 1) % 3][(i + 2) % 3] * m[(j + 2) % 3][(i + 1) % 3];
            }
        }
        let det = (0..3).map(|j| m[0][j] * c[j][0]).sum::<f64>();
        ensure!(det.is_finite() && det.abs() > 1e-10, "singular homography");
        for value in c.iter_mut().flatten() {
            *value /= det;
        }
        Ok(Self(c))
    }

    /// Convert a transform on [-1,1]^2 into the declared pixel-edge convention.
    pub fn from_normalized(m: [[f64; 3]; 3], width: usize, height: usize) -> Result<Self> {
        ensure!(width > 0 && height > 0, "empty image");
        let [sx, sy] = [width as f64 / 2., height as f64 / 2.];
        let mul = |a: [[f64; 3]; 3], b: [[f64; 3]; 3]| {
            std::array::from_fn(|i| {
                std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum())
            })
        };
        let result = Self(mul(
            mul([[sx, 0., sx], [0., sy, sy], [0., 0., 1.]], m),
            [[1. / sx, 0., -1.], [0., 1. / sy, -1.], [0., 0., 1.]],
        ));
        result.inverse()?;
        Ok(result)
    }
}

/// Inverse bilinear sampling with constant gray outside the source image.
pub fn warp_rgb(rgb: &[f32], width: usize, height: usize, forward: Homography) -> Result<Vec<f32>> {
    ensure!(
        width > 0
            && height > 0
            && rgb.len() == width * height * 3
            && rgb.iter().all(|v| v.is_finite() && (0. ..=1.).contains(v)),
        "invalid RGB image"
    );
    let inverse = forward.inverse()?;
    let mut out = vec![0.5; rgb.len()];
    for y in 0..height {
        for x in 0..width {
            let Some([px, py]) = inverse.map([x as f64 + 0.5, y as f64 + 0.5]) else {
                continue;
            };
            if px < 0.5 || py < 0.5 || px > width as f64 - 0.5 || py > height as f64 - 0.5 {
                continue;
            }
            let (u, v) = (px - 0.5, py - 0.5);
            let (a, b) = (u.floor() as usize, v.floor() as usize);
            let (xx, yy) = ((a + 1).min(width - 1), (b + 1).min(height - 1));
            let (dx, dy) = (u - a as f64, v - b as f64);
            for channel in 0..3 {
                out[(y * width + x) * 3 + channel] = [
                    (a, b, (1. - dx) * (1. - dy)),
                    (xx, b, dx * (1. - dy)),
                    (a, yy, (1. - dx) * dy),
                    (xx, yy, dx * dy),
                ]
                .into_iter()
                .map(|(a, b, w)| rgb[(b * width + a) * 3 + channel] as f64 * w)
                .sum::<f64>() as f32;
            }
        }
    }
    Ok(out)
}

#[derive(Debug)]
pub struct GridTargets {
    /// Row distributions; invalid query rows contain only zeros.
    pub probabilities: Vec<f32>,
    pub valid_queries: usize,
}

/// Known point locations become bilinear soft labels on the descriptor grid.
/// Entire query patches must map inside the destination; border fill is never a positive.
pub fn grid_targets(
    h: Homography,
    width: usize,
    height: usize,
    patch: usize,
) -> Result<GridTargets> {
    ensure!(
        patch > 0
            && width > 0
            && height > 0
            && width.is_multiple_of(patch)
            && height.is_multiple_of(patch),
        "invalid patch grid"
    );
    h.inverse()?;
    let (gw, gh) = (width / patch, height / patch);
    let n = gw * gh;
    let mut probabilities = vec![0.; n * n];
    let mut valid_queries = 0;
    for y in 0..gh {
        for x in 0..gw {
            let inside = [
                [x * patch, y * patch],
                [(x + 1) * patch, y * patch],
                [x * patch, (y + 1) * patch],
                [(x + 1) * patch, (y + 1) * patch],
            ]
            .into_iter()
            .all(|[a, b]| {
                h.map([a as f64, b as f64]).is_some_and(|[u, v]| {
                    u >= 0. && v >= 0. && u <= width as f64 && v <= height as f64
                })
            });
            if !inside {
                continue;
            }
            let Some([u, v]) = h.map([
                (x as f64 + 0.5) * patch as f64,
                (y as f64 + 0.5) * patch as f64,
            ]) else {
                continue;
            };
            let (u, v) = (u / patch as f64 - 0.5, v / patch as f64 - 0.5);
            if u < 0. || v < 0. || u > (gw - 1) as f64 || v > (gh - 1) as f64 {
                continue;
            }
            let (a, b) = (u.floor() as usize, v.floor() as usize);
            let (xx, yy) = ((a + 1).min(gw - 1), (b + 1).min(gh - 1));
            let (dx, dy) = (u - a as f64, v - b as f64);
            let row = (y * gw + x) * n;
            for (a, b, w) in [
                (a, b, (1. - dx) * (1. - dy)),
                (xx, b, dx * (1. - dy)),
                (a, yy, (1. - dx) * dy),
                (xx, yy, dx * dy),
            ] {
                probabilities[row + b * gw + a] += w as f32;
            }
            valid_queries += 1;
        }
    }
    Ok(GridTargets {
        probabilities,
        valid_queries,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inverse_warp_and_labels_agree_for_identity_and_one_patch_translation() {
        let rgb = (0..64 * 48 * 3)
            .map(|i| (i % 251) as f32 / 251.)
            .collect::<Vec<_>>();
        let identity = Homography([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
        assert_eq!(warp_rgb(&rgb, 64, 48, identity).unwrap(), rgb);
        let labels = grid_targets(identity, 64, 48, 16).unwrap();
        assert_eq!(labels.valid_queries, 12);
        assert!((0..12).all(|i| labels.probabilities[i * 12 + i] == 1.));
        let right = Homography([[1., 0., 16.], [0., 1., 0.], [0., 0., 1.]]);
        let warped = warp_rgb(&rgb, 64, 48, right).unwrap();
        assert_eq!(&warped[16 * 3..64 * 3], &rgb[..48 * 3]);
        let forward = grid_targets(right, 64, 48, 16).unwrap();
        let backward = grid_targets(right.inverse().unwrap(), 64, 48, 16).unwrap();
        assert_eq!(forward.valid_queries, 9);
        for y in 0..3 {
            for x in 0..3 {
                let (a, b) = (y * 4 + x, y * 4 + x + 1);
                assert_eq!(forward.probabilities[a * 12 + b], 1.);
                assert_eq!(backward.probabilities[b * 12 + a], 1.);
            }
        }
        assert!(Homography([[0.; 3]; 3]).inverse().is_err());
    }
    #[test]
    fn subpatch_translation_has_fractional_labels_and_projective_inverse_roundtrips() {
        let shift = Homography([[1., 0., 4.], [0., 1., 8.], [0., 0., 1.]]);
        let t = grid_targets(shift, 64, 64, 16).unwrap();
        assert_eq!(
            &t.probabilities[0..6],
            &[0.375, 0.125, 0., 0., 0.375, 0.125]
        );
        let h = Homography::from_normalized(
            [[0.96, -0.2, 0.1], [0.2, 0.96, -0.05], [0.02, -0.03, 1.]],
            256,
            192,
        )
        .unwrap();
        for p in [[0.5, 0.5], [128., 96.], [240.5, 180.5]] {
            let q = h.inverse().unwrap().map(h.map(p).unwrap()).unwrap();
            assert!((q[0] - p[0]).abs().max((q[1] - p[1]).abs()) < 1e-9);
        }
    }
}
