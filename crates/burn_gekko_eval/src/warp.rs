//! Image-transform diagnostics, distinct from real viewpoint-transfer evidence.
use anyhow::{Result, ensure};
use serde::Serialize;

#[derive(Debug, Serialize)]
pub struct GridError {
    pub valid_queries: usize,
    pub mean_epe: f64,
    pub pck_half_patch: f64,
    pub pck_one_patch: f64,
    /// Query index, predicted point, known transformed point, error in pixels.
    pub points: Vec<(usize, [f64; 2], [f64; 2], f64)>,
}

/// The nonzero soft-label row's barycenter is the exact transformed query point.
pub fn score(
    indices: &[usize],
    labels: &[f32],
    grid: [usize; 2],
    patch: usize,
) -> Result<GridError> {
    let [h, w] = grid;
    let n = h * w;
    ensure!(
        n > 0 && indices.len() == n && indices.iter().all(|i| *i < n),
        "invalid warp indices"
    );
    let coordinates = indices
        .iter()
        .map(|i| [(i % w) as f64, (i / w) as f64])
        .collect::<Vec<_>>();
    score_coordinates(&coordinates, labels, grid, patch)
}

/// Fractional patch coordinates use integer grid indices for patch centers.
pub fn score_coordinates(
    coordinates: &[[f64; 2]],
    labels: &[f32],
    grid: [usize; 2],
    patch: usize,
) -> Result<GridError> {
    let [h, w] = grid;
    let n = h * w;
    ensure!(
        n > 0
            && patch > 0
            && coordinates.len() == n
            && labels.len() == n * n
            && coordinates.iter().all(|p| p[0].is_finite()
                && p[1].is_finite()
                && p[0] >= 0.
                && p[0] <= (w - 1) as f64
                && p[1] >= 0.
                && p[1] <= (h - 1) as f64)
            && labels.iter().all(|p| p.is_finite() && *p >= 0.),
        "invalid warp scoring inputs"
    );
    let center = |i: usize| {
        [
            (i % w) as f64 * patch as f64 + patch as f64 / 2.,
            (i / w) as f64 * patch as f64 + patch as f64 / 2.,
        ]
    };
    let mut points = Vec::new();
    for (query, row) in labels.chunks_exact(n).enumerate() {
        let mass: f64 = row.iter().map(|x| *x as f64).sum();
        if mass == 0. {
            continue;
        }
        ensure!(
            (mass - 1.).abs() < 1e-5,
            "warp target row is not normalized"
        );
        let mut truth = [0.; 2];
        for (key, p) in row.iter().enumerate() {
            for (t, x) in truth.iter_mut().zip(center(key)) {
                *t += x * *p as f64 / mass;
            }
        }
        let prediction = coordinates[query].map(|x| (x + 0.5) * patch as f64);
        let error = (prediction[0] - truth[0]).hypot(prediction[1] - truth[1]);
        points.push((query, prediction, truth, error));
    }
    ensure!(!points.is_empty(), "no valid warp queries");
    let count = points.len();
    Ok(GridError {
        valid_queries: count,
        mean_epe: points.iter().map(|p| p.3).sum::<f64>() / count as f64,
        pck_half_patch: points.iter().filter(|p| p.3 <= patch as f64 / 2.).count() as f64
            / count as f64,
        pck_one_patch: points.iter().filter(|p| p.3 <= patch as f64).count() as f64 / count as f64,
        points,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fractional_location_and_invalid_rows_have_exact_geometric_error() {
        let labels = [0.75, 0.25, 0., 0.];
        let report = score(&[0, 1], &labels, [1, 2], 16).unwrap();
        assert_eq!(report.valid_queries, 1);
        assert_eq!(report.mean_epe, 4.);
        assert_eq!(report.points[0].2, [12., 8.]);
        assert_eq!(score(&[1, 0], &labels, [1, 2], 16).unwrap().mean_epe, 12.);
        let fine = score_coordinates(&[[0.25, 0.], [1., 0.]], &labels, [1, 2], 16).unwrap();
        assert_eq!(fine.mean_epe, 0.);
        assert_eq!(fine.points[0].1, [12., 8.]);
        assert!(score_coordinates(&[[f64::NAN, 0.], [1., 0.]], &labels, [1, 2], 16).is_err());
        assert!(score(&[0, 1], &[0.; 4], [1, 2], 16).is_err());
    }
}
