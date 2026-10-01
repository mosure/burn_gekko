//! Subpatch readout of RGB-derived correspondence probabilities, without geometry.
use anyhow::{Result, ensure};

/// Average probability mass in the 3x3 neighborhood of each declared hard match.
/// Coordinates are descriptor-grid indices: integer (x, y) is a patch center.
/// This changes localization, not the coarse match or its mutual-match flag.
pub fn local_coordinates(
    scores: &[f32],
    indices: &[usize],
    grid: [usize; 2],
) -> Result<Vec<[f64; 2]>> {
    let [h, w] = grid;
    let n = h.checked_mul(w).unwrap_or(0);
    ensure!(
        n > 0
            && scores.len() == n * n
            && indices.len() == n
            && indices.iter().all(|&i| i < n)
            && scores.iter().all(|&v| v.is_finite() && v >= 0.),
        "invalid subpatch probabilities or grid"
    );
    indices
        .iter()
        .enumerate()
        .map(|(query, &index)| {
            let (cx, cy) = (index % w, index / w);
            let mut mass = 0.;
            let mut point = [0.; 2];
            for y in cy.saturating_sub(1)..=(cy + 1).min(h - 1) {
                for x in cx.saturating_sub(1)..=(cx + 1).min(w - 1) {
                    let p = scores[query * n + y * w + x] as f64;
                    mass += p;
                    point[0] += p * x as f64;
                    point[1] += p * y as f64;
                }
            }
            ensure!(mass > 0., "empty local probability mass");
            Ok([point[0] / mass, point[1] / mass])
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fractional_translation_edges_and_distant_modes() {
        // A correct bilinear target at x=1.25 should recover the fractional point,
        // while a distant secondary mode must not drag the local estimate away.
        let row = [0., 0.75, 0.25, 0., 0.7];
        let p = local_coordinates(&row.repeat(5), &[1; 5], [1, 5]).unwrap();
        assert_eq!(p, vec![[1.25, 0.]; 5]);
        let row = [0.75, 0.25];
        assert_eq!(
            local_coordinates(&row.repeat(2), &[0; 2], [1, 2]).unwrap(),
            vec![[0.25, 0.]; 2]
        );
        assert!(local_coordinates(&[0.; 4], &[0; 2], [1, 2]).is_err());
        assert!(local_coordinates(&[f32::NAN; 4], &[0; 2], [1, 2]).is_err());
        assert!(local_coordinates(&[-1.; 4], &[0; 2], [1, 2]).is_err());
    }
}
