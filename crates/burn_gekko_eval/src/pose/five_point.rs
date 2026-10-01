//! Checked adapter to the MIT-licensed vision-geometry minimal calibrated solver.
use nalgebra::{Matrix3, Point2, SMatrix};

pub(super) fn candidates(matches: &[[[f64; 2]; 2]]) -> Vec<Matrix3<f64>> {
    if matches.len() != 5 || !matches.iter().flatten().flatten().all(|v| v.is_finite()) {
        return vec![];
    }
    // Reject duplicate/collinear rank-deficient samples before the polynomial
    // solver. Its five rows must be independent; geometry must not be guessed.
    let a = SMatrix::<f64, 5, 9>::from_fn(|i, j| {
        let [[x, y], [u, v]] = matches[i];
        [u * x, u * y, u, v * x, v * y, v, x, y, 1.][j]
    });
    let gram = a * a.transpose();
    if !a.iter().chain(gram.iter()).all(|v| v.is_finite()) {
        return vec![];
    }
    let Some(eig) = nalgebra::linalg::SymmetricEigen::try_new(gram, 1e-12, 1000) else {
        return vec![];
    };
    if !eig.eigenvalues.iter().all(|v| v.is_finite())
        || eig.eigenvalues.min() <= eig.eigenvalues.max().abs() * 1e-10
    {
        return vec![];
    }
    let a = matches
        .iter()
        .map(|p| Point2::from(p[0]))
        .collect::<Vec<_>>();
    let b = matches
        .iter()
        .map(|p| Point2::from(p[1]))
        .collect::<Vec<_>>();
    vision_geometry::epipolar::essential_5point(&a, &b)
        .unwrap_or_default()
        .into_iter()
        .filter_map(|e| {
            let norm = e.norm();
            if !norm.is_finite() || norm < 1e-12 {
                return None;
            }
            let e = e / norm;
            let eet = e * e.transpose();
            // Reject numerically invalid algebraic roots before consensus scoring.
            ((2. * eet * e - eet.trace() * e).norm() <= 1e-6
                && e.determinant().abs() <= 1e-8
                && matches
                    .iter()
                    .all(|&p| super::solver::sampson(&e, p) <= 1e-10))
            .then_some(e)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn minimal_roots_include_known_pose_across_rotation_and_translation_axes() {
        for k in 0..16 {
            let r = nalgebra::Rotation3::from_euler_angles(
                (k as f64 - 8.) * 0.015,
                0.17 - k as f64 * 0.01,
                k as f64 * 0.007,
            );
            let t = nalgebra::Vector3::new(0.3 + k as f64 * 0.01, -0.1 + k as f64 * 0.014, 0.08);
            let matches = (0..5)
                .map(|i| {
                    let x = nalgebra::Vector3::new(
                        ((i * 31 + 7 * k) % 97) as f64 / 48. - 1.,
                        ((i * 43 + 11 * k) % 89) as f64 / 44. - 1.,
                        3. + ((i * 17 + 13 * k) % 73) as f64 / 25.,
                    );
                    let y = r * x + t;
                    [[x.x / x.z, x.y / x.z], [y.x / y.z, y.y / y.z]]
                })
                .collect::<Vec<_>>();
            let cross = Matrix3::new(0., -t.z, t.y, t.z, 0., -t.x, -t.y, t.x, 0.);
            let truth = cross * r.matrix();
            let truth = truth / truth.norm();
            let error = candidates(&matches)
                .iter()
                .map(|e| (e - truth).norm().min((e + truth).norm()))
                .reduce(f64::min)
                .unwrap_or(f64::INFINITY);
            assert!(error < 1e-5, "fixture {k}, matrix error {error}");
        }
    }
    #[test]
    fn invalid_or_rank_deficient_samples_have_no_hypotheses() {
        assert!(candidates(&[]).is_empty());
        assert!(candidates(&[[[0., 0.], [1., 0.]]; 5]).is_empty());
        assert!(candidates(&[[[f64::NAN, 0.], [1., 0.]]; 5]).is_empty());
        assert!(candidates(&[[[1e200, 1.], [1e200, 0.]]; 5]).is_empty());
        let line = (0..5)
            .map(|i| [[i as f64, 0.], [i as f64 + 1., 0.]])
            .collect::<Vec<_>>();
        assert!(candidates(&line).is_empty());
    }
}
