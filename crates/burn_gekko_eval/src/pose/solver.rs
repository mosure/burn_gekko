//! Deterministic calibrated RANSAC; legacy eight-point and explicit five-point hypotheses.
use anyhow::{Result, ensure};
use nalgebra::{Matrix3, SMatrix, SVector, Vector3};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MinimalSolver {
    #[default]
    EightPoint,
    FivePoint,
}
impl MinimalSolver {
    pub fn sample_size(self) -> usize {
        match self {
            Self::EightPoint => 8,
            Self::FivePoint => 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SolverConfig {
    pub max_trials: usize,
    pub min_trials: usize,
    pub confidence: f64,
    /// Sampson distance in normalized camera coordinates (not squared).
    pub threshold: f64,
    pub min_inliers: usize,
    pub seed: u64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelativePose {
    /// X_reference = R * X_target + t, with unit-length translation.
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PoseFit {
    pub pose: Option<RelativePose>,
    pub failure: Option<String>,
    pub trials: usize,
    pub inliers: Vec<bool>,
    pub positive_depth_points: usize,
}
type Match = [[f64; 2]; 2];
fn bearing(x: [f64; 2]) -> Vector3<f64> {
    Vector3::new(x[0], x[1], 1.)
}

fn normalization(matches: &[Match], side: usize) -> Option<Matrix3<f64>> {
    let mut mean = [0.; 2];
    for p in matches {
        for (j, v) in mean.iter_mut().enumerate() {
            *v += p[side][j] / matches.len() as f64;
        }
    }
    let radius = matches
        .iter()
        .map(|p| ((p[side][0] - mean[0]).powi(2) + (p[side][1] - mean[1]).powi(2)).sqrt())
        .sum::<f64>()
        / matches.len() as f64;
    if radius < 1e-10 {
        return None;
    }
    let s = 2_f64.sqrt() / radius;
    Some(Matrix3::new(
        s,
        0.,
        -s * mean[0],
        0.,
        s,
        -s * mean[1],
        0.,
        0.,
        1.,
    ))
}
fn essential(matches: &[Match]) -> Option<Matrix3<f64>> {
    if matches.len() < 8 {
        return None;
    }
    let ta = normalization(matches, 0)?;
    let tb = normalization(matches, 1)?;
    let mut ata = SMatrix::<f64, 9, 9>::zeros();
    for p in matches {
        let a = ta * bearing(p[0]);
        let b = tb * bearing(p[1]);
        let row = SVector::<f64, 9>::from_row_slice(&[
            b.x * a.x,
            b.x * a.y,
            b.x,
            b.y * a.x,
            b.y * a.y,
            b.y,
            a.x,
            a.y,
            1.,
        ]);
        ata += row * row.transpose();
    }
    let eig = nalgebra::linalg::SymmetricEigen::try_new(ata, 1e-12, 1000)?;
    let mut order = (0..9).collect::<Vec<_>>();
    order.sort_by(|&a, &b| eig.eigenvalues[a].total_cmp(&eig.eigenvalues[b]));
    // More than one null direction indicates a rank-deficient configuration.
    if eig.eigenvalues[order[1]] <= 1e-10 * eig.eigenvalues[order[8]].abs() {
        return None;
    }
    let v = eig.eigenvectors.column(order[0]);
    let e = tb.transpose() * Matrix3::from_row_slice(v.as_slice()) * ta;
    let svd = nalgebra::linalg::SVD::try_new(e, true, true, 1e-12, 1000)?;
    let s = (svd.singular_values[0] + svd.singular_values[1]) / 2.;
    if !s.is_finite() || s < 1e-12 {
        return None;
    }
    let e = svd.u? * Matrix3::from_diagonal(&Vector3::new(s, s, 0.)) * svd.v_t?;
    let norm = e.norm();
    (norm.is_finite() && norm > 1e-12).then(|| e / norm)
}
pub(crate) fn sampson(e: &Matrix3<f64>, p: Match) -> f64 {
    let a = bearing(p[0]);
    let b = bearing(p[1]);
    let ea = e * a;
    let etb = e.transpose() * b;
    let denom = ea.x * ea.x + ea.y * ea.y + etb.x * etb.x + etb.y * etb.y;
    if denom < 1e-20 {
        return f64::INFINITY;
    }
    b.dot(&ea).powi(2) / denom
}
fn classify(e: &Matrix3<f64>, matches: &[Match], threshold2: f64) -> (Vec<bool>, usize, f64) {
    let errors = matches.iter().map(|&p| sampson(e, p)).collect::<Vec<_>>();
    let inliers = errors.iter().map(|&v| v <= threshold2).collect::<Vec<_>>();
    let count = inliers.iter().filter(|&&x| x).count();
    let cost = errors.iter().map(|v| v.min(threshold2)).sum();
    (inliers, count, cost)
}
fn positive_depth(r: &Matrix3<f64>, t: &Vector3<f64>, p: Match) -> bool {
    let a = r * bearing(p[0]);
    let b = bearing(p[1]);
    let aa = a.dot(&a);
    let bb = b.dot(&b);
    let ab = -a.dot(&b);
    let det = aa * bb - ab * ab;
    if det < 1e-12 {
        return false;
    }
    let ra = -a.dot(t);
    let rb = b.dot(t);
    let za = (ra * bb - ab * rb) / det;
    let zb = (aa * rb - ab * ra) / det;
    za > 1e-6 && zb > 1e-6
}
fn recover(e: Matrix3<f64>, matches: &[Match], mask: &[bool]) -> Option<(RelativePose, usize)> {
    let svd = nalgebra::linalg::SVD::try_new(e, true, true, 1e-12, 1000)?;
    let mut u = svd.u?;
    let mut vt = svd.v_t?;
    if u.determinant() < 0. {
        u.column_mut(2).neg_mut();
    }
    if vt.determinant() < 0. {
        vt.row_mut(2).neg_mut();
    }
    let w = Matrix3::new(0., -1., 0., 1., 0., 0., 0., 0., 1.);
    let mut best = None;
    let mut best_count = 0;
    for r in [u * w * vt, u * w.transpose() * vt] {
        for sign in [1., -1.] {
            let t = u.column(2).into_owned() * sign;
            let count = matches
                .iter()
                .zip(mask)
                .filter(|(p, m)| **m && positive_depth(&r, &t, **p))
                .count();
            if count > best_count {
                best_count = count;
                best = Some(RelativePose {
                    rotation: std::array::from_fn(|i| std::array::from_fn(|j| r[(i, j)])),
                    translation: [t.x, t.y, t.z],
                });
            }
        }
    }
    let required = 8.max(mask.iter().filter(|&&v| v).count().div_ceil(2));
    best.filter(|_| best_count >= required)
        .map(|p| (p, best_count))
}
struct Rng(u64);
impl Rng {
    fn index(&mut self, n: usize) -> usize {
        let n = n as u64;
        loop {
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut v = self.0;
            v = (v ^ (v >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            v = (v ^ (v >> 27)).wrapping_mul(0x94d049bb133111eb);
            v ^= v >> 31;
            if v >= n.wrapping_neg() % n {
                return (v % n) as usize;
            }
        }
    }
}
pub fn estimate(matches: &[Match], c: &SolverConfig) -> Result<PoseFit> {
    estimate_with(matches, c, MinimalSolver::EightPoint)
}

/// Change only the minimal hypothesis generator. Consensus scoring, optional
/// eight-point final refit and signed cheirality remain identical in both modes.
pub fn estimate_with(
    matches: &[Match],
    c: &SolverConfig,
    solver: MinimalSolver,
) -> Result<PoseFit> {
    ensure!(
        c.max_trials >= c.min_trials
            && c.min_trials > 0
            && c.max_trials <= 100_000
            && c.min_inliers >= 8
            && c.threshold.is_finite()
            && c.threshold > 0.
            && c.confidence > 0.
            && c.confidence < 1.,
        "invalid pose solver settings"
    );
    ensure!(
        matches.iter().flatten().flatten().all(|x| x.is_finite()),
        "nonfinite correspondences"
    );
    let mut fit = PoseFit {
        pose: None,
        failure: Some("insufficient_correspondences".into()),
        trials: 0,
        inliers: vec![false; matches.len()],
        positive_depth_points: 0,
    };
    if matches.len() < c.min_inliers {
        return Ok(fit);
    }
    let mut rng = Rng(c.seed);
    let mut limit = c.max_trials;
    let mut best = None;
    let mut best_count = 0;
    let mut best_cost = f64::INFINITY;
    for trial in 0..c.max_trials {
        if trial >= limit {
            break;
        }
        fit.trials = trial + 1;
        let sample_size = solver.sample_size();
        let mut indices = Vec::with_capacity(sample_size);
        while indices.len() < sample_size {
            let i = rng.index(matches.len());
            if !indices.contains(&i) {
                indices.push(i);
            }
        }
        let subset = indices.iter().map(|&i| matches[i]).collect::<Vec<_>>();
        let candidates = match solver {
            MinimalSolver::EightPoint => essential(&subset).into_iter().collect(),
            MinimalSolver::FivePoint => super::five_point::candidates(&subset),
        };
        for e in candidates {
            let (mask, count, cost) = classify(&e, matches, c.threshold * c.threshold);
            if count > best_count || (count == best_count && cost < best_cost) {
                best = Some(e);
                fit.inliers = mask;
                best_count = count;
                best_cost = cost;
                let probability = (count as f64 / matches.len() as f64).powi(sample_size as i32);
                if probability > 0. {
                    let needed = if probability >= 1. {
                        c.min_trials
                    } else {
                        ((1. - c.confidence).ln() / (-probability).ln_1p()).ceil() as usize
                    };
                    limit = limit.min(needed.max(c.min_trials));
                }
            }
        }
    }
    fit.failure = Some("insufficient_inliers_or_degenerate_geometry".into());
    if best_count < c.min_inliers {
        return Ok(fit);
    }
    let mut e = best.unwrap();
    let inliers = matches
        .iter()
        .zip(&fit.inliers)
        .filter_map(|(&p, &m)| m.then_some(p))
        .collect::<Vec<_>>();
    if let Some(refit) = essential(&inliers) {
        let (mask, count, cost) = classify(&refit, matches, c.threshold * c.threshold);
        if count > best_count || (count == best_count && cost <= best_cost) {
            e = refit;
            fit.inliers = mask;
        }
    }
    fit.failure = Some("cheirality_failure".into());
    if let Some((pose, count)) = recover(e, matches, &fit.inliers) {
        fit.pose = Some(pose);
        fit.positive_depth_points = count;
        fit.failure = None;
    }
    Ok(fit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::{rotation_degrees, translation_degrees};
    fn fixture(noise: bool) -> (Vec<Match>, RelativePose) {
        let a = 0.12_f64;
        let r = Matrix3::new(a.cos(), 0., a.sin(), 0., 1., 0., -a.sin(), 0., a.cos());
        let t = Vector3::new(0.6, 0.1, 0.15);
        let mut points = Vec::new();
        for i in 0..96 {
            let x = Vector3::new(
                ((i * 31 % 97) as f64 / 48. - 1.) * 1.8,
                ((i * 43 % 89) as f64 / 44. - 1.) * 1.1,
                3. + (i * 17 % 73) as f64 / 25.,
            );
            let y = r * x + t;
            let mut p = [[x.x / x.z, x.y / x.z], [y.x / y.z, y.y / y.z]];
            if noise {
                if i % 5 == 0 {
                    p[1] = [
                        ((i * 7 % 31) as f64 / 30. - 0.5) * 0.8,
                        ((i * 13 % 37) as f64 / 36. - 0.5) * 0.7,
                    ];
                } else {
                    p[1][0] += (i as f64 * 1.7).sin() * 0.0001;
                    p[1][1] += (i as f64 * 2.3).cos() * 0.0001;
                }
            }
            points.push(p);
        }
        (
            points,
            RelativePose {
                rotation: std::array::from_fn(|i| std::array::from_fn(|j| r[(i, j)])),
                translation: [t.x, t.y, t.z],
            },
        )
    }
    fn config() -> SolverConfig {
        SolverConfig {
            max_trials: 2048,
            min_trials: 64,
            confidence: 0.999,
            threshold: 0.001,
            min_inliers: 12,
            seed: 781,
        }
    }
    #[test]
    fn exact_and_noisy_nonplanar_geometry_recovers_signed_pose() {
        for noise in [false, true] {
            let (points, truth) = fixture(noise);
            let fit = estimate(&points, &config()).unwrap();
            let p = fit.pose.as_ref().expect("pose recovered");
            assert!(
                rotation_degrees(p.rotation, truth.rotation).unwrap() < 0.15,
                "{fit:?}"
            );
            assert!(
                translation_degrees(p.translation, truth.translation)
                    .unwrap()
                    .unwrap()
                    < 0.8,
                "{fit:?}"
            );
            assert!(fit.positive_depth_points >= 70);
            assert_eq!(
                serde_json::to_value(&fit).unwrap(),
                serde_json::to_value(estimate(&points, &config()).unwrap()).unwrap()
            );
        }
    }
    #[test]
    fn five_point_recovers_pose_deterministically_with_outliers() {
        for noise in [false, true] {
            let (points, truth) = fixture(noise);
            let fit = estimate_with(&points, &config(), MinimalSolver::FivePoint).unwrap();
            let pose = fit.pose.as_ref().expect("five-point pose recovered");
            assert!(
                rotation_degrees(pose.rotation, truth.rotation).unwrap() < 0.15,
                "{fit:?}"
            );
            assert!(
                translation_degrees(pose.translation, truth.translation)
                    .unwrap()
                    .unwrap()
                    < 0.8,
                "{fit:?}"
            );
            assert!(fit.positive_depth_points >= 70);
            assert_eq!(
                serde_json::to_value(&fit).unwrap(),
                serde_json::to_value(
                    estimate_with(&points, &config(), MinimalSolver::FivePoint).unwrap()
                )
                .unwrap()
            );
        }
        let constant = vec![[[0.1, 0.2], [0.3, 0.4]]; 32];
        assert!(
            estimate_with(&constant, &config(), MinimalSolver::FivePoint)
                .unwrap()
                .pose
                .is_none()
        );
    }
    #[test]
    fn insufficient_degenerate_and_nonfinite_inputs_are_explicit() {
        assert!(estimate(&[], &config()).unwrap().pose.is_none());
        let constant = vec![[[0.1, 0.2], [0.3, 0.4]]; 32];
        assert!(estimate(&constant, &config()).unwrap().pose.is_none());
        assert!(estimate(&[[[f64::NAN, 0.], [0., 0.]]], &config()).is_err());
        let points = (0..32)
            .map(|i| [[i as f64 / 100., 0.], [i as f64 / 100. + 0.1, 0.]])
            .collect::<Vec<_>>();
        assert!(estimate(&points, &config()).unwrap().pose.is_none());
    }
}
