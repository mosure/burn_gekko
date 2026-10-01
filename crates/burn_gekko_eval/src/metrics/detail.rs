//! Separate spatial amplitude from structure without treating oracle rescaling as inference.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetailMetrics {
    pub tokens: usize,
    pub channels: usize,
    pub mse: f64,
    /// Error in each channel's mean over the selected positions.
    pub mean_bias_mse: f64,
    pub centered_mse: f64,
    pub teacher_spatial_power: f64,
    pub prediction_spatial_power: f64,
    pub centered_covariance: f64,
    pub spatial_correlation: Option<f64>,
    /// Per-image scalar fit using truth: diagnostic only, never a model output.
    pub oracle_centered_gain: Option<f64>,
    pub oracle_gain_mse: f64,
    pub variance_matching_mse: Option<f64>,
    /// Horizontal/vertical neighbors with both endpoints hidden; no wraparound.
    pub adjacent_pairs: usize,
    pub adjacent_difference_mse: Option<f64>,
    pub adjacent_power_ratio: Option<f64>,
    pub adjacent_correlation: Option<f64>,
}

pub fn measure(
    pred: &[f32],
    truth: &[f32],
    grid: [usize; 2],
    channels: usize,
    selected: &[usize],
) -> Result<DetailMetrics> {
    let [h, w] = grid;
    let tokens = h.checked_mul(w).unwrap_or(0);
    ensure!(
        tokens > 0 && channels > 0 && tokens.checked_mul(channels) == Some(pred.len()),
        "detail grid/shape mismatch"
    );
    let base = super::completion(pred, truth, channels, selected)?;
    let mut pm = vec![0.; channels];
    let mut tm = vec![0.; channels];
    let mut hidden = vec![false; tokens];
    for &i in selected {
        hidden[i] = true;
        for c in 0..channels {
            pm[c] += pred[i * channels + c] as f64 / selected.len() as f64;
            tm[c] += truth[i * channels + c] as f64 / selected.len() as f64;
        }
    }
    let bias = pm
        .iter()
        .zip(&tm)
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>()
        / channels as f64;
    let (mut pp, mut tt, mut pt) = (0., 0., 0.);
    let (mut dp, mut dt, mut dc, mut pairs) = (0., 0., 0., 0usize);
    for &i in selected {
        for c in 0..channels {
            let p = pred[i * channels + c] as f64 - pm[c];
            let t = truth[i * channels + c] as f64 - tm[c];
            pp += p * p;
            tt += t * t;
            pt += p * t;
        }
        for j in [
            (i % w + 1 < w).then_some(i + 1),
            (i / w + 1 < h).then_some(i + w),
        ]
        .into_iter()
        .flatten()
        .filter(|&j| hidden[j])
        {
            pairs += 1;
            for c in 0..channels {
                let p = pred[j * channels + c] as f64 - pred[i * channels + c] as f64;
                let t = truth[j * channels + c] as f64 - truth[i * channels + c] as f64;
                dp += p * p;
                dt += t * t;
                dc += p * t;
            }
        }
    }
    let n = (selected.len() * channels) as f64;
    pp /= n;
    tt /= n;
    pt /= n;
    let centered = (pp + tt - 2. * pt).max(0.);
    ensure!(
        (base.mse - bias - centered).abs() < 1e-10 * (1. + base.mse),
        "MSE decomposition failed"
    );
    let correlation = |p: f64, t: f64, c: f64| {
        (p > 1e-24 && t > 1e-24).then(|| (c / (p * t).sqrt()).clamp(-1., 1.))
    };
    let gain = (pp > 1e-24).then(|| pt / pp);
    Ok(DetailMetrics {
        tokens: selected.len(),
        channels,
        mse: base.mse,
        mean_bias_mse: bias,
        centered_mse: centered,
        teacher_spatial_power: tt,
        prediction_spatial_power: pp,
        centered_covariance: pt,
        spatial_correlation: correlation(pp, tt, pt),
        oracle_centered_gain: gain,
        oracle_gain_mse: bias + (tt - gain.unwrap_or(0.) * pt).max(0.),
        variance_matching_mse: (pp > 1e-24).then(|| {
            let s = (tt / pp).sqrt();
            bias + (s * s * pp + tt - 2. * s * pt).max(0.)
        }),
        adjacent_pairs: pairs,
        adjacent_difference_mse: (pairs > 0)
            .then(|| (dp + dt - 2. * dc).max(0.) / (pairs * channels) as f64),
        adjacent_power_ratio: (dt > 1e-24).then(|| dp / dt),
        adjacent_correlation: correlation(dp, dt, dc),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn amplitude_bias_and_wrong_structure_are_distinct() {
        let t = [-3., -1., 1., 3.];
        let p = [-0.5, 0.5, 1.5, 2.5]; // half contrast, plus one
        let r = measure(&p, &t, [2, 2], 1, &[0, 1, 2, 3]).unwrap();
        assert_eq!(r.mean_bias_mse, 1.);
        assert_eq!(r.centered_mse, 1.25);
        assert_eq!(r.mse, 2.25);
        assert_eq!(r.oracle_centered_gain, Some(2.));
        assert_eq!(r.oracle_gain_mse, 1.);
        assert_eq!(r.variance_matching_mse, Some(1.));
        assert_eq!(r.spatial_correlation, Some(1.));
        assert_eq!(r.adjacent_pairs, 4);
        assert_eq!(r.adjacent_power_ratio, Some(0.25));
        // Orthogonal structure with the same variance cannot be fixed by amplification.
        let wrong = measure(&[-1., 3., -3., 1.], &t, [2, 2], 1, &[0, 1, 2, 3]).unwrap();
        assert_eq!(wrong.spatial_correlation, Some(0.));
        assert_eq!(wrong.oracle_centered_gain, Some(0.));
        assert_eq!(wrong.oracle_gain_mse, 5.);
        assert_eq!(wrong.variance_matching_mse, Some(10.));
    }

    #[test]
    fn mask_excludes_observed_values_and_grid_does_not_wrap() {
        let p = [3., 2., 1., 0.];
        let a = measure(&p, &p, [2, 2], 1, &[1, 2]).unwrap();
        assert_eq!(a.adjacent_pairs, 0);
        assert_eq!(a.adjacent_correlation, None);
        let b = measure(&[99., 2., 1., -99.], &p, [2, 2], 1, &[1, 2]).unwrap();
        assert_eq!(a.mse, b.mse);
        let flat = measure(&[0.; 4], &[0.; 4], [2, 2], 1, &[0, 1]).unwrap();
        assert_eq!(flat.oracle_centered_gain, None);
        assert_eq!(flat.spatial_correlation, None);
        assert_eq!(flat.adjacent_power_ratio, None);
        assert_eq!(flat.oracle_gain_mse, 0.);
        assert!(measure(&p, &p, [3, 2], 1, &[0]).is_err());
        assert!(measure(&p, &p, [2, 2], 1, &[0, 0]).is_err());
        assert!(measure(&[f32::NAN; 4], &p, [2, 2], 1, &[0]).is_err());
    }
}
