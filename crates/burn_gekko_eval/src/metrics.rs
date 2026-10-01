//! Reconstruction metrics with explicit masks, units and undefined cases.
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

pub mod detail;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompletionMetrics {
    pub tokens: usize,
    pub channels: usize,
    pub mse: f64,
    pub cosine: Option<f64>,
    pub cosine_defined_tokens: usize,
    pub spatial_variance_ratio: Option<f64>,
    /// Mean squared teacher amplitude over the declared mask (not a peak range).
    #[serde(default)]
    pub signal_power: f64,
    /// 10 log10(signal power / MSE); undefined at zero signal or zero error.
    #[serde(default)]
    pub signal_to_error_db: Option<f64>,
}

/// Equal-token weighting. A zero-norm token has undefined cosine, never perfect cosine.
pub fn completion(
    pred: &[f32],
    truth: &[f32],
    channels: usize,
    selected: &[usize],
) -> Result<CompletionMetrics> {
    ensure!(
        channels > 0
            && !pred.is_empty()
            && pred.len() == truth.len()
            && pred.len().is_multiple_of(channels),
        "invalid latent shape"
    );
    ensure!(
        !selected.is_empty() && selected.iter().all(|i| *i < pred.len() / channels),
        "invalid evaluation mask"
    );
    let mut ids = selected.to_vec();
    ids.sort_unstable();
    ids.dedup();
    ensure!(ids.len() == selected.len(), "duplicate evaluation token");
    ensure!(
        pred.iter().chain(truth).all(|v| v.is_finite()),
        "nonfinite latent"
    );
    let (mut mse, mut cosine, mut defined, mut power) = (0., 0., 0, 0.);
    for &i in selected {
        let (mut dot, mut a, mut b) = (0., 0., 0.);
        for c in 0..channels {
            let p = pred[i * channels + c] as f64;
            let t = truth[i * channels + c] as f64;
            mse += (p - t).powi(2);
            dot += p * t;
            a += p * p;
            b += t * t;
            power += t * t;
        }
        if a > 1e-24 && b > 1e-24 {
            cosine += dot / (a * b).sqrt();
            defined += 1;
        }
    }
    let variance = |v: &[f32]| -> f64 {
        (0..channels)
            .map(|c| {
                let avg = selected
                    .iter()
                    .map(|&i| v[i * channels + c] as f64)
                    .sum::<f64>()
                    / selected.len() as f64;
                selected
                    .iter()
                    .map(|&i| (v[i * channels + c] as f64 - avg).powi(2))
                    .sum::<f64>()
                    / selected.len() as f64
            })
            .sum::<f64>()
            / channels as f64
    };
    let tv = variance(truth);
    let denominator = (selected.len() * channels) as f64;
    let mse = mse / denominator;
    let signal_power = power / denominator;
    Ok(CompletionMetrics {
        tokens: selected.len(),
        channels,
        mse,
        signal_power,
        signal_to_error_db: signal_to_error_db(signal_power, mse)?,
        cosine: (defined > 0).then(|| cosine / defined as f64),
        cosine_defined_tokens: defined,
        spatial_variance_ratio: (tv > 1e-24).then(|| variance(pred) / tv),
    })
}

/// A signal-normalized error in dB, deliberately distinct from peak-based PSNR.
pub fn signal_to_error_db(signal_power: f64, mse: f64) -> Result<Option<f64>> {
    ensure!(
        signal_power.is_finite() && signal_power >= 0. && mse.is_finite() && mse >= 0.,
        "invalid signal/error power"
    );
    Ok((signal_power > 0. && mse > 0.).then(|| 10. * (signal_power / mse).log10()))
}

/// PSNR for declared data range; exact reconstruction is represented as None (+infinity).
pub fn psnr(mse: f64, data_range: f64) -> Result<Option<f64>> {
    ensure!(
        mse.is_finite() && mse >= 0. && data_range.is_finite() && data_range > 0.,
        "invalid PSNR input"
    );
    Ok((mse > 0.).then(|| 10. * (data_range * data_range / mse).log10()))
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CorrespondenceMetrics {
    pub points: usize,
    pub aepe: f64,
    pub pck1: f64,
    pub pck3: f64,
    pub pck5: f64,
    pub correct1: usize,
    pub correct3: usize,
    pub correct5: usize,
}
pub fn correspondence(errors: &[f64]) -> Result<CorrespondenceMetrics> {
    ensure!(
        !errors.is_empty() && errors.iter().all(|e| e.is_finite() && *e >= 0.),
        "invalid correspondence errors"
    );
    let n = errors.len();
    let correct = |t| errors.iter().filter(|&&e| e <= t).count();
    let (a, b, c) = (correct(1.), correct(3.), correct(5.));
    Ok(CorrespondenceMetrics {
        points: n,
        aepe: errors.iter().sum::<f64>() / n as f64,
        pck1: a as f64 / n as f64,
        pck3: b as f64 / n as f64,
        pck5: c as f64 / n as f64,
        correct1: a,
        correct3: b,
        correct5: c,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mask_zero_vectors_and_thresholds_are_explicit() {
        let r = completion(&[1., 0., 100., 100.], &[1., 0., 0., 0.], 2, &[0]).unwrap();
        assert_eq!(r.mse, 0.);
        assert_eq!(r.cosine, Some(1.));
        assert_eq!(r.spatial_variance_ratio, None);
        assert_eq!(
            completion(&[0., 0.], &[0., 0.], 2, &[0]).unwrap().cosine,
            None
        );
        assert!(completion(&[1., 0.], &[1., 0.], 2, &[0, 0]).is_err());
        let r = correspondence(&[0., 1., 3., 5., 6.]).unwrap();
        assert_eq!((r.correct1, r.correct3, r.correct5), (2, 3, 4));
        assert_eq!(psnr(0., 1.).unwrap(), None);
        assert!((psnr(0.01, 1.).unwrap().unwrap() - 20.).abs() < 1e-10);
    }

    #[test]
    fn signal_ratio_uses_masked_teacher_power_and_is_scale_invariant() {
        let r = completion(&[0.9, 1.8, 99., 99.], &[1., 2., 8., 8.], 2, &[0]).unwrap();
        assert_eq!(r.signal_power, 2.5);
        assert!((r.signal_to_error_db.unwrap() - 20.).abs() < 1e-5);
        let scaled = completion(&[9., 18.], &[10., 20.], 2, &[0]).unwrap();
        assert!((r.signal_to_error_db.unwrap() - scaled.signal_to_error_db.unwrap()).abs() < 1e-5);
        assert_eq!(signal_to_error_db(0., 1.).unwrap(), None);
        assert_eq!(signal_to_error_db(1., 0.).unwrap(), None);
        assert!(signal_to_error_db(f64::NAN, 1.).is_err());
    }
}
