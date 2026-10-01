//! Pixel errors with explicit masks, data range and undefined cases.
use anyhow::{Result, ensure};

/// Equal-token/channel MSE with the same accumulation order as offline completion.
pub fn masked_mse(pred: &[f32], truth: &[f32], channels: usize, selected: &[usize]) -> Result<f64> {
    ensure!(
        channels > 0
            && !pred.is_empty()
            && pred.len() == truth.len()
            && pred.len().is_multiple_of(channels),
        "invalid reconstruction shape"
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
        "nonfinite reconstruction"
    );
    let mut error = 0.;
    for &i in selected {
        for c in 0..channels {
            error += (pred[i * channels + c] as f64 - truth[i * channels + c] as f64).powi(2);
        }
    }
    Ok(error / (selected.len() * channels) as f64)
}

/// PSNR for declared data range; exact reconstruction is represented as None (+infinity).
pub fn psnr(mse: f64, data_range: f64) -> Result<Option<f64>> {
    ensure!(
        mse.is_finite() && mse >= 0. && data_range.is_finite() && data_range > 0.,
        "invalid PSNR input"
    );
    Ok((mse > 0.).then(|| 10. * (data_range * data_range / mse).log10()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hidden_pixel_psnr_has_explicit_range_and_mask() {
        let mse = masked_mse(&[9., 0.1, 0.1], &[9., 0., 0.], 1, &[1, 2]).unwrap();
        assert!((psnr(mse, 1.).unwrap().unwrap() - 20.).abs() < 1e-5);
        assert_eq!(psnr(0., 1.).unwrap(), None);
        assert!(masked_mse(&[0.; 3], &[0.; 3], 1, &[1, 1]).is_err());
        assert!(masked_mse(&[0.; 3], &[0.; 3], 1, &[]).is_err());
    }
}
