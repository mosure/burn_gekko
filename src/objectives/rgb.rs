use crate::model::Predictions;
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use burn_vjepa::SparseTokenMask;

/// Nonoverlapping RGB patches, pixel-major RGB order; normalization is per RGB patch.
pub fn rgb_patches<B: Backend>(rgb: Tensor<B, 4>, patch: usize, normalize: bool) -> Tensor<B, 3> {
    let [b, c, h, w] = rgb.dims();
    assert_eq!(c, 3);
    let x = rgb
        .reshape([b, 3, h / patch, patch, w / patch, patch])
        .permute([0, 2, 4, 3, 5, 1])
        .reshape([b, (h / patch) * (w / patch), patch * patch * 3]);
    if !normalize {
        return x;
    }
    normalize_patches(x)
}

pub fn normalize_patches<B: Backend>(x: Tensor<B, 3>) -> Tensor<B, 3> {
    let d = x.dims()[2];
    let centered = x.clone() - x.mean_dim(2);
    // PyTorch's default unbiased patch variance, matching the Gekko/MAE target convention.
    let variance = centered.clone().powf_scalar(2.0).sum_dim(2) / (d - 1) as f32;
    centered / (variance + 1e-6).sqrt()
}

/// Optional last two values are predicted patch mean and log standard deviation.
pub fn content_prediction<B: Backend>(prediction: Tensor<B, 3>, d: usize) -> Tensor<B, 3> {
    if prediction.dims()[2] == d {
        prediction
    } else {
        prediction.slice_dim(2, 0..d)
    }
}
pub fn calibrated_rgb<B: Backend>(prediction: Tensor<B, 3>, d: usize) -> Tensor<B, 3> {
    assert_eq!(prediction.dims()[2], d + 2);
    let mean = prediction.clone().slice_dim(2, d..d + 1);
    let std = prediction
        .clone()
        .slice_dim(2, d + 1..d + 2)
        .clamp(-10.0, 1.0)
        .exp();
    prediction.slice_dim(2, 0..d) * std + mean
}

/// Preserve the normalized Gekko/RI objective while calibrating a standalone RGB output.
/// Statistics are supervised from training RGB only; inference uses predicted statistics.
pub fn reconstruction_loss<B: Backend>(
    out: Predictions<B>,
    raw: Tensor<B, 3>,
    visible: &SparseTokenMask,
    normalize: bool,
    predict_stats: bool,
) -> Losses<B> {
    let [_, n, d] = raw.dims();
    assert!(!predict_stats || normalize);
    let auxiliary = if predict_stats {
        let mean = raw.clone().mean_dim(2);
        let log_std = (((raw.clone() - mean.clone()).powf_scalar(2.0).sum_dim(2) / (d - 1) as f32)
            + 1e-6)
            .sqrt()
            .log();
        let hidden: Vec<i64> = (0..n)
            .filter(|i| visible.indices().binary_search(i).is_err())
            .map(|i| i as i64)
            .collect();
        let k = hidden.len();
        let indices = Tensor::<B, 1, Int>::from_data(TensorData::new(hidden, [k]), &raw.device());
        let losses: Vec<_> = [&out.cross_rgb, &out.mae_rgb]
            .into_iter()
            .map(|prediction| {
                let mean_error = (prediction.clone().slice_dim(2, d..d + 1) - mean.clone())
                    .powf_scalar(2.0)
                    * 16.0;
                let scale_error = (prediction.clone().slice_dim(2, d + 1..d + 2) - log_std.clone())
                    .powf_scalar(2.0)
                    * 0.1;
                let rgb_error = (calibrated_rgb(prediction.clone(), d) - raw.clone())
                    .powf_scalar(2.0)
                    .mean_dim(2);
                (mean_error + scale_error + rgb_error)
                    .select(1, indices.clone())
                    .mean()
            })
            .collect();
        Some(Tensor::cat(losses, 0).sum())
    } else {
        None
    };
    let target = if normalize {
        normalize_patches(raw)
    } else {
        raw
    };
    let mut losses = gekko_loss(
        Predictions {
            cross_rgb: content_prediction(out.cross_rgb, d),
            mae_rgb: content_prediction(out.mae_rgb, d),
            ri: out.ri,
        },
        target,
        visible,
    );
    if let Some(auxiliary) = auxiliary {
        losses.total = losses.total + auxiliary;
    }
    losses
}

pub struct Losses<B: Backend> {
    pub total: Tensor<B, 1>,
    pub cross: Tensor<B, 1>,
    pub mae: Tensor<B, 1>,
    pub ri: Tensor<B, 1>,
}

/// Released-code RI objective (epsilon 1e-2). RI targets and weighting never backpropagate into RGB heads.
pub fn gekko_loss<B: Backend>(
    out: Predictions<B>,
    target: Tensor<B, 3>,
    visible: &SparseTokenMask,
) -> Losses<B> {
    let [b, n, d] = target.dims();
    let pixels = d / 3;
    let device = target.device();
    let per_pixel = |x: Tensor<B, 3>| {
        (x - target.clone())
            .reshape([b, n, pixels, 3])
            .powf_scalar(2.0)
            .mean_dim(3)
            .reshape([b, n, pixels])
    };
    let mae = per_pixel(out.mae_rgb);
    let cross = per_pixel(out.cross_rgb);
    let ri = ((mae.clone() - cross.clone()).detach()
        - mae.clone().detach().clamp_min(1e-2) * out.ri)
        .powf_scalar(2.0);
    let hidden: Vec<i64> = (0..n)
        .filter(|i| visible.indices().binary_search(i).is_err())
        .map(|i| i as i64)
        .collect();
    let k = hidden.len();
    assert!(k > 0);
    let indices = Tensor::<B, 1, Int>::from_data(TensorData::new(hidden, [k]), &device);
    let mae = mae.select(1, indices.clone()).mean();
    let cross = cross.select(1, indices.clone()).mean();
    let ri = ri.select(1, indices).mean();
    Losses {
        total: mae.clone() + cross.clone() + ri.clone(),
        cross,
        mae,
        ri,
    }
}
