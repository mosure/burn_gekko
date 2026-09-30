//! The primary completion path never receives the augmented dense target.
use super::*;
use crate::data::augmentation::{EquivarianceConfig, WarpBatch};
use burn_gekko::objectives::correspondence::bidirectional_nll;

pub(super) fn losses<B: AutodiffBackend>(
    model: &LatentModel<B>,
    original: Tensor<B, 3>,
    batch: WarpBatch<B::InnerBackend>,
    c: &EquivarianceConfig,
    grid: [usize; 2],
) -> Result<(Tensor<B, 1>, Tensor<B, 1>)> {
    let augmented = model.encode(Tensor::from_inner(batch.rgb), None);
    let descriptor = |a: Tensor<B, 3>, context: Tensor<B, 3>| -> Result<Tensor<B, 3>> {
        let features = model
            .fusion
            .decoder
            .pair_features(a.clone(), context, grid)?;
        model
            .spatial_descriptor(a, features)
            .ok_or_else(|| anyhow::anyhow!("missing spatial descriptor"))
    };
    let forward = Tensor::from_inner(batch.forward);
    let backward = Tensor::from_inner(batch.backward);
    let pair = bidirectional_nll(
        descriptor(original.clone(), augmented.clone())?,
        descriptor(augmented.clone(), original.clone())?,
        forward.clone(),
        backward.clone(),
        c.temperature,
    );
    let independent = if c.self_weight > 0. {
        bidirectional_nll(
            descriptor(original.clone(), original)?,
            descriptor(augmented.clone(), augmented)?,
            forward,
            backward,
            c.temperature,
        )
    } else {
        Tensor::zeros([1], &pair.device())
    };
    Ok((pair, independent))
}
