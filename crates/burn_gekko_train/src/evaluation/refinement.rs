//! Fixed local probability readout, shared by known-warp and real-image exports.
use crate::{correspondence, latent::LatentModel, latent_eval::values};
use anyhow::{Result, ensure};
use burn::tensor::{Tensor, backend::Backend};
use serde::Serialize;

#[derive(Clone, Serialize)]
pub struct Readout {
    pub method: String,
    pub indices: Vec<usize>,
    pub mutual: Vec<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coordinates: Option<Vec<[f64; 2]>>,
}

/// Both operators share a single device readback and identical coarse matches.
pub fn descriptor_readouts<B: Backend>(
    name: &str,
    a: Tensor<B, 3>,
    b: Tensor<B, 3>,
    grid: [usize; 2],
    temperature: f64,
) -> Result<[Readout; 2]> {
    ensure!(
        temperature.is_finite() && temperature > 0.,
        "invalid readout temperature"
    );
    let n = grid[0] * grid[1];
    ensure!(
        a.dims()[0] == 1 && a.dims()[1] == n && b.dims() == a.dims(),
        "invalid descriptor grid"
    );
    let scores = correspondence::cosine_scores(a, b) / temperature;
    let scores =
        correspondence::reciprocal_conditionals(vec![scores.clone()], vec![scores.swap_dims(1, 2)]);
    let scores = values(scores)?;
    score_readouts(name, &scores, grid)
}

/// Canonical hard and local readouts from one finite log-probability matrix.
pub(crate) fn score_readouts(name: &str, scores: &[f32], grid: [usize; 2]) -> Result<[Readout; 2]> {
    let n = grid[0] * grid[1];
    let (indices, mutual) = correspondence::nearest_checked(scores, n)?;
    // The shared matcher returns log probabilities. Subtract each row maximum
    // before exponentiation; this scale cancels in the centroid and avoids underflow.
    let probabilities = scores
        .chunks_exact(n)
        .flat_map(|row| {
            let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            row.iter().map(move |s| (*s - max).exp())
        })
        .collect::<Vec<_>>();
    let coordinates =
        burn_gekko_eval::refinement::local_coordinates(&probabilities, &indices, grid)?;
    let hard = Readout {
        method: name.into(),
        indices,
        mutual,
        coordinates: None,
    };
    let fine = Readout {
        method: format!("{name}_local"),
        coordinates: Some(coordinates),
        ..hard.clone()
    };
    Ok([hard, fine])
}

/// Pair, trained same-image, and encoder controls all receive the same operator.
/// Optional same-image descriptors are cached once per image by HPatches.
pub fn spatial_readouts<B: Backend>(
    model: &LatentModel<B>,
    a: Tensor<B, 3>,
    b: Tensor<B, 3>,
    grid: [usize; 2],
    layer: usize,
    independent: Option<(Tensor<B, 3>, Tensor<B, 3>)>,
) -> Result<Vec<Readout>> {
    let width = model.encoder_config.encoder.embed_dim;
    ensure!(
        a.dims()[2] == 2 * width && b.dims()[2] == 2 * width,
        "missing spatial route"
    );
    let pair = |x: Tensor<B, 3>, context: Tensor<B, 3>| -> Result<Tensor<B, 3>> {
        model
            .spatial_descriptor(
                x.clone(),
                model.fusion.decoder.pair_features(x, context, grid)?,
            )
            .ok_or_else(|| anyhow::anyhow!("missing spatial head"))
    };
    let (sa, sb) = match independent {
        Some(x) => x,
        None => (
            correspondence::self_conditioned_descriptor(model, a.clone(), grid)?,
            correspondence::self_conditioned_descriptor(model, b.clone(), grid)?,
        ),
    };
    let center = |x: Tensor<B, 3>| {
        let x = x.slice_dim(2, width..2 * width);
        x.clone() - x.mean_dim(1)
    };
    let mut readouts = Vec::new();
    readouts.extend(descriptor_readouts(
        "spatial_residual_conditional",
        pair(a.clone(), b.clone())?,
        pair(b.clone(), a.clone())?,
        grid,
        0.07,
    )?);
    readouts.extend(descriptor_readouts(
        "spatial_self_conditional",
        sa,
        sb,
        grid,
        0.07,
    )?);
    readouts.extend(descriptor_readouts(
        &format!("student_l{layer:02}_centered_conditional"),
        center(a),
        center(b),
        grid,
        0.07,
    )?);
    Ok(readouts)
}

#[cfg(all(test, feature = "ndarray"))]
mod tests {
    use super::*;
    use burn::{backend::NdArray, tensor::TensorData};
    #[test]
    fn refined_export_preserves_legacy_hard_indices_and_mutual_flags() {
        type B = NdArray<f32>;
        let device = Default::default();
        let a = Tensor::<B, 3>::from_data(
            TensorData::new(vec![1., 0., 0.8, 0.2, 0., 1., -1., 0.], [1, 4, 2]),
            &device,
        );
        let b = a.clone().flip([1]);
        let old = correspondence::conditional_matches(a.clone(), b.clone(), 0.07).unwrap();
        let [hard, fine] = descriptor_readouts("fixture", a, b, [2, 2], 0.07).unwrap();
        assert_eq!(hard.indices, old.0);
        assert_eq!(hard.mutual, old.1);
        assert_eq!(fine.indices, hard.indices);
        assert_eq!(fine.mutual, hard.mutual);
        assert_eq!(fine.coordinates.unwrap().len(), 4);
        assert!(hard.coordinates.is_none());
    }
}
