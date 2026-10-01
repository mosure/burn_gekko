//! Read-only controls for the encoder's trained hierarchical feature levels.
//! No fitting, ground-truth inputs, new weights, or decoder changes are involved.
use crate::{
    correspondence::{ScoredReadouts, cosine_scores, nearest_checked, reciprocal_conditionals},
    encoder::normalize,
    fusion_audit::NamedMatches,
    latent_eval::values,
};
use anyhow::{Result, ensure};
use burn::tensor::{Tensor, backend::Backend};
use burn_vjepa::{VJepaConfig, VJepaEncoder};
use std::collections::BTreeMap;

pub struct EncodedLayer<B: Backend> {
    /// Zero-based encoder block index; exported names use one-based numbering.
    pub index: usize,
    pub views: Vec<Tensor<B, 3>>,
}

pub fn capture_views<B: Backend>(
    encoder: &VJepaEncoder<B>,
    config: &VJepaConfig,
    rgb: &[Tensor<B, 4>],
) -> Result<Vec<EncodedLayer<B>>> {
    capture_view_layers(encoder, config, rgb, &config.encoder.hierarchical_layers())
}

pub fn capture_view_layers<B: Backend>(
    encoder: &VJepaEncoder<B>,
    config: &VJepaConfig,
    rgb: &[Tensor<B, 4>],
    layers: &[usize],
) -> Result<Vec<EncodedLayer<B>>> {
    ensure!(!rgb.is_empty(), "no layer-audit images");
    ensure!(
        !layers.is_empty()
            && layers
                .iter()
                .all(|l| config.encoder.hierarchical_layers().contains(l)),
        "requested encoder output has no trained hierarchical norm"
    );
    let batch = rgb[0].dims()[0];
    ensure!(batch == 1, "matching audit requires one image per view");
    let output = encoder.forward_image_capture_layers(
        normalize(Tensor::cat(rgb.to_vec(), 0), config),
        None,
        layers,
    );
    ensure!(
        output.captured_layers == layers,
        "unexpected captured layers"
    );
    Ok(layers
        .iter()
        .copied()
        .zip(output.hierarchical)
        .map(|(index, tokens)| EncodedLayer {
            index,
            views: (0..rgb.len())
                .map(|i| tokens.clone().slice_dim(0, i..i + 1))
                .collect(),
        })
        .collect())
}

fn score_readouts<B: Backend>(
    prefix: &str,
    scores: Tensor<B, 3>,
) -> Result<ScoredReadouts<Vec<f32>>> {
    let n = scores.dims()[1];
    let conditional = reciprocal_conditionals(
        vec![scores.clone() / 0.07],
        vec![scores.clone().swap_dims(1, 2) / 0.07],
    );
    let raw = nearest_checked(&values(scores)?, n)?;
    let conditional = values(conditional)?;
    Ok(ScoredReadouts {
        readouts: vec![
            (prefix.to_owned(), raw),
            (
                format!("{prefix}_conditional"),
                nearest_checked(&conditional, n)?,
            ),
        ],
        scores: conditional,
    })
}

/// Raw/centered cosine, with/without the already fixed conditional operator.
/// The ensemble is a uniform mean of cosine matrices, not unnormalized features.
pub fn readouts<B: Backend>(
    prefix: &str,
    layers: &[EncodedLayer<B>],
    target: usize,
    reference: usize,
) -> Result<Vec<NamedMatches>> {
    Ok(readouts_with_scores(prefix, layers, target, reference)?.readouts)
}

pub(crate) fn readouts_with_scores<B: Backend>(
    prefix: &str,
    layers: &[EncodedLayer<B>],
    target: usize,
    reference: usize,
) -> Result<ScoredReadouts<BTreeMap<String, Vec<f32>>>> {
    ensure!(!layers.is_empty(), "no captured layers");
    let mut result = Vec::new();
    let mut captured = BTreeMap::new();
    for centered in [false, true] {
        let mode = if centered { "centered" } else { "raw" };
        let mut scores = Vec::new();
        for layer in layers {
            ensure!(
                target < layer.views.len() && reference < layer.views.len(),
                "invalid layer-audit view index"
            );
            let transform = |x: Tensor<B, 3>| {
                if centered {
                    x.clone() - x.mean_dim(1)
                } else {
                    x
                }
            };
            let score = cosine_scores(
                transform(layer.views[target].clone()),
                transform(layer.views[reference].clone()),
            );
            let name = format!("{prefix}_l{:02}_{mode}", layer.index + 1);
            let scored = score_readouts(&name, score.clone())?;
            result.extend(scored.readouts);
            captured.insert(format!("{name}_conditional"), scored.scores);
            scores.push(score);
        }
        if layers.len() > 1 {
            let mean = scores.into_iter().reduce(|a, b| a + b).unwrap() / layers.len() as f64;
            let name = format!("{prefix}_mean_{mode}");
            let scored = score_readouts(&name, mean)?;
            result.extend(scored.readouts);
            captured.insert(format!("{name}_conditional"), scored.scores);
        }
    }
    Ok(ScoredReadouts {
        readouts: result,
        scores: captured,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{backend::NdArray, tensor::TensorData};
    type B = NdArray<f32>;

    #[test]
    fn captures_trained_norms_and_preserves_final_forward_values() {
        let device = Default::default();
        let mut config = VJepaConfig::tiny_for_tests();
        config.encoder.depth = 12;
        config.encoder.n_output_distillation = 4;
        let encoder = VJepaEncoder::<B>::new(&config, &device);
        let rgb = Tensor::from_data(
            TensorData::new(
                (0..3 * 32 * 32)
                    .map(|i| (i as f32 * 0.017).sin() * 0.5 + 0.5)
                    .collect(),
                [1, 3, 32, 32],
            ),
            &device,
        );
        let captured = capture_views(&encoder, &config, &[rgb.clone(), rgb.clone()]).unwrap();
        let single =
            capture_view_layers(&encoder, &config, std::slice::from_ref(&rgb), &[5]).unwrap();
        assert_eq!(single.len(), 1);
        assert_eq!(single[0].index, 5);
        assert_eq!(readouts("student", &single, 0, 0).unwrap().len(), 4);
        assert!(capture_view_layers(&encoder, &config, std::slice::from_ref(&rgb), &[1]).is_err());
        assert_eq!(
            captured.iter().map(|x| x.index).collect::<Vec<_>>(),
            [2, 5, 8, 11]
        );
        let plain = encoder
            .forward_image_capture_layers(
                normalize(Tensor::cat(vec![rgb.clone(), rgb], 0), &config),
                None,
                &[],
            )
            .tokens;
        for (i, view) in captured.last().unwrap().views.iter().enumerate() {
            assert_eq!(
                values(view.clone()).unwrap(),
                values(plain.clone().slice_dim(0, i..i + 1)).unwrap()
            );
        }
    }

    #[test]
    fn all_levels_and_uniform_ensemble_recover_permutation_despite_layer_scale() {
        let device = Default::default();
        let a = Tensor::<B, 3>::from_data(
            TensorData::new(vec![1., 0., 0., 1., -1., 0., 0., -1.], [1, 4, 2]),
            &device,
        );
        let b = Tensor::<B, 3>::from_data(
            TensorData::new(vec![-1., 0., 1., 0., 0., -1., 0., 1.], [1, 4, 2]),
            &device,
        );
        let layers: Vec<_> = [2, 5, 8, 11]
            .into_iter()
            .map(|index| EncodedLayer {
                index,
                views: vec![a.clone() * (index + 1) as f64, b.clone() * 0.5],
            })
            .collect();
        let result = readouts("student", &layers, 0, 1).unwrap();
        assert_eq!(result.len(), 20);
        for (_, (indices, mutual)) in result {
            assert_eq!(indices, [1, 3, 0, 2]);
            assert!(mutual.into_iter().all(|x| x));
        }
    }
}
