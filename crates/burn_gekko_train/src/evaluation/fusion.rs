//! Causal readouts from a fixed fusion checkpoint; RGB-derived features only.
use crate::{
    correspondence::{centered_matches, matches, nearest},
    latent_eval::values,
    model::{AttentionTrace, GekkoDecoder},
};
use anyhow::{Result, ensure};
use burn::tensor::{Tensor, backend::Backend};

pub type NamedMatches = (String, (Vec<usize>, Vec<bool>));

fn reciprocal<B: Backend>(
    a: Tensor<B, 3>,
    b: Tensor<B, 3>,
    probability: bool,
) -> Result<(Vec<usize>, Vec<bool>)> {
    let [batch, n, m] = a.dims();
    ensure!(
        batch == 1 && n == m && b.dims() == [1, n, n],
        "invalid audit score shape"
    );
    let b = b.swap_dims(1, 2);
    let score = if probability {
        (a * b).sqrt()
    } else {
        (a + b) / 2.
    };
    let x = values(score)?;
    ensure!(x.iter().all(|v| v.is_finite()), "nonfinite audit scores");
    Ok(nearest(&x, n))
}

/// Returns the entire predeclared family, without selecting against geometry.
pub fn readouts<B: Backend>(
    decoder: &GekkoDecoder<B>,
    a: Tensor<B, 3>,
    b: Tensor<B, 3>,
    grid: [usize; 2],
) -> Result<Vec<NamedMatches>> {
    let mut result = Vec::new();
    for cross_rope in [true, false] {
        let prefix = if cross_rope { "trace" } else { "no_cross_rope" };
        let forward = decoder.pair_trace(a.clone(), b.clone(), grid, cross_rope)?;
        let backward = decoder.pair_trace(b.clone(), a.clone(), grid, cross_rope)?;
        result.push((
            format!("{prefix}_decoder"),
            matches(forward.features.clone(), backward.features.clone())?,
        ));
        result.push((
            format!("{prefix}_centered_decoder"),
            centered_matches(forward.features, backward.features)?,
        ));
        for kind in ["probability", "logits", "content", "centered_content"] {
            let select = |x: &AttentionTrace<B>| match kind {
                "probability" => x.probability.clone(),
                "logits" => x.logits.clone(),
                "content" => x.content.clone(),
                _ => x.centered_content.clone(),
            };
            let mut f = Vec::new();
            let mut r = Vec::new();
            for (i, (first, second)) in forward.layers.iter().zip(&backward.layers).enumerate() {
                let (first, second) = (select(first), select(second));
                result.push((
                    format!("{prefix}_{kind}_layer{}", i + 1),
                    reciprocal(first.clone(), second.clone(), kind == "probability")?,
                ));
                f.push(first);
                r.push(second);
            }
            let mean = |x: Vec<Tensor<B, 3>>| Tensor::cat(x, 0).mean_dim(0);
            result.push((
                format!("{prefix}_{kind}_mean"),
                reciprocal(mean(f), mean(r), kind == "probability")?,
            ));
        }
    }
    Ok(result)
}
