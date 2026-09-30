//! RGB-only dense auxiliary objectives for transferable fusion. Renderer
//! geometry is neither an input nor a target. Teacher affinities are semantic
//! soft targets, not ground-truth correspondence or co-visibility labels.
use anyhow::{Result, ensure};
use burn::tensor::{Tensor, activation, backend::Backend};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FusionAuxiliary {
    pub attention_weight: f64,
    pub dense_weight: f64,
    pub descriptor_weight: f64,
    pub bidirectional: bool,
    pub teacher_temperature: f64,
    /// Separate descriptor temperature permits registered affinity sharpening.
    /// None retains the historical teacher/student temperature equality.
    pub descriptor_temperature: Option<f64>,
    /// Freeze the audited warm-start student's encoder as an auxiliary teacher.
    /// This preserves its gained matching skill; the primary teacher stays fixed.
    pub anchor_warm_start: bool,
    /// One-based trained hierarchical blocks for affinity targets only.
    /// Empty keeps final-layer affinities; dense latent targets stay final-layer.
    pub teacher_layers: Vec<usize>,
}
impl Default for FusionAuxiliary {
    fn default() -> Self {
        Self {
            attention_weight: 0.,
            dense_weight: 0.,
            descriptor_weight: 0.,
            bidirectional: false,
            teacher_temperature: 0.07,
            descriptor_temperature: None,
            anchor_warm_start: false,
            teacher_layers: Vec::new(),
        }
    }
}
impl FusionAuxiliary {
    pub fn enabled(&self) -> bool {
        self.attention_weight > 0. || self.dense_weight > 0. || self.descriptor_weight > 0.
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            (0. ..=1.).contains(&self.attention_weight) && (0. ..=1.).contains(&self.dense_weight),
            "invalid fusion auxiliary weights"
        );
        ensure!(
            (0. ..=4.).contains(&self.descriptor_weight),
            "invalid descriptor weight"
        );
        ensure!(
            self.descriptor_weight == 0. || self.bidirectional,
            "descriptor alignment requires both pair directions"
        );
        ensure!(
            (0.01..=1.).contains(&self.teacher_temperature),
            "invalid affinity temperature"
        );
        ensure!(
            self.descriptor_temperature
                .is_none_or(|t| (0.01..=1.).contains(&t)),
            "invalid descriptor temperature"
        );
        ensure!(
            self.teacher_layers.iter().all(|&x| x > 0)
                && self.teacher_layers.windows(2).all(|x| x[0] < x[1]),
            "affinity teacher layers must be unique, increasing one-based blocks"
        );
        Ok(())
    }
}

/// Each view's shared feature offset is removed before cosine similarity.
/// Detach precedes every operation so teacher values never build a gradient path.
pub fn teacher_affinity<B: Backend>(
    a: Tensor<B, 3>,
    b: Tensor<B, 3>,
    temperature: f64,
) -> Tensor<B, 3> {
    activation::softmax(centered_cosine(a, b) / temperature, 2)
}

fn centered_cosine<B: Backend>(a: Tensor<B, 3>, b: Tensor<B, 3>) -> Tensor<B, 3> {
    let unit = |x: Tensor<B, 3>| {
        let x = x.detach();
        let x = x.clone() - x.mean_dim(1);
        x.clone() / x.powf_scalar(2.).sum_dim(2).sqrt().clamp_min(1e-6)
    };
    unit(a).matmul(unit(b).swap_dims(1, 2))
}

/// Uniform mean of unit-cosine score matrices, before temperature and softmax.
/// This is the same feature-level operator used by the read-only layer audit.
pub fn hierarchical_affinity<B: Backend>(
    pairs: Vec<(Tensor<B, 3>, Tensor<B, 3>)>,
    temperature: f64,
) -> Tensor<B, 3> {
    assert!(!pairs.is_empty());
    let count = pairs.len();
    let scores = pairs
        .into_iter()
        .map(|(a, b)| centered_cosine(a, b))
        .reduce(|a, b| a + b)
        .unwrap();
    activation::softmax(scores / count as f64 / temperature, 2)
}

/// Mean row KL, averaged across decoder layers. Heads retain their own value
/// routing; only their mean scores are guided toward the semantic teacher.
pub fn attention_kl<B: Backend>(logits: Vec<Tensor<B, 3>>, target: Tensor<B, 3>) -> Tensor<B, 1> {
    assert!(!logits.is_empty());
    let target = target.detach();
    let log_target = target.clone().clamp_min(1e-12).log();
    let losses = logits
        .into_iter()
        .map(|x| {
            (target.clone() * (log_target.clone() - activation::log_softmax(x, 2)))
                .sum_dim(2)
                .mean()
        })
        .collect();
    Tensor::cat(losses, 0).mean()
}

/// Guide the decoder's own cosine geometry, independently of its prediction
/// head's coordinate system. Both descriptor branches receive gradients.
pub fn descriptor_kl<B: Backend>(
    a: Tensor<B, 3>,
    b: Tensor<B, 3>,
    forward_target: Tensor<B, 3>,
    backward_target: Tensor<B, 3>,
    temperature: f64,
) -> Tensor<B, 1> {
    let unit = |x: Tensor<B, 3>| x.clone() / (x.powf_scalar(2.).sum_dim(2) + 1e-6).sqrt();
    let scores = unit(a).matmul(unit(b).swap_dims(1, 2)) / temperature;
    (attention_kl(vec![scores.clone()], forward_target)
        + attention_kl(vec![scores.swap_dims(1, 2)], backward_target))
        * 0.5
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::TensorData,
    };
    type B = Autodiff<NdArray<f32>>;
    #[test]
    fn descriptor_weight_range_supports_registered_preservation_screen() {
        for weight in [0., 1., 4.] {
            assert!(
                FusionAuxiliary {
                    descriptor_weight: weight,
                    bidirectional: true,
                    ..Default::default()
                }
                .validate()
                .is_ok()
            );
        }
        for weight in [-0.01, 4.01, f64::NAN, f64::INFINITY] {
            assert!(
                FusionAuxiliary {
                    descriptor_weight: weight,
                    bidirectional: true,
                    ..Default::default()
                }
                .validate()
                .is_err()
            );
        }
    }
    #[test]
    fn hierarchical_targets_average_scores_and_stop_every_teacher_gradient() {
        let d = Default::default();
        let a = Tensor::<B, 3>::from_data(
            TensorData::new(vec![1., 0., 0., 1., -1., 0., 0., -1.], [1, 4, 2]),
            &d,
        )
        .require_grad();
        let b = Tensor::<B, 3>::from_data(
            TensorData::new(vec![0., 1., -1., 0., 0., -1., 1., 0.], [1, 4, 2]),
            &d,
        )
        .require_grad();
        let single = hierarchical_affinity(vec![(a.clone(), b.clone())], 0.07);
        let plain = teacher_affinity(a.clone(), b.clone(), 0.07);
        assert_eq!(
            crate::tensor::values(single).unwrap(),
            crate::tensor::values(plain).unwrap()
        );
        let target = hierarchical_affinity(
            vec![
                (a.clone(), a.clone()),
                (a.clone() * 3. + 9., b.clone() * 2. - 7.),
            ],
            0.2,
        );
        let expected = activation::softmax(
            (centered_cosine(a.clone(), a.clone()) + centered_cosine(a.clone(), b.clone()))
                / 2.
                / 0.2,
            2,
        );
        assert!(crate::tensor::scalar((target.clone() - expected).abs().max()).unwrap() < 1e-6);
        let scores = Tensor::<B, 3>::zeros([1, 4, 4], &d).require_grad();
        let grads = attention_kl(vec![scores.clone()], target).backward();
        assert!(a.grad(&grads).is_none() && b.grad(&grads).is_none());
        assert!(crate::tensor::scalar(scores.grad(&grads).unwrap().abs().sum()).unwrap() > 0.1);
    }
    #[test]
    fn descriptor_alignment_trains_both_branches_without_teacher_gradients() {
        let d = Default::default();
        let values = vec![1., 0., 0., 1., -1., 0., 0., -1.];
        let teacher = Tensor::<B, 3>::from_data(TensorData::new(values.clone(), [1, 4, 2]), &d)
            .require_grad();
        let target = teacher_affinity(teacher.clone(), teacher.clone(), 0.2);
        let a = Tensor::<B, 3>::from_data(TensorData::new(values, [1, 4, 2]), &d).require_grad();
        let exact = descriptor_kl(a.clone(), a.clone(), target.clone(), target.clone(), 0.2);
        assert!(crate::tensor::scalar(exact).unwrap().abs() < 1e-5);
        let b = Tensor::<B, 3>::from_data(
            TensorData::new(vec![0.3, 0.9, 1., 0.2, -0.5, 0.3, 0.4, -1.], [1, 4, 2]),
            &d,
        )
        .require_grad();
        let loss = descriptor_kl(a.clone(), b.clone(), target.clone(), target, 0.2);
        let grads = loss.backward();
        assert!(teacher.grad(&grads).is_none());
        for x in [a, b] {
            assert!(crate::tensor::scalar(x.grad(&grads).unwrap().abs().sum()).unwrap() > 1e-3);
        }
    }
    #[test]
    fn semantic_guidance_is_shift_invariant_detached_and_trains_scores() {
        let d = Default::default();
        let a = Tensor::<B, 3>::from_data(
            TensorData::new(vec![1., 0., 0., 1., -1., 0., 0., -1.], [1, 4, 2]),
            &d,
        )
        .require_grad();
        let b = a.clone();
        let target = teacher_affinity(a.clone(), b.clone(), 0.07);
        let shifted = teacher_affinity(a.clone() + 13., b + 3., 0.07);
        assert!(crate::tensor::scalar((target.clone() - shifted).abs().max()).unwrap() < 1e-5);
        let scores = Tensor::<B, 3>::zeros([1, 4, 4], &d).require_grad();
        let loss = attention_kl(vec![scores.clone()], target.clone());
        let grads = loss.backward();
        assert!(a.grad(&grads).is_none());
        assert!(crate::tensor::scalar(scores.grad(&grads).unwrap().abs().sum()).unwrap() > 0.1);
        let exact = attention_kl(vec![target.clone().clamp_min(1e-12).log()], target);
        assert!(crate::tensor::scalar(exact).unwrap().abs() < 1e-6);
    }
}
