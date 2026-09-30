//! Descriptor supervision from known image transforms. Label rows are detached
//! bilinear distributions; cropped queries are zero rows and contribute no loss.
use burn::tensor::{Tensor, activation, backend::Backend};

/// Mean NLL over valid queries in each direction, then mean of the directions.
/// Both descriptors receive gradients; neither label tensor can receive them.
pub fn bidirectional_nll<B: Backend>(
    a: Tensor<B, 3>,
    b: Tensor<B, 3>,
    forward: Tensor<B, 3>,
    backward: Tensor<B, 3>,
    temperature: f64,
) -> Tensor<B, 1> {
    assert!(temperature.is_finite() && temperature > 0.);
    let unit = |x: Tensor<B, 3>| x.clone() / (x.powf_scalar(2.).sum_dim(2) + 1e-6).sqrt();
    let scores = unit(a).matmul(unit(b).swap_dims(1, 2)) / temperature;
    let nll = |scores: Tensor<B, 3>, labels: Tensor<B, 3>| {
        assert_eq!(scores.dims(), labels.dims());
        let labels = labels.detach();
        let count = labels.clone().sum().clamp_min(1.);
        -(activation::log_softmax(scores, 2) * labels).sum() / count
    };
    (nll(scores.clone(), forward) + nll(scores.swap_dims(1, 2), backward)) * 0.5
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
    fn known_permutation_beats_wrong_matches_and_labels_have_no_gradients() {
        let d = Default::default();
        let a = Tensor::<B, 3>::from_data(
            TensorData::new(vec![1., 0., 0., 1., -1., 0.], [1, 3, 2]),
            &d,
        )
        .require_grad();
        let b = Tensor::<B, 3>::from_data(
            TensorData::new(vec![0.1, 1., -1., 0.1, 1., 0.1], [1, 3, 2]),
            &d,
        )
        .require_grad();
        let labels = Tensor::<B, 3>::from_data(
            TensorData::new(vec![0., 0., 1., 1., 0., 0., 0., 1., 0.], [1, 3, 3]),
            &d,
        )
        .require_grad();
        let correct = bidirectional_nll(
            a.clone(),
            b.clone(),
            labels.clone(),
            labels.clone().swap_dims(1, 2),
            0.2,
        );
        let wrong = bidirectional_nll(
            a.clone(),
            b.clone(),
            labels.clone().swap_dims(1, 2),
            labels.clone(),
            0.2,
        );
        assert!(crate::tensor::scalar(wrong - correct.clone()).unwrap() > 3.);
        let grads = correct.backward();
        assert!(labels.grad(&grads).is_none());
        for x in [a, b] {
            assert!(crate::tensor::scalar(x.grad(&grads).unwrap().abs().sum()).unwrap() > 1e-4);
        }
    }

    #[test]
    fn invalid_rows_do_not_dilute_loss_and_empty_labels_are_zero() {
        let d = Default::default();
        let a = Tensor::<B, 3>::ones([1, 3, 2], &d).require_grad();
        let labels = Tensor::<B, 3>::from_data(
            TensorData::new(vec![0.25, 0.75, 0., 0., 0., 0., 0., 0., 0.], [1, 3, 3]),
            &d,
        );
        let loss = bidirectional_nll(a.clone(), a.clone(), labels.clone(), labels, 0.07);
        assert!((crate::tensor::scalar(loss).unwrap() - 3_f64.ln()).abs() < 1e-6);
        let zeros = Tensor::zeros([1, 3, 3], &d);
        let loss = bidirectional_nll(a.clone(), a.clone(), zeros.clone(), zeros, 0.07);
        assert_eq!(crate::tensor::scalar(loss.clone()).unwrap(), 0.);
        assert_eq!(
            crate::tensor::scalar(a.grad(&loss.backward()).unwrap().abs().sum()).unwrap(),
            0.
        );
    }
}
