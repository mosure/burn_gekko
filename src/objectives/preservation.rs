//! Preserve an encoder's feature coordinates against a frozen, audited ancestor.
use burn::tensor::{Tensor, backend::Backend};

/// Mean per-image squared feature error divided by that image's anchor energy.
/// Token count and activation scale therefore do not weight one view over another.
/// Unlike cosine alignment, this penalizes changes in feature magnitude and offset.
pub fn relative_feature_mse<B: Backend>(
    student: Tensor<B, 3>,
    anchor: Tensor<B, 3>,
) -> Tensor<B, 1> {
    assert_eq!(student.dims(), anchor.dims(), "preservation feature shape");
    let anchor = anchor.detach();
    let energy = anchor.clone().powf_scalar(2.).mean_dim(2).mean_dim(1);
    ((student - anchor).powf_scalar(2.).mean_dim(2).mean_dim(1) / energy.clamp_min(1e-6)).mean()
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
    fn energy_normalization_is_per_image_and_anchor_has_no_gradient() {
        let d = Default::default();
        let anchor =
            Tensor::<B, 3>::from_data(TensorData::new(vec![1., 1., 2., 2.], [2, 1, 2]), &d)
                .require_grad();
        let student =
            Tensor::<B, 3>::from_data(TensorData::new(vec![2., 2., 4., 4.], [2, 1, 2]), &d)
                .require_grad();
        let loss = relative_feature_mse(student.clone(), anchor.clone());
        assert_eq!(crate::tensor::scalar(loss.clone()).unwrap(), 1.);
        let grads = loss.backward();
        assert!(anchor.grad(&grads).is_none());
        assert_eq!(
            crate::tensor::values(student.grad(&grads).unwrap()).unwrap(),
            [0.5, 0.5, 0.25, 0.25]
        );
        assert_eq!(
            crate::tensor::scalar(relative_feature_mse(student * 7., anchor * 7.)).unwrap(),
            1.
        );
    }

    #[test]
    fn identical_or_zero_anchors_are_finite_without_discarding_offsets() {
        let d = Default::default();
        let ones = Tensor::<B, 3>::ones([1, 3, 2], &d);
        assert_eq!(
            crate::tensor::scalar(relative_feature_mse(ones.clone(), ones.clone())).unwrap(),
            0.
        );
        assert_eq!(
            crate::tensor::scalar(relative_feature_mse(ones.clone() + 1., ones)).unwrap(),
            1.
        );
        let zeros = Tensor::<B, 3>::zeros([1, 3, 2], &d);
        assert_eq!(
            crate::tensor::scalar(relative_feature_mse(zeros.clone(), zeros)).unwrap(),
            0.
        );
    }
}
