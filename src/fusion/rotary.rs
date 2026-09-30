//! Two-dimensional rotary coordinates, with reference views sharing the same image grid.
use burn::tensor::{Tensor, TensorData, backend::Backend};

pub struct Rotary2d<B: Backend> {
    sin: Tensor<B, 4>,
    cos: Tensor<B, 4>,
}
impl<B: Backend> Rotary2d<B> {
    pub fn select(&self, indices: Tensor<B, 1, burn::tensor::Int>) -> Self {
        Self {
            sin: self.sin.clone().select(2, indices.clone()),
            cos: self.cos.clone().select(2, indices),
        }
    }
    pub fn new(grid: [usize; 2], head_dim: usize, device: &B::Device) -> Self {
        assert!(head_dim.is_multiple_of(4));
        let quarter = head_dim / 4;
        let n = grid[0] * grid[1];
        let mut sin = Vec::with_capacity(n * head_dim);
        let mut cos = Vec::with_capacity(n * head_dim);
        for token in 0..n {
            for coordinate in [token / grid[1], token % grid[1]] {
                for _ in 0..2 {
                    for frequency in 0..quarter {
                        let angle =
                            coordinate as f32 / 100_f32.powf(frequency as f32 / quarter as f32);
                        sin.push(angle.sin());
                        cos.push(angle.cos());
                    }
                }
            }
        }
        Self {
            sin: Tensor::from_data(TensorData::new(sin, [1, 1, n, head_dim]), device),
            cos: Tensor::from_data(TensorData::new(cos, [1, 1, n, head_dim]), device),
        }
    }
    pub fn apply(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [b, h, n, d] = x.dims();
        let spatial_tokens = self.sin.dims()[2];
        assert!(n.is_multiple_of(spatial_tokens));
        let banks = n / spatial_tokens;
        let axes = x.clone().reshape([b, h, n, 2, 2, d / 4]);
        let rotated = Tensor::cat(
            vec![
                axes.clone().slice_dim(4, 1..2).neg(),
                axes.slice_dim(4, 0..1),
            ],
            4,
        )
        .reshape([b, h, n, d]);
        x * self.cos.clone().repeat_dim(2, banks) + rotated * self.sin.clone().repeat_dim(2, banks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    #[test]
    fn coordinates_rotate_the_correct_feature_axis() {
        type B = NdArray<f32>;
        let device = Default::default();
        let rope = Rotary2d::<B>::new([1, 2], 4, &device);
        let x = Tensor::from_data(
            TensorData::new(vec![1., 2., 3., 4., 1., 2., 3., 4.], [1, 1, 2, 4]),
            &device,
        );
        let values = rope.apply(x).into_data().to_vec::<f32>().unwrap();
        let expected = [
            1.,
            2.,
            3.,
            4.,
            1.,
            2.,
            3. * 1_f32.cos() - 4. * 1_f32.sin(),
            4. * 1_f32.cos() + 3. * 1_f32.sin(),
        ];
        for (a, b) in values.iter().zip(expected) {
            assert!((a - b).abs() < 1e-6);
        }
    }
    #[test]
    fn rotation_preserves_norm_and_repeats_coordinates_for_reference_sets() {
        type B = NdArray<f32>;
        let device = Default::default();
        let rope = Rotary2d::<B>::new([2, 3], 8, &device);
        let x = Tensor::<B, 4>::from_data(
            TensorData::new((0..48).map(|i| i as f32 / 19.0).collect(), [1, 1, 6, 8]),
            &device,
        );
        let r = rope.apply(x.clone());
        let before = x.clone().powf_scalar(2.0).sum_dim(3);
        let after = r.clone().powf_scalar(2.0).sum_dim(3);
        assert!(crate::tensor::scalar((before - after).abs().max()).unwrap() < 1e-5);
        let set = rope.apply(Tensor::cat(vec![x.clone(), x], 2));
        assert!(
            crate::tensor::scalar((set - Tensor::cat(vec![r.clone(), r], 2)).abs().max()).unwrap()
                < 1e-6
        );
    }
}
