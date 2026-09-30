//! Checked tensor readback helpers shared by diagnostics and tests.
use anyhow::{Result, ensure};
use burn::tensor::{Tensor, backend::Backend};
pub fn scalar<B: Backend>(x: Tensor<B, 1>) -> Result<f64> {
    let data = x.into_data().convert::<f32>();
    let v = data.as_slice::<f32>()?[0] as f64;
    ensure!(v.is_finite(), "nonfinite scalar");
    Ok(v)
}

pub fn values<B: Backend, const D: usize>(x: Tensor<B, D>) -> Result<Vec<f32>> {
    let v = x.into_data().convert::<f32>().to_vec::<f32>()?;
    ensure!(v.iter().all(|x| x.is_finite()), "nonfinite latent output");
    Ok(v)
}
