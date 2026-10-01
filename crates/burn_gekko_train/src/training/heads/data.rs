use anyhow::Result;
use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn_gekko_data::{
    Split,
    head_cache::{HeadCache, HeadSample},
};
use std::path::Path;
pub struct Sample {
    pub meta: HeadSample,
    pub cross: Vec<f32>,
    pub mono: Vec<f32>,
    pub camera: Vec<f32>,
    pub rgb: Vec<f32>,
}
pub fn load(root: &Path, cache: &HeadCache) -> Result<Vec<Sample>> {
    cache
        .samples
        .iter()
        .map(|m| {
            Ok(Sample {
                meta: m.clone(),
                cross: m.completion.load(root)?,
                mono: m.monocular.load(root)?,
                camera: m.camera_features.load(root)?,
                rgb: m.rgb.load(root)?,
            })
        })
        .collect()
}
pub struct TrainingTensors<B: Backend> {
    pub cross: Tensor<B, 3>,
    pub mono: Tensor<B, 3>,
    pub rgb: Tensor<B, 3>,
    pub camera: Tensor<B, 3>,
    pub labels: Tensor<B, 2>,
    pub translation_valid: Tensor<B, 2>,
    pub hidden_indices: Vec<usize>,
    pub train_indices: Vec<usize>,
}
impl<B: Backend> TrainingTensors<B> {
    pub fn new(samples: &[Sample], cache: &HeadCache, device: &B::Device) -> Self {
        let selected: Vec<_> = samples
            .iter()
            .enumerate()
            .filter(|(_, s)| s.meta.split == Split::Train)
            .collect();
        let n = selected.len();
        let tokens = cache.grid.iter().product::<usize>();
        let t3 =
            |values: Vec<f32>, shape| Tensor::from_data(TensorData::new(values, shape), device);
        Self {
            cross: t3(
                selected
                    .iter()
                    .flat_map(|(_, s)| s.cross.iter().copied())
                    .collect(),
                [n * tokens, 1, cache.latent_width],
            ),
            mono: t3(
                selected
                    .iter()
                    .flat_map(|(_, s)| s.mono.iter().copied())
                    .collect(),
                [n * tokens, 1, cache.latent_width],
            ),
            rgb: t3(
                selected
                    .iter()
                    .flat_map(|(_, s)| s.rgb.iter().copied())
                    .collect(),
                [n * tokens, 1, 768],
            ),
            camera: t3(
                selected
                    .iter()
                    .flat_map(|(_, s)| s.camera.iter().copied())
                    .collect(),
                [n, 32, cache.camera_width],
            ),
            labels: Tensor::from_data(
                TensorData::new(
                    selected
                        .iter()
                        .flat_map(|(_, s)| s.meta.camera.regression())
                        .collect(),
                    [n, 11],
                ),
                device,
            ),
            translation_valid: Tensor::from_data(
                TensorData::new(
                    selected
                        .iter()
                        .map(|(_, s)| f32::from(s.meta.camera.translation_valid()))
                        .collect(),
                    [n, 1],
                ),
                device,
            ),
            hidden_indices: selected
                .iter()
                .enumerate()
                .flat_map(|(i, (_, s))| s.meta.hidden_tokens.iter().map(move |t| i * tokens + t))
                .collect(),
            train_indices: selected.iter().map(|(i, _)| *i).collect(),
        }
    }
}
