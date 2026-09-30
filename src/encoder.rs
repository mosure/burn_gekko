use anyhow::{Result, ensure};
use burn::tensor::{Int, Tensor, TensorData, backend::Backend};
use burn_vjepa::{SparseTokenMask, VJepaConfig, VJepaEncoder, VJepaLoadOptions};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use std::path::Path;

/// Optional appearance bypass. It only sees RGB patches available to each branch.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageFeatures {
    #[default]
    Semantic,
    SemanticRgb,
}
impl ImageFeatures {
    pub fn width(self, encoder_dim: usize) -> usize {
        encoder_dim
            + if self == Self::SemanticRgb {
                16 * 16 * 3
            } else {
                0
            }
    }
}

pub fn image_features<B: Backend>(
    tokens: Tensor<B, 3>,
    rgb: Tensor<B, 4>,
    mask: Option<&SparseTokenMask>,
    mode: ImageFeatures,
) -> Tensor<B, 3> {
    if mode == ImageFeatures::Semantic {
        return tokens;
    }
    let mut patches = crate::loss::rgb_patches(rgb, 16, false) * 2.0 - 1.0;
    if let Some(mask) = mask {
        let indices = Tensor::<B, 1, Int>::from_data(
            TensorData::new(
                mask.indices().iter().map(|&i| i as i64).collect(),
                [mask.len()],
            ),
            &tokens.device(),
        );
        patches = patches.select(1, indices);
    }
    Tensor::cat(vec![tokens, patches], 2)
}

/// Counter-based sampling: checkpointing the step and seed completely specifies the next mask.
pub fn visible_mask(n: usize, ratio: f32, seed: u64, step: usize) -> Result<SparseTokenMask> {
    ensure!(
        n >= 2 && ratio.is_finite() && ratio > 0.0 && ratio < 1.0,
        "invalid mask ratio"
    );
    let mut indices: Vec<_> = (0..n).collect();
    let mut rng = ChaCha8Rng::seed_from_u64(
        seed.wrapping_add((step as u64).wrapping_mul(0x9e3779b97f4a7c15)),
    );
    indices.shuffle(&mut rng);
    indices.truncate(((n as f32 * (1.0 - ratio)).round() as usize).clamp(1, n - 1));
    SparseTokenMask::new(indices, n)
}

#[derive(Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RgbUploadLayout {
    LegacyNhwc,
    FlatNhwc,
    PlanarNchw,
}

/// Upload packed RGB as one flat buffer before making an NCHW tensor view.
/// Uploading NHWC directly can request a pitched CUDA copy with 12-byte rows.
pub fn upload_rgb<B: Backend>(
    data: Vec<f32>,
    shape: [usize; 3],
    device: &B::Device,
) -> Tensor<B, 4> {
    upload_rgb_layout(data, shape, RgbUploadLayout::FlatNhwc, device)
}

/// Explicit alternatives are retained for the bounded upload benchmark.
pub fn upload_rgb_layout<B: Backend>(
    data: Vec<f32>,
    [batch, height, width]: [usize; 3],
    layout: RgbUploadLayout,
    device: &B::Device,
) -> Tensor<B, 4> {
    let count = batch * height * width * 3;
    assert!(batch > 0 && height > 0 && width > 0 && data.len() == count);
    match layout {
        RgbUploadLayout::LegacyNhwc => {
            Tensor::from_data(TensorData::new(data, [batch, height, width, 3]), device)
                .permute([0, 3, 1, 2])
        }
        RgbUploadLayout::FlatNhwc => {
            Tensor::<B, 1>::from_data(TensorData::new(data, [count]), device)
                .reshape([batch, height, width, 3])
                .permute([0, 3, 1, 2])
        }
        RgbUploadLayout::PlanarNchw => {
            let mut planar = Vec::with_capacity(count);
            for image in data.chunks_exact(height * width * 3) {
                for channel in 0..3 {
                    planar.extend(image.as_chunks::<3>().0.iter().map(|pixel| pixel[channel]));
                }
            }
            Tensor::from_data(TensorData::new(planar, [batch, 3, height, width]), device)
        }
    }
}
pub fn normalize<B: Backend>(rgb: Tensor<B, 4>, config: &VJepaConfig) -> Tensor<B, 4> {
    // Dataset color is already [0,1]; applying rescale_factor again would be incorrect.
    let device = rgb.device();
    let mean = Tensor::<B, 4>::from_data(
        TensorData::new(config.preprocess.image_mean.to_vec(), [1, 3, 1, 1]),
        &device,
    );
    let std = Tensor::<B, 4>::from_data(
        TensorData::new(config.preprocess.image_std.to_vec(), [1, 3, 1, 1]),
        &device,
    );
    (rgb - mean) / std
}

pub fn load_checkpoint<B: Backend>(
    directory: &Path,
    device: &B::Device,
) -> Result<(VJepaEncoder<B>, VJepaConfig)> {
    let options = VJepaLoadOptions {
        allow_partial: false,
        ..VJepaLoadOptions::default()
    };
    let (model, config, report) = options.load_model(directory, device)?;
    ensure!(
        report.missing.is_empty() && report.errors.is_empty(),
        "incomplete checkpoint: missing={:?}, errors={:?}",
        report.missing,
        report.errors
    );
    ensure!(
        config.model_type == "vjepa2_1",
        "pretrained mode requires a V-JEPA 2.1 configuration"
    );
    Ok((model.encoder, config))
}
