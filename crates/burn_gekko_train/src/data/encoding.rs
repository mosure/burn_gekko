//! Dataset-to-tensor adapters, isolated from the model library.
use anyhow::{Result, ensure};
use burn::tensor::{
    Tensor,
    backend::{AutodiffBackend, Backend},
};
pub use burn_gekko::encoder::*;
use burn_gekko_data::RgbScene;
use burn_vjepa::{SparseTokenMask, VJepaConfig, VJepaEncoder};
pub fn image_tensor<B: Backend>(scene: &RgbScene, view: usize, device: &B::Device) -> Tensor<B, 4> {
    upload_rgb(
        scene.views[view].clone(),
        [1, scene.height, scene.width],
        device,
    )
}

pub struct FrozenViews<B: AutodiffBackend> {
    pub masked: Tensor<B, 3>,
    pub full: Tensor<B, 3>,
    pub references: Vec<Tensor<B, 3>>,
    pub target_rgb: Tensor<B, 4>,
}
#[allow(clippy::too_many_arguments)]
pub fn encode_views<B: AutodiffBackend>(
    encoder: &VJepaEncoder<B::InnerBackend>,
    config: &VJepaConfig,
    scene: &RgbScene,
    target_view: usize,
    reference_count: usize,
    mask: &SparseTokenMask,
    features: ImageFeatures,
    device: &B::Device,
) -> Result<FrozenViews<B>> {
    ensure!(
        target_view < scene.views.len()
            && reference_count > 0
            && reference_count < scene.views.len(),
        "invalid view selection"
    );
    ensure!(
        scene.width.is_multiple_of(config.patch_size)
            && scene.height.is_multiple_of(config.patch_size),
        "image/patch mismatch"
    );
    let rgb = image_tensor::<B::InnerBackend>(scene, target_view, device);
    let image = normalize(rgb.clone(), config);
    // Crucially, hidden patches are removed before encoder attention, not after a dense forward.
    let masked = image_features(
        encoder.forward_image(image.clone(), Some(mask)).tokens,
        rgb.clone(),
        Some(mask),
        features,
    );
    let full = image_features(
        encoder.forward_image(image, None).tokens,
        rgb.clone(),
        None,
        features,
    );
    let references = (1..=reference_count)
        .map(|i| {
            let view = (target_view + i) % scene.views.len();
            let rgb = image_tensor::<B::InnerBackend>(scene, view, device);
            let image = normalize(rgb.clone(), config);
            Tensor::<B, 3>::from_inner(image_features(
                encoder.forward_image(image, None).tokens,
                rgb,
                None,
                features,
            ))
        })
        .collect();
    Ok(FrozenViews {
        masked: Tensor::from_inner(masked),
        full: Tensor::from_inner(full),
        references,
        target_rgb: Tensor::from_inner(rgb),
    })
}
