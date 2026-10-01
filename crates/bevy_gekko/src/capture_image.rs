//! Decode Zeroverse floating-point render targets after Bevy GPU readback.
use anyhow::{Result, ensure};
use bevy::{prelude::Image, render::render_resource::TextureFormat};
use burn_gekko_inference::RgbInput;

pub fn input(image: &Image, name: String) -> Result<RgbInput> {
    let dynamic = if image.texture_descriptor.format == TextureFormat::Rgba32Float {
        let bytes = image
            .data
            .as_deref()
            .ok_or_else(|| anyhow::anyhow!("empty capture"))?;
        let size = image.texture_descriptor.size;
        ensure!(
            bytes.len() == size.width as usize * size.height as usize * 16,
            "invalid float capture shape"
        );
        let mut values: Vec<_> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(f32::from_le_bytes)
            .collect();
        ensure!(
            values.iter().all(|x| x.is_finite()),
            "nonfinite capture pixels"
        );
        // Match the published dataset exporter exactly: the render target is
        // tone-mapped linear RGB; training inputs and uploaded photos are sRGB.
        for pixel in values.as_chunks_mut::<4>().0 {
            for channel in &mut pixel[..3] {
                *channel = bevy_zeroverse::render::color::linear_to_srgb(*channel);
            }
        }
        image::DynamicImage::ImageRgba32F(
            image::ImageBuffer::from_raw(size.width, size.height, values)
                .ok_or_else(|| anyhow::anyhow!("invalid float capture buffer"))?,
        )
    } else {
        image
            .clone()
            .try_into_dynamic()
            .map_err(|e| anyhow::anyhow!("{e}"))?
    };
    // Integer/sRGB inputs already have their transfer function applied.
    RgbInput::from_image(name, dynamic, 256)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::{
        asset::RenderAssetUsages,
        render::render_resource::{Extent3d, TextureDimension},
    };
    #[test]
    fn float_readback_preserves_rgb_and_rejects_nan() {
        let pixel: Vec<_> = [0.18_f32, 0.5, 1., 1.]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect();
        let mut image = Image::new_fill(
            Extent3d {
                width: 64,
                height: 64,
                depth_or_array_layers: 1,
            },
            TextureDimension::D2,
            &pixel,
            TextureFormat::Rgba32Float,
            RenderAssetUsages::MAIN_WORLD,
        );
        let captured = input(&image, "test".into()).unwrap();
        assert_eq!(&captured.pixels[..3], &[118, 188, 255]);
        image.data.as_mut().unwrap()[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        assert!(input(&image, "invalid".into()).is_err());
    }
}
