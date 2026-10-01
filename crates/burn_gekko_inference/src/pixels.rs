use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RgbInput {
    pub name: String,
    pub size: usize,
    pub pixels: Vec<u8>,
}
impl RgbInput {
    /// A centered square crop followed by Lanczos resize; no guessed camera metadata.
    pub fn decode(name: String, bytes: &[u8], size: usize) -> Result<Self> {
        ensure!(bytes.len() <= 32 * 1024 * 1024, "image exceeds 32 MiB");
        let mut reader =
            image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format()?;
        let mut limits = image::Limits::default();
        limits.max_image_width = Some(8192);
        limits.max_image_height = Some(8192);
        limits.max_alloc = Some(256 * 1024 * 1024);
        reader.limits(limits);
        Self::from_image(name, reader.decode()?, size)
    }
    pub fn from_image(name: String, image: image::DynamicImage, size: usize) -> Result<Self> {
        ensure!(
            (64..=256).contains(&size) && size.is_multiple_of(16),
            "invalid image size"
        );
        let width = image.width();
        let height = image.height();
        let side = width.min(height);
        ensure!(side > 0, "empty image");
        let image = image
            .crop_imm((width - side) / 2, (height - side) / 2, side, side)
            .resize_exact(
                size as u32,
                size as u32,
                image::imageops::FilterType::Lanczos3,
            )
            .to_rgb8();
        Ok(Self {
            name,
            size,
            pixels: image.into_raw(),
        })
    }
    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.size > 0 && self.size <= 256 && self.pixels.len() == self.size * self.size * 3,
            "invalid image pixels"
        );
        Ok(())
    }
    pub fn patches(&self) -> Vec<f32> {
        let grid = self.size / 16;
        let mut out = vec![0.; self.pixels.len()];
        for y in 0..self.size {
            for x in 0..self.size {
                for c in 0..3 {
                    out[((y / 16 * grid + x / 16) * 256 + (y % 16 * 16 + x % 16)) * 3 + c] =
                        self.pixels[(y * self.size + x) * 3 + c] as f32 / 255.;
                }
            }
        }
        out
    }
    pub fn masked(&self, visible: &[usize]) -> Vec<u8> {
        let grid = self.size / 16;
        let mut pixels = self.pixels.clone();
        for y in 0..self.size {
            for x in 0..self.size {
                if !visible.contains(&(y / 16 * grid + x / 16)) {
                    pixels[(y * self.size + x) * 3..(y * self.size + x) * 3 + 3].fill(20);
                }
            }
        }
        pixels
    }
}
pub fn unpatch(values: &[f32], size: usize) -> Result<Vec<u8>> {
    ensure!(
        values.len() == size * size * 3 && values.iter().all(|v| v.is_finite()),
        "invalid RGB output"
    );
    let grid = size / 16;
    let mut out = vec![0; values.len()];
    for y in 0..size {
        for x in 0..size {
            for c in 0..3 {
                out[(y * size + x) * 3 + c] = (values
                    [((y / 16 * grid + x / 16) * 256 + (y % 16 * 16 + x % 16)) * 3 + c]
                    .clamp(0., 1.)
                    * 255.)
                    .round() as u8;
            }
        }
    }
    Ok(out)
}
