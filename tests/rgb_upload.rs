use burn::backend::NdArray;
use burn_gekko::encoder::{RgbUploadLayout, upload_rgb_layout};

#[test]
fn upload_layouts_preserve_batch_channel_and_nonsquare_pixel_coordinates() {
    type B = NdArray<f32>;
    let device = Default::default();
    let mut packed = Vec::new();
    for batch in 0..2 {
        for pixel in 0..15 {
            for channel in 0..3 {
                packed.push((100 * batch + 20 * channel + pixel) as f32);
            }
        }
    }
    for layout in [
        RgbUploadLayout::LegacyNhwc,
        RgbUploadLayout::FlatNhwc,
        RgbUploadLayout::PlanarNchw,
    ] {
        let image = upload_rgb_layout::<B>(packed.clone(), [2, 3, 5], layout, &device);
        assert_eq!(image.dims(), [2, 3, 3, 5]);
        let values = burn_gekko::tensor::values(image).unwrap();
        for batch in 0..2 {
            for channel in 0..3 {
                for pixel in 0..15 {
                    assert_eq!(
                        values[batch * 45 + channel * 15 + pixel],
                        (100 * batch + 20 * channel + pixel) as f32
                    );
                }
            }
        }
    }
}
