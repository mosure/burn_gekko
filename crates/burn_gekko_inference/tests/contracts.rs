use burn_gekko_inference::{RgbInput, pixels::unpatch};

#[test]
fn crop_patch_round_trip_and_mask_preserve_pixel_coordinates() {
    let image = image::RgbImage::from_fn(128, 64, |x, y| image::Rgb([x as u8, y as u8, 200]));
    let input = RgbInput::from_image("wide".into(), image.into(), 64).unwrap();
    assert_eq!(&input.pixels[..3], &[32, 0, 200]);
    assert_eq!(unpatch(&input.patches(), 64).unwrap(), input.pixels);
    let masked = input.masked(&[5]);
    assert_eq!(&masked[..3], &[20, 20, 20]);
    let position = (16 * 64 + 16) * 3;
    assert_eq!(
        &masked[position..position + 3],
        &input.pixels[position..position + 3]
    );
    assert!(RgbInput::decode("bad".into(), b"not an image", 256).is_err());
    assert!(unpatch(&[f32::NAN; 3], 1).is_err());
}

#[cfg(feature = "ndarray")]
mod model {
    use super::*;
    use burn::{
        backend::NdArray,
        module::Module,
        nn::LinearConfig,
        record::{FullPrecisionSettings, NamedMpkBytesRecorder, Recorder},
        tensor::backend::Backend,
    };
    use burn_gekko::{
        heads::{
            calibration::CalibrationHead, reconstruction::RgbReconstructionHead,
            spatial::SpatialDescriptorConfig,
        },
        latent::LatentModel,
        model::DecoderConfig,
    };
    use burn_gekko_inference::{Bundle, Inference, ModelFile, Request, bundle::digest};
    use burn_vjepa::{VJepaConfig, VJepaEncoder};
    type B = NdArray<f32>;

    fn fixture() -> (Bundle, Vec<u8>, Vec<u8>, Vec<u8>) {
        let device = Default::default();
        B::seed(&device, 47);
        let mut encoder = VJepaConfig::tiny_for_tests();
        encoder.image_size = 64;
        let decoder = DecoderConfig {
            encoder_dim: 32,
            width: 32,
            depth: 1,
            heads: 4,
            patch: 16,
        };
        let spatial = SpatialDescriptorConfig {
            residual_radius: 0.25,
        };
        let model = LatentModel::<B>::new(
            VJepaEncoder::new(&encoder, &device),
            encoder.clone(),
            &decoder,
            &device,
        )
        .unwrap()
        .prepare_spatial_descriptor(Some(&spatial))
        .unwrap()
        .with_spatial_input_layer(Some(2), true)
        .unwrap();
        let recorder = NamedMpkBytesRecorder::<FullPrecisionSettings>::default();
        let foundation = recorder.record(model.into_record(), ()).unwrap();
        let camera = recorder
            .record(CalibrationHead::<B>::new(32, 8, &device).into_record(), ())
            .unwrap();
        let mut rgb = RgbReconstructionHead::<B>::new(32, 8, &device);
        // Nonconstant output makes the hidden-pixel isolation check meaningful.
        rgb.output = LinearConfig::new(8, 768).init(&device);
        let rgb = recorder.record(rgb.into_record(), ()).unwrap();
        let pin = |name: &str, bytes: &[u8]| ModelFile {
            name: name.into(),
            bytes: bytes.len(),
            sha256: digest(bytes),
        };
        let bundle = Bundle {
            schema: 1,
            id: "test".into(),
            foundation_sha256: digest(&foundation),
            camera_sha256: digest(&camera),
            rgb_sha256: digest(&rgb),
            precision: "f32".into(),
            image_size: 64,
            encoder,
            decoder,
            spatial_descriptor: spatial,
            spatial_input_layer: 2,
            head_width: 8,
            mask_ratio: 0.75,
            mask_seed: 857,
            foundation: vec![pin("foundation.mpk", &foundation)],
            camera: pin("camera.mpk", &camera),
            rgb: pin("rgb.mpk", &rgb),
            license: "test".into(),
            qualification: "random fixture, not deployed".into(),
        };
        (bundle, foundation, camera, rgb)
    }

    #[test]
    fn exact_record_round_trip_and_hidden_target_isolation() {
        let (bundle, foundation, camera, rgb) = fixture();
        let model =
            Inference::<B>::load(bundle, foundation, camera, rgb, Default::default()).unwrap();
        let images = (0..3)
            .map(|v| RgbInput {
                name: format!("view {v}"),
                size: 64,
                pixels: (0..64 * 64 * 3)
                    .map(|i| ((i * 13 + v * 37) % 251) as u8)
                    .collect(),
            })
            .collect();
        let mut request = Request {
            revision: 7,
            target: 0,
            images,
        };
        let a = pollster::block_on(model.run(request.clone())).unwrap();
        assert!(a.rgb_score.psnr_db.unwrap().is_finite());
        assert!(a.rgb.iter().any(|v| *v != a.rgb[0]));
        assert_eq!(a.camera.len(), 11);
        assert_eq!(a.improvement.len(), 16);
        for y in 0..64 {
            for x in 0..64 {
                if !a.visible.contains(&(y / 16 * 4 + x / 16)) {
                    for c in 0..3 {
                        request.images[0].pixels[(y * 64 + x) * 3 + c] ^= 255;
                    }
                }
            }
        }
        let b = pollster::block_on(model.run(request.clone())).unwrap();
        assert_eq!(
            a.rgb, b.rgb,
            "dense annotation inputs must not leak into completion"
        );
        assert_eq!(a.monocular_rgb, b.monocular_rgb);
        assert_ne!(
            a.rgb_score.mse, b.rgb_score.mse,
            "changed hidden truth must change scoring"
        );
        request.images[1].pixels.fill(0);
        let c = pollster::block_on(model.run(request)).unwrap();
        assert_ne!(a.rgb, c.rgb, "the test model must use reference pixels");
    }

    #[test]
    fn malformed_manifests_and_corrupted_weights_are_rejected() {
        let (mut bundle, mut foundation, camera, rgb) = fixture();
        assert!(Bundle::parse(&toml::to_string(&bundle).unwrap()).is_ok());
        foundation[0] ^= 1;
        assert!(
            Inference::<B>::load(bundle.clone(), foundation, camera, rgb, Default::default())
                .is_err()
        );
        bundle.foundation[0].name = "../escape".into();
        assert!(bundle.validate().is_err());
        bundle.foundation[0].name = "camera.mpk".into();
        assert!(bundle.validate().is_err());
    }
}
