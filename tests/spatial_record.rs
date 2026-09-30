use burn::{
    backend::NdArray,
    module::Module,
    nn::{Linear, LinearConfig},
    record::{FullPrecisionSettings, NamedMpkFileRecorder},
};
use burn_gekko::{
    latent::LatentFusion,
    model::{DecoderConfig, GekkoDecoder},
};
type B = NdArray<f32>;

/// The legacy field layout is intentionally frozen to exercise old named records.
#[derive(Module, Debug)]
struct LegacyFusion<B: burn::tensor::backend::Backend> {
    decoder: GekkoDecoder<B>,
    prediction: Linear<B>,
    improvement: Linear<B>,
}

#[test]
fn old_fusion_records_load_without_an_untrained_spatial_head() {
    let d = Default::default();
    let decoder = GekkoDecoder::<B>::new(
        &DecoderConfig {
            encoder_dim: 32,
            width: 16,
            depth: 1,
            heads: 2,
            patch: 16,
        },
        &d,
    )
    .unwrap();
    let legacy = LegacyFusion {
        decoder: decoder.clone(),
        prediction: LinearConfig::new(16, 32).init(&d),
        improvement: LinearConfig::new(16, 1).init(&d),
    };
    // Resolve Burn's lazy initializers before comparing cloned parameter records.
    let legacy = legacy.clone().load_record(legacy.into_record());
    let current = LatentFusion {
        decoder,
        prediction: legacy.prediction.clone(),
        improvement: legacy.improvement.clone(),
        spatial_descriptor: None,
    };
    std::fs::create_dir_all(".data/tests").unwrap();
    let tmp = tempfile::tempdir_in(".data/tests").unwrap();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    legacy
        .clone()
        .save_file(tmp.path().join("legacy"), &recorder)
        .unwrap();
    let loaded = current
        .load_file(tmp.path().join("legacy"), &recorder, &d)
        .unwrap();
    assert!(loaded.spatial_descriptor.is_none());
    assert_eq!(
        burn_gekko::tensor::values(legacy.prediction.weight.val()).unwrap(),
        burn_gekko::tensor::values(loaded.prediction.weight.val()).unwrap()
    );
}
