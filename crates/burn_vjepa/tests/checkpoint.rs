//! Native-image adaptation of upstream's tiny_safetensors_loader_round_trips_burn_weights.
use burn::{
    backend::NdArray,
    tensor::{Tensor, TensorData},
};
use burn_vjepa::{
    SparseTokenMask, VJepa2_1Model, VJepaConfig, VJepaLoadOptions, load_burnpack_parts,
};
use burn_store::{BurnpackStore, ModuleSnapshot, SafetensorsStore};
type B = NdArray<f32>;

#[test]
fn image_checkpoint_roundtrip_preserves_dense_and_sparse_features() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.data/test-tmp");
    std::fs::create_dir_all(&root).unwrap();
    let tmp = tempfile::tempdir_in(root).unwrap();
    let device = Default::default();
    let config = VJepaConfig::tiny_for_tests();
    let model = VJepa2_1Model::<B>::new(&config, &device);
    let values = (0..3 * 32 * 32).map(|i| (i % 251) as f32 / 251.0).collect();
    let image = Tensor::from_data(TensorData::new(values, [1, 3, 32, 32]), &device);
    let mask = SparseTokenMask::new(vec![1, 3], 4).unwrap();
    std::fs::write(
        tmp.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let mut store =
        SafetensorsStore::from_file(tmp.path().join("model.safetensors")).overwrite(true);
    model.save_into(&mut store).unwrap();
    let (loaded, _, report) = VJepaLoadOptions {
        allow_partial: false,
        pytorch_adapter: false,
        upstream_vjepa21_names: false,
        ..Default::default()
    }
    .load_model::<B>(tmp.path(), &device)
    .unwrap();
    assert!(report.missing.is_empty() && report.errors.is_empty());
    for mask in [None, Some(&mask)] {
        let a = model.encoder.forward_image(image.clone(), mask).tokens;
        let b = loaded.encoder.forward_image(image.clone(), mask).tokens;
        let delta = (a - b).abs().max().into_data().to_vec::<f32>().unwrap()[0];
        assert!(delta < 1e-6);
    }
    let path = tmp.path().join("model.bpk");
    let mut store = BurnpackStore::from_file(&path).overwrite(true);
    model.save_into(&mut store).unwrap();
    let bytes = std::fs::read(path).unwrap();
    assert!(load_burnpack_parts::<B>(&config, vec![bytes.clone()], &device).is_ok());
    assert!(load_burnpack_parts::<B>(&config, vec![bytes.clone(), bytes], &device).is_err());
    assert!(load_burnpack_parts::<B>(&config, Vec::new(), &device).is_err());
}
