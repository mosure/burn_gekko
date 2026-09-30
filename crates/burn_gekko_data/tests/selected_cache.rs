use burn_gekko_data::*;
use safetensors::{Dtype, tensor::TensorView};
use std::{fs, path::Path};

fn fixture(root: &Path) -> DatasetManifest {
    fs::create_dir(root.join("raw")).unwrap();
    let config = CaptureConfig {
        seed: 120,
        width: 32,
        height: 32,
        cameras: 2,
        split_scenes: [1, 1, 1],
        ..Default::default()
    };
    let binary_sha256 = "00".repeat(32);
    let mut manifest = DatasetManifest {
        schema: SCHEMA,
        dataset_id: fingerprint(&(SCHEMA, GENERATOR, &binary_sha256, &config)).unwrap(),
        generator: GENERATOR.into(),
        binary_sha256,
        config,
        scenes: Vec::new(),
    };
    for i in 0..3 {
        let seed = 120 + i as u64;
        let rgb = (0..2 * 32 * 32 * 3)
            .flat_map(|v| ((v % 19) as f32 / 19.).to_le_bytes())
            .collect::<Vec<_>>();
        let indoor = format!("{{\"seed\":{seed}}}").into_bytes();
        let arrays = [
            ("color", Dtype::F32, vec![1, 1, 2, 32, 32, 3], rgb),
            ("color_encoding", Dtype::U8, vec![1], vec![2]),
            ("annotation_precision", Dtype::U8, vec![1], vec![1]),
            ("indoor_manifest_0", Dtype::U8, vec![indoor.len()], indoor),
        ];
        let views = arrays
            .iter()
            .map(|(name, dtype, shape, bytes)| {
                (
                    *name,
                    TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        let file = format!("room-{seed}.safetensors");
        let path = root.join("raw").join(&file);
        safetensors::tensor::serialize_to_file(views, None, &path).unwrap();
        manifest.scenes.push(SceneEntry {
            file,
            sha256: sha256_file(&path).unwrap(),
            seed,
            split: manifest.config.split(i).unwrap(),
        });
    }
    write_json(&root.join("manifest.json"), &manifest).unwrap();
    manifest
}

#[test]
fn selected_reads_preserve_pixels_and_checksums_while_full_audit_still_checks_unused_shards() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.data/test-tmp");
    fs::create_dir_all(&root).unwrap();
    let tmp = tempfile::tempdir_in(root).unwrap();
    let m = fixture(tmp.path());
    let validated = open_dataset(tmp.path()).unwrap();
    let rgb = load_dataset_rgb(tmp.path(), &validated, &validated.scenes[0]).unwrap();
    let original = load_rgb(&tmp.path().join("raw").join(&m.scenes[0].file)).unwrap();
    assert!(rgb.views == original.views);
    assert_eq!(rgb.seed, original.seed);
    fs::write(
        tmp.path().join("raw").join(&m.scenes[2].file),
        b"corrupted unused shard",
    )
    .unwrap();
    let selected = read_dataset_manifest(tmp.path()).unwrap();
    load_dataset_rgb(tmp.path(), &selected, &selected.scenes[0]).unwrap();
    assert!(
        load_dataset_rgb(tmp.path(), &selected, &selected.scenes[2])
            .unwrap_err()
            .to_string()
            .contains("checksum")
    );
    assert!(
        open_dataset(tmp.path())
            .unwrap_err()
            .to_string()
            .contains("checksum")
    );
    let mut forged = selected.scenes[0].clone();
    forged.file = selected.scenes[1].file.clone();
    assert!(
        load_dataset_rgb(tmp.path(), &selected, &forged)
            .unwrap_err()
            .to_string()
            .contains("manifest")
    );
}
