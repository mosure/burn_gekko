use burn::backend::{Autodiff, NdArray};
use burn_gekko_data::{
    Split,
    head_cache::{ArrayFile, CAMERA_FRAME, CameraTarget, HeadCache, HeadSample},
    sha256_file, write_json,
};
use burn_gekko_train::training::heads::{HeadTrainConfig, run};
use serde_json::Value;
use std::{fs, path::Path};
type B = Autodiff<NdArray<f32>>;
fn fixture(root: &Path) -> HeadTrainConfig {
    let cache = root.join("cache");
    fs::create_dir(&cache).unwrap();
    let mut samples = Vec::new();
    let array = |file: String, data: Vec<f32>| {
        fs::write(
            cache.join(&file),
            data.iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        ArrayFile {
            sha256: sha256_file(&cache.join(&file)).unwrap(),
            file,
            values: data.len(),
        }
    };
    for i in 0..3 {
        let features: Vec<_> = (0..16 * 8)
            .map(|j| ((j + i * 11) as f32 * 0.1).sin())
            .collect();
        samples.push(HeadSample {
            room_seed: i + 1,
            split: if i < 2 {
                Split::Train
            } else {
                Split::Validation
            },
            target_view: 0,
            reference_view: 1,
            hidden_tokens: (1..16).collect(),
            completion: array(format!("{i}-cross.f32"), features.clone()),
            monocular: array(
                format!("{i}-mono.f32"),
                features.iter().map(|v| v * 0.8).collect(),
            ),
            camera_features: array(
                format!("{i}-camera.f32"),
                (0..32 * 4)
                    .map(|j| ((j + i * 5) as f32 * 0.15).cos())
                    .collect(),
            ),
            rgb: array(format!("{i}-rgb.f32"), vec![0.2 + i as f32 * 0.1; 16 * 768]),
            camera: CameraTarget {
                rotation: [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                translation: if i % 2 == 0 {
                    [1., 0., 0.]
                } else {
                    [0., 1., 0.]
                },
                focal: [0.7, 0.7],
            },
        });
    }
    let manifest = HeadCache {
        schema: 1,
        checkpoint_sha256: "a".repeat(64),
        dataset_id: "test".into(),
        dataset_manifest_sha256: "b".repeat(64),
        config_sha256: "c".repeat(64),
        source_sha256: "d".repeat(64),
        grid: [4, 4],
        latent_width: 8,
        camera_width: 4,
        camera_frame: CAMERA_FRAME.into(),
        input_contract: "synthetic fixture only".into(),
        samples,
    };
    manifest.validate().unwrap();
    write_json(&cache.join("manifest.json"), &manifest).unwrap();
    HeadTrainConfig {
        cache_sha256: sha256_file(&cache.join("manifest.json")).unwrap(),
        cache,
        checkpoint_sha256: "a".repeat(64),
        seed: 71,
        steps: 8,
        warmup_steps: 2,
        width: 8,
        rgb_batch: 4,
        camera_batch: 2,
        learning_rate: 0.001,
        weight_decay: 0.,
        max_seconds: 90,
        eval_every: 8,
    }
}
fn read(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}
#[test]
fn heads_resume_replays_the_same_updates_and_rejects_tampered_inputs() {
    fs::create_dir_all(".data").unwrap();
    let dir = tempfile::tempdir_in(".data").unwrap();
    let c = fixture(dir.path());
    let d = Default::default();
    let partial = dir.path().join("partial");
    let resumed = dir.path().join("resumed");
    let full = dir.path().join("full");
    run::<B>(&c, &partial, None, Some(4), &d).unwrap();
    run::<B>(&c, &resumed, Some(&partial.join("final")), None, &d).unwrap();
    run::<B>(&c, &full, None, None, &d).unwrap();
    let r = read(&resumed.join("report.json"));
    let f = read(&full.join("report.json"));
    assert_eq!(r["validation"], f["validation"]);
    assert_eq!(r["training"], f["training"]);
    let rows = |path: &Path| {
        fs::read_to_string(path.join("metrics.jsonl"))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .collect::<Vec<_>>()
    };
    assert_eq!(rows(&resumed), rows(&full)[4..]);
    let mut wrong = c.clone();
    wrong.learning_rate *= 2.;
    assert!(
        run::<B>(
            &wrong,
            &dir.path().join("wrong"),
            Some(&partial.join("final")),
            None,
            &d
        )
        .is_err()
    );
    let mut m = HeadCache::load(&c.cache, &c.cache_sha256).unwrap();
    m.samples[2].room_seed = 1;
    assert!(m.validate().is_err());
    fs::write(c.cache.join("0-cross.f32"), [0; 4]).unwrap();
    assert!(run::<B>(&c, &dir.path().join("tampered"), None, None, &d).is_err());
}
