use burn_gekko_data::{sha256_file, write_json};
use burn_gekko_eval::{
    latent_detail::{DetailConfig, analyze},
    latent_replay,
};
use serde_json::json;
use std::{fs, path::Path};

fn fixture(root: &Path) -> DetailConfig {
    fs::create_dir_all(root.join("assessment/model/room-1-view-0")).unwrap();
    let directory = root.join("assessment/model");
    let dir = directory.join("room-1-view-0");
    for (name, values) in [
        ("target", [-3_f32, -1., 1., 3.]),
        ("cross", [-1.5, -0.5, 0.5, 1.5]),
        ("monocular", [-0.75, -0.25, 0.25, 0.75]),
    ] {
        fs::write(
            dir.join(format!("{name}-latent.f32")),
            values
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
    write_json(&dir.join("metadata.json"),&json!({"room_seed":1,"target_view":0,"grid":[2,2],"latent_shape":[4,1],"hidden_tokens":[0,1,2,3],"float_encoding":"little_endian_f32"})).unwrap();
    write_json(&directory.join("metrics.json"),&json!({"task":"fixed_vjepa21_latent_prediction","split":"test","hidden_tokens":4,"visible_tokens":0,"target_views":1,"mean_cross_mse":1.25,"mean_monocular_mse":2.8125,"rows":[{"room_seed":1,"target_view":0,"cross_mse":1.25,"monocular_mse":2.8125,"prediction_spatial_variance":1.25,"teacher_spatial_variance":5.,"cross_cosine":1.,"monocular_cosine":1.,"teacher_signal_power":5.,"feature_snr_db":6.020599913,"spatial_variance_ratio":0.25}]})).unwrap();
    let config = root.join("assessment/config.toml");
    fs::write(&config,"seed = 1\nreferences = 2\nmask_ratio = 0.9\nmask_pattern = \"random\"\nstable_attention = false\n").unwrap();
    write_json(&directory.join("provenance.json"),&json!({"checkpoint":{"model_sha256":"fixture"},"dataset_id":"fixture","teacher_id":"fixture","training_mean_sha256":"fixture","assessment_config_sha256":sha256_file(&config).unwrap()})).unwrap();
    DetailConfig {
        metrics_sha256: sha256_file(&directory.join("metrics.json")).unwrap(),
        provenance_sha256: sha256_file(&directory.join("provenance.json")).unwrap(),
        directory,
        checkpoint_sha256: "fixture".into(),
        output: root.join("detail.json"),
    }
}

#[test]
fn detail_requires_complete_arrays_and_reconciles_reported_metrics() {
    fs::create_dir_all(".data").unwrap();
    let tmp = tempfile::tempdir_in(".data").unwrap();
    let mut config = fixture(tmp.path());
    let result = analyze(&config).unwrap();
    assert_eq!(
        result["methods"]["cross"]["oracle_centered_gain"]["mean"],
        2.
    );
    assert_eq!(result["paired_reference_gains"]["mse"]["mean"], 1.5625);
    assert_eq!(result["sources"].as_object().unwrap().len(), 6);
    config.output = tmp.path().join("second.json");
    config.checkpoint_sha256 = "different".into();
    assert!(
        analyze(&config)
            .unwrap_err()
            .to_string()
            .contains("checkpoint mismatch")
    );
    config.checkpoint_sha256 = "fixture".into();
    let path = config.directory.join("room-1-view-0/cross-latent.f32");
    fs::write(&path, vec![0; 16]).unwrap();
    assert!(
        analyze(&config)
            .unwrap_err()
            .to_string()
            .contains("arrays disagree")
    );
    fs::remove_file(&path).unwrap();
    assert!(analyze(&config).is_err());
}

#[test]
fn replay_rejects_numeric_changes_and_relabels_without_new_inference() {
    fs::create_dir_all(".data").unwrap();
    let tmp = tempfile::tempdir_in(".data").unwrap();
    let a = fixture(&tmp.path().join("a"));
    let b = fixture(&tmp.path().join("b"));
    let export = |c: DetailConfig| latent_replay::Export {
        directory: c.directory,
        metrics_sha256: c.metrics_sha256,
        provenance_sha256: c.provenance_sha256,
    };
    let mut c = latent_replay::Config {
        original: export(a),
        replay: export(b),
        checkpoint_sha256: "fixture".into(),
        expected_target_views: 1,
        latent_absolute_tolerance: 1e-5,
        metric_absolute_tolerance: 2e-6,
        output: tmp.path().join("replay.json"),
    };
    assert_eq!(latent_replay::verify(&c).unwrap()["passed"], true);
    c.output = tmp.path().join("bad.json");
    fs::write(
        c.replay.directory.join("room-1-view-0/cross-latent.f32"),
        vec![0; 16],
    )
    .unwrap();
    let bad = latent_replay::verify(&c).unwrap();
    assert_eq!(bad["passed"], false);
    assert_eq!(bad["maximum_latent_difference"], 1.5);
    c.expected_target_views = 2;
    c.output = tmp.path().join("missing.json");
    assert!(latent_replay::verify(&c).is_err());
}
