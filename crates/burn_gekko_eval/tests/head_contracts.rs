use burn_gekko_eval::reference_count::{Assessment, Config, score};
use serde_json::json;
use std::{fs, path::Path};
fn assessment(root: &Path, refs: usize, cross: f64, mono: f64) -> Assessment {
    let root = root.join(format!("refs-{refs}"));
    let dir = root.join("model");
    fs::create_dir_all(&dir).unwrap();
    fs::write(
        root.join("config.toml"),
        format!("references = {refs}\nseed = 1\nmask_pattern = \"random\"\n"),
    )
    .unwrap();
    burn_gekko_data::write_json(&dir.join("metrics.json"),&json!({"visible_tokens":[0],"hidden_tokens":[1],"split":"test","rows":[{"room_seed":1,"target_view":0,"cross_mse":cross,"monocular_mse":mono}]})).unwrap();
    burn_gekko_data::write_json(&dir.join("provenance.json"),&json!({"checkpoint":{"model_sha256":"a".repeat(64)},"dataset_id":"fixture","assessment_config_sha256":burn_gekko_data::sha256_file(&root.join("config.toml")).unwrap()})).unwrap();
    Assessment {
        references: refs,
        metrics_sha256: burn_gekko_data::sha256_file(&dir.join("metrics.json")).unwrap(),
        provenance_sha256: burn_gekko_data::sha256_file(&dir.join("provenance.json")).unwrap(),
        directory: dir,
    }
}
#[test]
fn reference_counts_bind_weights_masks_actual_counts_and_monocular_isolation() {
    let t = tempfile::tempdir().unwrap();
    let mut c = Config {
        checkpoint_sha256: "a".repeat(64),
        evaluations: vec![
            assessment(t.path(), 1, 0.2, 0.4),
            assessment(t.path(), 3, 0.1, 0.4),
        ],
        output: t.path().join("result.json"),
    };
    score(&c).unwrap();
    let result: serde_json::Value = serde_json::from_slice(&fs::read(&c.output).unwrap()).unwrap();
    assert_eq!(result["capability"]["metrics"][1]["value"], 0.1);
    c.output = t.path().join("bad.json");
    c.evaluations[1] = assessment(t.path(), 3, 0.1, 0.5);
    assert!(score(&c).unwrap_err().to_string().contains("monocular"));
    c.evaluations[1] = assessment(t.path(), 3, 0.1, 0.4);
    c.evaluations[1].references = 2;
    assert!(score(&c).unwrap_err().to_string().contains("count label"));
}
#[test]
fn reference_uncertainty_pairs_rooms_with_unequal_view_counts() {
    let t = tempfile::tempdir().unwrap();
    let mut evaluations = vec![
        assessment(t.path(), 3, 0.1, 0.8),
        assessment(t.path(), 1, 0.4, 0.8),
    ];
    for e in &mut evaluations {
        let path = e.directory.join("metrics.json");
        let mut data: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        data["rows"] = json!([(1,0,0.2),(1,1,0.),(2,0,0.6)].into_iter().map(|(seed,view,cross)|json!({"room_seed":seed,"target_view":view,"cross_mse":if e.references==1 {0.4} else {cross},"monocular_mse":0.8})).collect::<Vec<_>>());
        burn_gekko_data::write_json(&path, &data).unwrap();
        e.metrics_sha256 = burn_gekko_data::sha256_file(&path).unwrap();
    }
    let config = Config {
        checkpoint_sha256: "a".repeat(64),
        evaluations,
        output: t.path().join("result.json"),
    };
    score(&config).unwrap();
    let result: serde_json::Value =
        serde_json::from_slice(&fs::read(config.output.with_extension("provenance.json")).unwrap())
            .unwrap();
    let contrast = &result["contrasts"][0];
    assert_eq!(contrast["baseline_references"], 1);
    let interval = &contrast["paired_room_mse_reduction"];
    assert_eq!(interval["clusters"], 2);
    assert!((interval["mean"].as_f64().unwrap() - 0.05).abs() < 1e-12);
}
#[test]
fn camera_export_reports_valid_baselines_and_binds_its_input_contract() {
    use burn_gekko_eval::camera_export::{CameraScoreConfig, score};
    let t = tempfile::tempdir().unwrap();
    let records = t.path().join("camera.jsonl");
    let provenance = t.path().join("provenance.json");
    let eye = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    let record = json!({"sample":"camera-pair","predicted_rotation":eye,"target_rotation":eye,"predicted_translation":[1.,0.,0.],"target_translation":[2.,0.,0.],"predicted_focal":[0.8,0.8],"target_focal":[0.8,0.8]});
    fs::write(&records, format!("{record}\n")).unwrap();
    burn_gekko_data::write_json(&provenance,&json!({"checkpoint_sha256":"b".repeat(64),"input_contract":"full RGB pair","coordinate_frame":"anchor camera"})).unwrap();
    let mut config = CameraScoreConfig {
        records_sha256: burn_gekko_data::sha256_file(&records).unwrap(),
        provenance_sha256: burn_gekko_data::sha256_file(&provenance).unwrap(),
        records,
        provenance,
        checkpoint_sha256: "b".repeat(64),
        evaluation_use: "held_out".into(),
        input_contract: "full RGB pair".into(),
        coordinate_frame: "anchor camera".into(),
        output: t.path().join("head.json"),
    };
    score(&config).unwrap();
    let v: serde_json::Value = serde_json::from_slice(&fs::read(&config.output).unwrap()).unwrap();
    assert_eq!(v["capability"]["metrics"][0]["value"], 0.);
    assert_eq!(v["capability"]["metrics"][3]["value"], 1.);
    config.output = t.path().join("invalid.json");
    config.input_contract = "sparse target".into();
    assert!(
        score(&config)
            .unwrap_err()
            .to_string()
            .contains("protocol differs")
    );
}
