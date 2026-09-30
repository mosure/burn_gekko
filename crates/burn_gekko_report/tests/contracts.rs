use burn_gekko_eval::metrics::completion;
use burn_gekko_report::{
    artifact,
    experiment::{Experiment, LatentEvidence},
};
use serde_json::json;
use std::{fs, path::Path};

fn write(path: &Path, value: serde_json::Value) {
    fs::write(path, serde_json::to_vec_pretty(&value).unwrap()).unwrap();
}
fn fixture(root: &Path) -> Experiment {
    let run = root.join("run");
    let checkpoint = run.join("final");
    let eval = root.join("eval");
    fs::create_dir_all(&checkpoint).unwrap();
    fs::create_dir_all(&eval).unwrap();
    fs::write(checkpoint.join("model.mpk"), b"fixture model bytes").unwrap();
    let sha = burn_gekko_data::sha256_file(&checkpoint.join("model.mpk")).unwrap();
    write(
        &checkpoint.join("metadata.json"),
        json!({"model_sha256":sha,"completed_steps":2}),
    );
    write(
        &run.join("provenance.json"),
        json!({"noncommercial_weight_dependencies":[],"teacher_id":"fixture","dataset_id":"fixture"}),
    );
    write(
        &run.join("report.json"),
        json!({"starting_step":0,"completed_steps":2,"stage_steps":[2,0,0]}),
    );
    fs::write(
        run.join("config.toml"),
        "train_rooms = 2\nbatch_size = 1\nmask_ratio = 0.5\n",
    )
    .unwrap();
    fs::write(run.join("metrics.jsonl"),"{\"step\":1,\"cross\":0.2,\"monocular\":0.3,\"samples\":[[1,0]]}\n{\"step\":2,\"cross\":0.1,\"monocular\":0.2,\"samples\":[[2,0]]}\n").unwrap();
    let dir = eval.join("room-1-view-0");
    fs::create_dir(&dir).unwrap();
    let target = (0..24)
        .map(|i| ((i * i + 13) as f32).sin())
        .collect::<Vec<_>>();
    let pred = target.iter().map(|v| v * 0.8).collect::<Vec<_>>();
    let hidden = [0, 2];
    let metric = completion(&pred, &target, 6, &hidden).unwrap();
    for (name, v) in [
        ("target-latent", target),
        ("cross-latent", pred),
        ("target-rgb", vec![0.4; 32 * 32 * 3]),
        ("reference-1-rgb", vec![0.5; 32 * 32 * 3]),
    ] {
        fs::write(
            dir.join(format!("{name}.f32")),
            v.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>(),
        )
        .unwrap();
    }
    fs::write(dir.join("visibility.u8"), [1, 255, 0, 255]).unwrap();
    write(
        &dir.join("metadata.json"),
        json!({"room_seed":1,"target_view":0,"rgb_shape":[32,32,3],"grid":[2,2],"latent_shape":[4,6],"hidden_tokens":hidden}),
    );
    write(
        &eval.join("metrics.json"),
        json!({"task":"fixed_vjepa21_latent_prediction","target_views":1,"geometry_protocol":"fixture geometry after inference",
        "mean_cross_mse":metric.mse,"mean_cross_cosine":metric.cosine,"mean_monocular_mse":0.2,"mean_spatial_variance_ratio":metric.spatial_variance_ratio,
        "learned_ri_covisibility":{"pixels":2,"positives":1,"auroc":0.5,"average_precision":0.5},
        "rows":[{"room_seed":1,"target_view":0,"cross_mse":metric.mse,"cross_cosine":metric.cosine,"monocular_mse":0.2,"spatial_variance_ratio":metric.spatial_variance_ratio}]}),
    );
    write(
        &eval.join("provenance.json"),
        json!({"checkpoint":{"model_sha256":sha}}),
    );
    Experiment {
        schema: 1,
        id: "fixture".into(),
        title: "Fixture <test>".into(),
        author: "Fixture".into(),
        description: "Fixture".into(),
        architecture: "Fixture".into(),
        run,
        checkpoint,
        checkpoint_sha256: sha,
        latent: LatentEvidence {
            metrics_sha256: burn_gekko_data::sha256_file(&eval.join("metrics.json")).unwrap(),
            provenance_sha256: burn_gekko_data::sha256_file(&eval.join("provenance.json")).unwrap(),
            directory: eval,
            evaluation_use: "development".into(),
        },
        benchmarks: vec![],
        heads: vec![],
        equivariance: None,
        efficiency: None,
        limitations: vec!["Fixture only".into()],
        sample_count: 1,
    }
}
#[test]
fn publication_rejects_mixed_checkpoints_tampering_and_second_models() {
    let t = tempfile::tempdir().unwrap();
    let mut e = fixture(t.path());
    artifact::load(&e).unwrap();
    write(
        &e.latent.directory.join("provenance.json"),
        json!({"checkpoint":{"model_sha256":"another"}}),
    );
    assert!(
        artifact::load(&e)
            .unwrap_err()
            .to_string()
            .contains("checksum mismatch")
    );
    e.latent.provenance_sha256 =
        burn_gekko_data::sha256_file(&e.latent.directory.join("provenance.json")).unwrap();
    assert!(
        artifact::load(&e)
            .unwrap_err()
            .to_string()
            .contains("another checkpoint")
    );
    let mut v = serde_json::to_value(&e).unwrap();
    v["models"] = json!(["old", "new"]);
    assert!(serde_json::from_value::<Experiment>(v).is_err());
}
#[test]
fn report_uses_primary_population_and_phase_update_count() {
    let t = tempfile::tempdir().unwrap();
    let mut e = fixture(t.path());
    write(
        &e.run.join("report.json"),
        json!({"starting_step":1,"completed_steps":2}),
    );
    let path = t.path().join("benchmark.json");
    write(
        &path,
        json!({"checkpoint_sha256":e.checkpoint_sha256,
        "benchmark":"hpatches","evaluation_use":"development","pairs":580,
        "protocol":"viewpoint primary; illumination separate","comparability":"local readout",
        "methods":{"fused_decoder":{"primary":{"pairs":295,"aepe":10.,"pck3":0.25}}}}),
    );
    e.benchmarks
        .push(burn_gekko_report::experiment::PinnedFile {
            sha256: burn_gekko_data::sha256_file(&path).unwrap(),
            path,
        });
    let report = artifact::load(&e).unwrap();
    assert_eq!(report.training["completed_steps"], 1);
    let matching = report
        .capabilities
        .iter()
        .find(|c| c.id == "hpatches")
        .unwrap();
    assert!(matching.metrics.iter().all(|m| m.samples == 295));
}

#[test]
fn known_transform_report_rejects_mixed_weights_and_inconsistent_means() {
    let t = tempfile::tempdir().unwrap();
    let mut e = fixture(t.path());
    let path = t.path().join("warp.json");
    let mut data = json!({"status":"development_diagnostic", "checkpoint":{"model_sha256":e.checkpoint_sha256},"rooms":1,
        "methods":{"spatial_pair":{"mean_epe":4.,"pck8":1.,"pck16":1.,"nll":2.}},
        "records":[{"method":"spatial_pair","nll_bidirectional":2.,"score":{"mean_epe":4.,"pck_half_patch":1.,"pck_one_patch":1.}},
                   {"method":"spatial_pair","nll_bidirectional":2.,"score":{"mean_epe":4.,"pck_half_patch":1.,"pck_one_patch":1.}}]});
    let mut update = |data: &serde_json::Value| {
        write(&path, data.clone());
        e.equivariance = Some(burn_gekko_report::experiment::PinnedFile {
            path: path.clone(),
            sha256: burn_gekko_data::sha256_file(&path).unwrap(),
        });
        artifact::load(&e)
    };
    assert!(
        update(&data)
            .unwrap()
            .capabilities
            .iter()
            .any(|c| c.id == "image_transform")
    );
    data["checkpoint"]["model_sha256"] = json!("wrong");
    assert!(
        update(&data)
            .unwrap_err()
            .to_string()
            .contains("another checkpoint")
    );
    data["checkpoint"]["model_sha256"] =
        json!(burn_gekko_data::sha256_file(&t.path().join("run/final/model.mpk")).unwrap());
    data["methods"]["spatial_pair"]["mean_epe"] = json!(0.);
    assert!(
        update(&data)
            .unwrap_err()
            .to_string()
            .contains("differs from rows")
    );
}
#[test]
fn build_recomputes_visual_metrics_and_keeps_capability_gaps_visible() {
    let t = tempfile::tempdir().unwrap();
    let e = fixture(t.path());
    let spec = t.path().join("experiment.toml");
    burn_gekko_data::write_config(&spec, &e).unwrap();
    let out = t.path().join("site");
    burn_gekko_report::build(&spec, &out, false).unwrap();
    let html = fs::read_to_string(out.join("index.html")).unwrap();
    assert!(html.contains("Fixture &lt;test&gt;"));
    assert!(html.contains("Not trained"));
    assert!(!html.contains("<test>"));
    assert!(out.join("paper.tex").exists());
    assert!(out.join("bundle.json").exists());
    burn_gekko_report::validate::validate(&out).unwrap();
    fs::write(out.join("explore.js"), "changed").unwrap();
    assert!(
        burn_gekko_report::validate::validate(&out)
            .unwrap_err()
            .to_string()
            .contains("changed")
    );
    assert!(burn_gekko_report::build(&spec, &out, false).is_err());
    fs::write(
        e.latent.directory.join("room-1-view-0/cross-latent.f32"),
        vec![0u8; 24 * 4],
    )
    .unwrap();
    assert!(
        burn_gekko_report::build(&spec, &t.path().join("bad"), false)
            .unwrap_err()
            .to_string()
            .contains("disagree")
    );
}
