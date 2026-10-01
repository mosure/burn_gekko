use burn::backend::{Autodiff, NdArray};
use burn_gekko_data::{
    Split, sha256_file,
    view_targets::{Cache, PrepareConfig, prepare},
};
use burn_gekko_train::latent_pilot::{LatentConfig, ViewGeometryConfig, run};
use serde_json::Value;
use std::{fs, path::Path};

/// Exercise target cache integrity, training gradients, optimizer resume and RGB-only assessment.
pub fn verify(parent: &LatentConfig, root: &Path) {
    type B = Autodiff<NdArray<f32>>;
    let cache = root.join("view-targets");
    prepare(&PrepareConfig {
        dataset: parent.dataset.clone(),
        output: cache.clone(),
        rooms_per_split: [2, 1, 0],
        patch: 16,
    })
    .unwrap();
    let targets = Cache::load(&cache, &parent.dataset).unwrap();
    assert_eq!(targets.manifest.valid_queries, 24);
    assert!(
        targets
            .room_index(targets.manifest.rooms[0].seed, Split::Validation)
            .is_err()
    );
    let before = sha256_file(&cache.join("targets.bin")).unwrap();
    let mut c = parent.clone();
    c.view_geometry = Some(ViewGeometryConfig {
        cache: cache.clone(),
        weight: 0.1,
        temperature: 0.07,
    });
    c.steps = 1;
    let first = root.join("geometry-first");
    run::<B>(&c, &first, None, &Default::default()).unwrap();
    c.steps = 2;
    let resumed = root.join("geometry-resumed");
    run::<B>(
        &c,
        &resumed,
        Some(&first.join("final")),
        &Default::default(),
    )
    .unwrap();
    let full = root.join("geometry-full");
    run::<B>(&c, &full, None, &Default::default()).unwrap();
    let rows = |p: &Path| {
        fs::read_to_string(p.join("metrics.jsonl"))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap())
            .collect::<Vec<_>>()
    };
    let (r, f) = (rows(&resumed), rows(&full));
    for key in [
        "samples",
        "total",
        "gradient_norm",
        "view_geometry_nll",
        "view_geometry_valid_fraction",
    ] {
        assert_eq!(r[0][key], f[1][key]);
    }
    assert!(
        f.iter()
            .all(|r| r["view_geometry_nll"].as_f64().unwrap() > 0.
                && r["view_geometry_valid_fraction"] == 1.)
    );
    let report: Value =
        serde_json::from_slice(&fs::read(full.join("report.json")).unwrap()).unwrap();
    assert!(report["first_encoder_max_abs_delta"].as_f64().unwrap() > 0.);
    assert_eq!(report["teacher_max_abs_delta"], 0.);
    assert_eq!(
        report["preservation_anchor_qkv_max_abs_delta"],
        serde_json::json!([0., 0.])
    );
    assert_eq!(sha256_file(&cache.join("targets.bin")).unwrap(), before);
    // The inference loader does not read or require training label caches.
    let weights = burn_gekko_train::latent_pilot::WeightAncestor {
        checkpoint: full.join("final"),
        model_sha256: sha256_file(&full.join("final/model.mpk")).unwrap(),
    };
    let bytes = fs::read(cache.join("targets.bin")).unwrap();
    fs::write(cache.join("targets.bin"), b"corrupt").unwrap();
    assert!(Cache::load(&cache, &parent.dataset).is_err());
    burn_gekko_train::latent_assess::load_assessed_model::<NdArray<f32>>(
        &weights,
        &Default::default(),
    )
    .unwrap();
    let assessment = burn_gekko_train::latent_assess::AssessmentConfig {
        dataset: parent.dataset.clone(),
        split: Split::Validation,
        rooms: 1,
        export_rooms: 1,
        mask_ratio: 0.5,
        mask_pattern: burn_gekko_train::masking::MaskPattern::Random,
        seed: parent.seed,
        stable_attention: false,
        correspondence: false,
        references: Some(1),
        models: vec![burn_gekko_train::latent_assess::AssessmentModel {
            name: "model".into(),
            weights,
        }],
    };
    let output = root.join("geometry-assessment");
    burn_gekko_train::latent_assess::run::<NdArray<f32>>(&assessment, &output, &Default::default())
        .unwrap();
    let provenance: Value =
        serde_json::from_slice(&fs::read(output.join("model/provenance.json")).unwrap()).unwrap();
    assert_eq!(provenance["evaluation_geometry_used_for_training"], false);
    assert_eq!(
        provenance["selected_training_phase_view_geometry"]["weight"],
        0.1
    );
    assert_eq!(
        provenance["training_config_sha256"],
        sha256_file(&full.join("config.toml")).unwrap()
    );
    assert!(provenance.get("geometry_training_supervision").is_none());
    fs::write(cache.join("targets.bin"), bytes).unwrap();
}
