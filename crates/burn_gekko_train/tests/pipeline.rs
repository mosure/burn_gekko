use burn::backend::{Autodiff, NdArray};
use burn::module::Module;
use burn_gekko_data::{
    CaptureConfig, DatasetManifest, GENERATOR, SCHEMA, SceneEntry, Split, audit, fingerprint,
    generate, load_rgb, open_dataset, read_config, read_config_snapshot, sha256_file, write_config,
    write_json,
};
use burn_gekko_train::train::{TrainConfig, train};
use safetensors::{Dtype, tensor::TensorView};
use std::{fs, path::Path};

// NdArray seeds its process-global RNG; seeded training runs must not overlap.
static TRAINING_RNG: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn temp() -> tempfile::TempDir {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(".data/test-tmp");
    fs::create_dir_all(&root).unwrap();
    tempfile::tempdir_in(root).unwrap()
}

#[test]
fn toml_examples_validate_and_roundtrip_without_changing_fingerprints() {
    use burn_gekko_train::pilot::PilotConfig;
    use serde::{Serialize, de::DeserializeOwned};
    fn check<T: Serialize + DeserializeOwned>(name: &str, validate: impl Fn(&T)) {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures")
            .join(name);
        let value: T = read_config(&path).unwrap();
        validate(&value);
        let tmp = temp();
        let saved = tmp.path().join("resolved.toml");
        write_config(&saved, &value).unwrap();
        let restored: T = read_config(&saved).unwrap();
        assert_eq!(
            fingerprint(&value).unwrap(),
            fingerprint(&restored).unwrap()
        );
    }
    for name in ["data/capture-preflight.toml", "data/capture-pilot.toml"] {
        check::<CaptureConfig>(name, |c| c.validate().unwrap());
    }
    for name in [
        "train/train-preflight.toml",
        "train/train-vjepa-base-preflight.toml",
    ] {
        check::<TrainConfig>(name, |c| c.validate().unwrap());
    }
    for name in [
        "train/pilot-b1-online.toml",
        "train/pilot-b4-online.toml",
        "train/pilot-b4-cache.toml",
        "train/pilot-overfit.toml",
        "train/pilot-smallset.toml",
    ] {
        check::<PilotConfig>(name, |c| c.validate().unwrap());
    }
    for name in [
        "experiments/pilot07-equivariance-preflight.toml",
        "experiments/pilot08-equivariance-proposed.toml",
        "experiments/pilot08-equivariance.toml",
        "experiments/pilot08-tail-preflight.toml",
    ] {
        check::<burn_gekko_train::latent_pilot::LatentConfig>(name, |c| c.validate().unwrap());
    }
}

#[test]
fn initial_encoder_stage_requires_an_audited_phase_and_respects_the_cap() {
    use burn_gekko_train::latent_pilot::LatentConfig;
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/experiments/pilot08-equivariance-proposed.toml");
    let mut c: LatentConfig = read_config(&path).unwrap();
    c.initial_encoder_stage = 1;
    c.validate().unwrap();
    c.encoder_stage_cap = 0;
    assert!(c.validate().is_err());
    c.encoder_stage_cap = 1;
    c.unfreeze = false;
    assert!(c.validate().is_err());
    c.unfreeze = true;
    c.warm_start = None;
    assert!(c.validate().is_err());
    c.initial_encoder_stage = 0;
    c.validate().unwrap();
}

#[test]
fn toml_snapshots_take_precedence_and_invalid_toml_never_falls_back_to_json() {
    let tmp = temp();
    let legacy = TrainConfig::default();
    let json_path = tmp.path().join("config.json");
    write_json(&json_path, &legacy).unwrap();
    assert_eq!(
        read_config_snapshot::<TrainConfig>(tmp.path(), "config").unwrap(),
        legacy
    );
    assert!(read_config::<TrainConfig>(&json_path).is_err());
    let current = TrainConfig { steps: 3, ..legacy };
    let toml_path = tmp.path().join("config.toml");
    write_config(&toml_path, &current).unwrap();
    assert_eq!(
        read_config_snapshot::<TrainConfig>(tmp.path(), "config").unwrap(),
        current
    );
    let text = fs::read_to_string(&toml_path).unwrap();
    fs::write(&toml_path, format!("unexpected_key = true\n{text}")).unwrap();
    assert!(read_config_snapshot::<TrainConfig>(tmp.path(), "config").is_err());
    fs::write(&toml_path, "[broken TOML").unwrap();
    assert!(read_config_snapshot::<TrainConfig>(tmp.path(), "config").is_err());
}
fn chunk(path: &Path, seed: u64) {
    let mut tensors: Vec<(String, Dtype, Vec<usize>, Vec<u8>)> = Vec::new();
    let mut f32s = |name: &str, shape: Vec<usize>, values: Vec<f32>| {
        tensors.push((
            name.into(),
            Dtype::F32,
            shape,
            values.into_iter().flat_map(f32::to_le_bytes).collect(),
        ))
    };
    let rgb = (0..2 * 32 * 32 * 3)
        .map(|i| ((i as u64 + seed) % 251) as f32 / 251.0)
        .collect();
    f32s("color", vec![1, 1, 2, 32, 32, 3], rgb);
    f32s("depth", vec![1, 1, 2, 32, 32, 1], vec![2.0; 2 * 32 * 32]);
    let mut position = Vec::new();
    for _ in 0..2 {
        for y in 0..32 {
            for x in 0..32 {
                position.extend([
                    (x as f32 + 0.5 - 16.0) / 64.0 + 0.5,
                    -(y as f32 + 0.5 - 16.0) / 64.0 + 0.5,
                    0.25,
                ]);
            }
        }
    }
    f32s("position", vec![1, 1, 2, 32, 32, 3], position);
    f32s("aabb", vec![1, 2, 3], vec![-4., -4., -4., 4., 4., 4.]);
    let identity = vec![
        1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.,
    ];
    f32s("world_from_view", vec![1, 1, 2, 4, 4], identity.repeat(2));
    f32s(
        "fovy",
        vec![1, 1, 2, 1],
        vec![std::f32::consts::FRAC_PI_2; 2],
    );
    tensors.push(("color_encoding".into(), Dtype::U8, vec![1], vec![2]));
    tensors.push(("annotation_precision".into(), Dtype::U8, vec![1], vec![1]));
    let manifest = serde_json::to_vec(&serde_json::json!({"seed":seed})).unwrap();
    tensors.push((
        "indoor_manifest_0".into(),
        Dtype::U8,
        vec![manifest.len()],
        manifest,
    ));
    let views: Vec<_> = tensors
        .iter()
        .map(|(name, dtype, shape, data)| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), data).unwrap(),
            )
        })
        .collect();
    fs::write(path, safetensors::serialize(views, None).unwrap()).unwrap();
}
fn fixture(root: &Path) -> std::path::PathBuf {
    fixture_with_seed(root, CaptureConfig::default().seed)
}

#[test]
fn latent_resume_preserves_teacher_two_optimizers_and_sampling() {
    let _rng = TRAINING_RNG.lock().unwrap();
    use burn_gekko_train::latent_pilot::{LatentConfig, run};
    use burn_gekko_train::train::EncoderSource;
    type B = Autodiff<NdArray<f32>>;
    let tmp = temp();
    let mut c: LatentConfig = read_config(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/archive/pilot-07/pilot07-latent-screen.toml"),
    )
    .unwrap();
    c.dataset = fixture(tmp.path());
    c.teacher = EncoderSource::DiagnosticTiny;
    c.train_rooms = 2;
    c.validation_rooms = 1;
    c.export_rooms = 1;
    c.batch_size = 1;
    c.steps = 2;
    c.decay_steps = 8;
    c.warmup_steps = 0;
    c.max_seconds = 300;
    c.eval_every = 1;
    c.checkpoint_every = 1;
    c.references = 1;
    c.decoder_width = 32;
    c.decoder_depth = 1;
    c.decoder_heads = 4;
    c.ri_start_step = 0;
    c.unfreeze_min_steps = 1;
    c.unfreeze_min_improvement = 0.;
    let d = Default::default();
    let first = tmp.path().join("latent-first");
    run::<B>(&c, &first, None, &d).unwrap();
    c.steps = 3;
    let resumed = tmp.path().join("latent-resumed");
    run::<B>(&c, &resumed, Some(&first.join("checkpoint-000002")), &d).unwrap();
    let full = tmp.path().join("latent-full");
    run::<B>(&c, &full, None, &d).unwrap();
    let json =
        |p: &Path| serde_json::from_slice::<serde_json::Value>(&fs::read(p).unwrap()).unwrap();
    let a = json(&resumed.join("report.json"));
    let b = json(&full.join("report.json"));
    assert_eq!(a["teacher_max_abs_delta"], 0.);
    assert!(
        (a["final_validation"]["mean_cross_mse"].as_f64().unwrap()
            - b["final_validation"]["mean_cross_mse"].as_f64().unwrap())
        .abs()
            < 1e-7
    );
    let rows = |p: &Path| {
        fs::read_to_string(p)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
            .collect::<Vec<_>>()
    };
    let ar = rows(&resumed.join("metrics.jsonl"));
    let br = rows(&full.join("metrics.jsonl"));
    assert_eq!(ar[0]["samples"], br[2]["samples"]);
    assert_eq!(ar[0]["total"], br[2]["total"]);
    assert_eq!(ar[0]["stage"], br[2]["stage"]);
    assert!(
        br.iter()
            .any(|r| r["encoder_gradient_tensors"].as_u64().unwrap() > 0)
    );
    assert!(
        br.iter()
            .any(|r| r["encoder_gradient_tensors"].as_u64().unwrap() == 0)
    );
    let parent = full.join("final");
    let hash = json(&parent.join("metadata.json"))["model_sha256"]
        .as_str()
        .unwrap()
        .to_string();
    c.warm_start = Some(burn_gekko_train::latent_pilot::WeightAncestor {
        checkpoint: parent.clone(),
        model_sha256: hash.clone(),
    });
    c.steps = 1;
    let warm = tmp.path().join("latent-warm");
    run::<B>(&c, &warm, None, &d).unwrap();
    let w = json(&warm.join("report.json"));
    assert_eq!(w["starting_step"], 0);
    assert_eq!(
        rows(&warm.join("metrics.jsonl"))[0]["encoder_gradient_tensors"],
        0
    );
    assert_eq!(
        w["initial_validation_cross_mse"],
        b["final_validation"]["mean_cross_mse"]
    );
    let mut guided = c.clone();
    guided.steps = 2;
    guided.unfreeze = false;
    guided.encoder_stage_cap = 0;
    guided.fusion_auxiliary = burn_gekko_train::fusion_objective::FusionAuxiliary {
        attention_weight: 0.1,
        dense_weight: 0.1,
        descriptor_weight: 0.1,
        teacher_temperature: 0.07,
        descriptor_temperature: None,
        anchor_warm_start: true,
        bidirectional: true,
        teacher_layers: Vec::new(),
    };
    let guided_run = tmp.path().join("latent-guided");
    run::<B>(&guided, &guided_run, None, &d).unwrap();
    let guided_report = json(&guided_run.join("report.json"));
    assert_eq!(guided_report["teacher_max_abs_delta"], 0.0);
    assert_eq!(guided_report["first_encoder_max_abs_delta"], 0.0);
    assert!(
        rows(&guided_run.join("metrics.jsonl"))
            .iter()
            .all(|x| x["descriptor_kl"].as_f64().unwrap().is_finite())
    );
    // The tiny encoder's only trained hierarchy level is its final block.
    // Selecting it explicitly must preserve the complete auxiliary objective.
    guided.fusion_auxiliary.teacher_layers = vec![2];
    let hierarchical_run = tmp.path().join("latent-guided-hierarchy");
    run::<B>(&guided, &hierarchical_run, None, &d).unwrap();
    for (plain, hierarchical) in rows(&guided_run.join("metrics.jsonl"))
        .iter()
        .zip(rows(&hierarchical_run.join("metrics.jsonl")))
    {
        for key in [
            "total",
            "cross",
            "monocular",
            "attention_kl",
            "dense_latent_mse",
            "descriptor_kl",
        ] {
            assert_eq!(plain[key], hierarchical[key]);
        }
    }
    let assessment = burn_gekko_train::latent_assess::AssessmentConfig {
        references: Some(c.references),
        dataset: c.dataset.clone(),
        split: burn_gekko_data::Split::Validation,
        rooms: 1,
        export_rooms: 0,
        mask_ratio: 0.75,
        mask_pattern: burn_gekko_train::masking::MaskPattern::Random,
        seed: c.seed,
        stable_attention: false,
        correspondence: true,
        models: vec![burn_gekko_train::latent_assess::AssessmentModel {
            name: "tiny".into(),
            weights: c.warm_start.clone().unwrap(),
        }],
    };
    let eval = tmp.path().join("latent-assessment");
    let protocol = tmp.path().join("assessment.toml");
    burn_gekko_data::write_config(&protocol, &assessment).unwrap();
    let assessment: burn_gekko_train::latent_assess::AssessmentConfig =
        read_config(&protocol).unwrap();
    burn_gekko_train::latent_assess::run::<NdArray<f32>>(&assessment, &eval, &d).unwrap();
    let assessed = json(&eval.join("tiny/metrics.json"));
    assert_eq!(
        assessed["mean_cross_mse"],
        b["final_validation"]["mean_cross_mse"]
    );
    // A new direct feature route changes the projection width, but must retain
    // warm-start behavior and both optimizer states on exact continuation.
    // The tiny encoder has one trained norm, so use its final block as the
    // appended level here; the distinct middle-level path is covered separately.
    let mut spatial = c.clone();
    spatial.spatial_input_layer = Some(2);
    spatial.steps = 1;
    let spatial_first = tmp.path().join("latent-spatial-first");
    run::<B>(&spatial, &spatial_first, None, &d).unwrap();
    let neutral = json(&spatial_first.join("report.json"));
    assert!(
        (neutral["initial_validation_cross_mse"].as_f64().unwrap()
            - b["final_validation"]["mean_cross_mse"].as_f64().unwrap())
        .abs()
            < 1e-6
    );
    spatial.steps = 2;
    let spatial_resumed = tmp.path().join("latent-spatial-resumed");
    run::<B>(
        &spatial,
        &spatial_resumed,
        Some(&spatial_first.join("final")),
        &d,
    )
    .unwrap();
    let spatial_full = tmp.path().join("latent-spatial-full");
    run::<B>(&spatial, &spatial_full, None, &d).unwrap();
    let resumed_rows = rows(&spatial_resumed.join("metrics.jsonl"));
    let full_rows = rows(&spatial_full.join("metrics.jsonl"));
    for key in ["samples", "stage", "total", "gradient_norm"] {
        assert_eq!(resumed_rows[0][key], full_rows[1][key]);
    }
    let spatial_parent = burn_gekko_train::latent_pilot::WeightAncestor {
        checkpoint: spatial_full.join("final"),
        model_sha256: json(&spatial_full.join("final/metadata.json"))["model_sha256"]
            .as_str()
            .unwrap()
            .to_string(),
    };
    let loaded =
        burn_gekko_train::latent_assess::load_assessed_model::<NdArray<f32>>(&spatial_parent, &d)
            .unwrap();
    assert_eq!(
        loaded
            .model
            .encode(burn::tensor::Tensor::zeros([1, 3, 32, 32], &d), None)
            .dims(),
        [1, 4, 64]
    );
    let mut descriptor = spatial.clone();
    descriptor.unfreeze = false;
    descriptor.encoder_stage_cap = 0;
    descriptor.spatial_descriptor = Some(burn_gekko::heads::spatial::SpatialDescriptorConfig {
        residual_radius: 0.25,
    });
    descriptor.fusion_auxiliary = guided.fusion_auxiliary.clone();
    descriptor.fusion_auxiliary.teacher_temperature = 0.035;
    descriptor.fusion_auxiliary.descriptor_temperature = Some(0.07);
    descriptor.steps = 1;
    let descriptor_first = tmp.path().join("spatial-descriptor-first");
    run::<B>(&descriptor, &descriptor_first, None, &d).unwrap();
    descriptor.steps = 2;
    let descriptor_resume = tmp.path().join("spatial-descriptor-resume");
    run::<B>(
        &descriptor,
        &descriptor_resume,
        Some(&descriptor_first.join("final")),
        &d,
    )
    .unwrap();
    let descriptor_full = tmp.path().join("spatial-descriptor-full");
    run::<B>(&descriptor, &descriptor_full, None, &d).unwrap();
    for key in ["samples", "total", "gradient_norm", "descriptor_kl"] {
        assert_eq!(
            rows(&descriptor_resume.join("metrics.jsonl"))[0][key],
            rows(&descriptor_full.join("metrics.jsonl"))[1][key]
        );
    }
    let weights = burn_gekko_train::latent_pilot::WeightAncestor {
        checkpoint: descriptor_full.join("final"),
        model_sha256: json(&descriptor_full.join("final/metadata.json"))["model_sha256"]
            .as_str()
            .unwrap()
            .into(),
    };
    let loaded =
        burn_gekko_train::latent_assess::load_assessed_model::<NdArray<f32>>(&weights, &d).unwrap();
    assert!(
        burn_gekko::tensor::scalar(
            loaded
                .model
                .fusion
                .spatial_descriptor
                .as_ref()
                .unwrap()
                .correction
                .weight
                .val()
                .abs()
                .max()
        )
        .unwrap()
            > 0.
    );
    let bare = loaded
        .model
        .clone()
        .prepare_spatial_descriptor(None)
        .unwrap();
    assert!(
        bare.load_checked_record(loaded.model.into_record())
            .is_err()
    );
    // Known-transform training must replay its sampled transforms and both
    // descriptor branches across a genuine optimizer checkpoint boundary.
    let mut warped = descriptor.clone();
    warped.unfreeze = true;
    warped.encoder_stage_cap = 1;
    warped.initial_encoder_stage = 1;
    warped.fusion_auxiliary = Default::default();
    warped.equivariance = Some(burn_gekko_train::data::augmentation::EquivarianceConfig {
        max_rotation_degrees: 0.,
        min_scale: 1.,
        max_scale: 1.,
        max_perspective: 0.,
        min_valid_fraction: 0.2,
        ..Default::default()
    });
    warped.steps = 1;
    let warp_first = tmp.path().join("warp-first");
    run::<B>(&warped, &warp_first, None, &d).unwrap();
    warped.steps = 2;
    let warp_resume = tmp.path().join("warp-resume");
    run::<B>(&warped, &warp_resume, Some(&warp_first.join("final")), &d).unwrap();
    let warp_full = tmp.path().join("warp-full");
    run::<B>(&warped, &warp_full, None, &d).unwrap();
    let (resumed, full) = (
        rows(&warp_resume.join("metrics.jsonl")),
        rows(&warp_full.join("metrics.jsonl")),
    );
    for key in [
        "samples",
        "total",
        "gradient_norm",
        "warp_pair_nll",
        "warp_self_nll",
        "warp_valid_fraction",
    ] {
        assert_eq!(resumed[0][key], full[1][key]);
    }
    assert!(full[1]["warp_pair_nll"].as_f64().unwrap() > 0.);
    assert!(full[1]["warp_self_nll"].as_f64().unwrap() > 0.);
    assert!(full.iter().all(|row| row["stage"] == 1));
    let warp_report = json(&warp_full.join("report.json"));
    assert_eq!(warp_report["teacher_max_abs_delta"], 0.);
    assert!(warp_report["last_encoder_max_abs_delta"].as_f64().unwrap() > 0.);
    c.warm_start.as_mut().unwrap().model_sha256 = "wrong".into();
    assert!(run::<B>(&c, &tmp.path().join("latent-bad-warm"), None, &d).is_err());
    c.warm_start = None;
    c.steps = 3;
    let meta = first.join("checkpoint-000002/metadata.json");
    let mut bad = json(&meta);
    bad["noncommercial_weight_dependencies"] = serde_json::json!(["prohibited teacher"]);
    write_json(&meta, &bad).unwrap();
    assert!(
        run::<B>(
            &c,
            &tmp.path().join("latent-bad"),
            Some(&first.join("checkpoint-000002")),
            &d
        )
        .is_err()
    );
}

#[test]
fn e2e_resume_preserves_both_optimizers_and_rejects_nc_provenance() {
    e2e_resume_contract(false);
}
#[test]
fn appearance_resume_preserves_optimizer_state_and_audited_warm_start() {
    e2e_resume_contract(true);
}
fn e2e_resume_contract(appearance: bool) {
    let _rng = TRAINING_RNG.lock().unwrap();
    use burn_gekko_train::e2e_pilot::{E2eConfig, Initialization, run};
    type B = Autodiff<NdArray<f32>>;
    let tmp = temp();
    let dataset = fixture(tmp.path());
    let mut c: E2eConfig = read_config(
        &Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/archive/pilot-05/pilot05-cuda-smoke.toml"),
    )
    .unwrap();
    c.dataset = dataset;
    c.initialization = Initialization::Scratch {
        width: 32,
        depth: 2,
        heads: 4,
    };
    c.decoder_width = 32;
    c.decoder_depth = 1;
    c.decoder_heads = 4;
    c.references = 1;
    c.batch_size = 1;
    c.steps = 1;
    c.decay_steps = 4;
    c.warmup_steps = 1;
    c.learning_rate = 0.001;
    c.eval_every = 1;
    c.gradient_energy_weight = 0.;
    c.rgb_head = burn_gekko_train::e2e::RgbHead::Direct;
    c.appearance_transport = appearance;
    c.transport_matching = appearance;
    c.synthetic_reference_probability = if appearance { 1. } else { 0. };
    c.transport_pyramid_loss = appearance;
    c.transport_coarse_smoothness_weight = if appearance { 0.01 } else { 0. };
    c.transport_max_displacement = 16.;
    c.rgb_cache = burn_gekko_train::e2e_pilot::RgbCache::Host;
    c.checkpoint_every = 1;
    c.ri_start_step = 1;
    let device = Default::default();
    let first = tmp.path().join("e2e-first");
    run::<B>(&c, &first, None, None, false, &device).unwrap();
    c.steps = 2;
    let resume = tmp.path().join("e2e-resume");
    let full = tmp.path().join("e2e-full");
    run::<B>(
        &c,
        &resume,
        None,
        Some(&first.join("checkpoint-000001")),
        false,
        &device,
    )
    .unwrap();
    run::<B>(&c, &full, None, None, false, &device).unwrap();
    let json = |path: std::path::PathBuf| {
        serde_json::from_slice::<serde_json::Value>(&fs::read(path).unwrap()).unwrap()
    };
    let a = json(resume.join("report.json"));
    let b = json(full.join("report.json"));
    assert!(
        (a["validation_mse"].as_f64().unwrap() - b["validation_mse"].as_f64().unwrap()).abs()
            < 1e-7
    );
    let lines = |p: std::path::PathBuf| {
        fs::read_to_string(p)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str::<serde_json::Value>(l).unwrap())
            .collect::<Vec<_>>()
    };
    let resumed = lines(resume.join("metrics.jsonl"));
    let uninterrupted = lines(full.join("metrics.jsonl"));
    assert_eq!(resumed[0]["samples"], uninterrupted[1]["samples"]);
    assert_eq!(resumed[0]["loss"], uninterrupted[1]["loss"]);
    let mut fine_tune = c.clone();
    fine_tune.dataset = fixture_with_seed(&tmp.path().join("new-capture"), 909101);
    fine_tune.learning_rate = 0.0007;
    fine_tune.steps = 1;
    fine_tune.decay_steps = 6;
    fine_tune.appearance_transport = true;
    fine_tune.transport_matching = true;
    fine_tune.transport_pyramid_loss = true;
    fine_tune.transport_coarse_smoothness_weight = 0.01;
    let warm_run = tmp.path().join("weights-only");
    burn_gekko_train::e2e_pilot::run_with_warm_start::<B>(
        &fine_tune,
        &warm_run,
        None,
        None,
        Some(&first.join("final")),
        false,
        &device,
    )
    .unwrap();
    let warm_meta = json(warm_run.join("final/metadata.json"));
    assert_eq!(warm_meta["completed_steps"], 1);
    assert_eq!(warm_meta["warm_start"]["source_completed_steps"], 1);
    assert_eq!(warm_meta["warm_start"]["optimizer_reset"], true);
    assert_ne!(
        warm_meta["dataset_id"],
        warm_meta["warm_start"]["source_dataset_id"]
    );
    assert_eq!(json(warm_run.join("report.json"))["start_step"], 0);
    assert!(
        json(warm_run.join("report.json"))["matcher_max_abs_delta"]
            .as_f64()
            .unwrap()
            > 0.
    );
    assert_eq!(
        json(warm_run.join("provenance.json"))["matching_initialization"],
        if appearance {
            "own_checkpoint_weights"
        } else {
            "random"
        }
    );
    let mut wrong_head = fine_tune.clone();
    wrong_head.rgb_head = burn_gekko_train::e2e::RgbHead::Calibrated;
    wrong_head.appearance_transport = false;
    wrong_head.transport_matching = false;
    wrong_head.transport_pyramid_loss = false;
    wrong_head.transport_coarse_smoothness_weight = 0.;
    let error = burn_gekko_train::e2e_pilot::run_with_warm_start::<B>(
        &wrong_head,
        &tmp.path().join("wrong-warm-head"),
        None,
        None,
        Some(&first.join("final")),
        false,
        &device,
    )
    .unwrap_err();
    assert!(error.to_string().contains("architecture"));
    let meta = first.join("final/metadata.json");
    let mut forbidden = json(meta.clone());
    forbidden["noncommercial_weight_dependencies"] = serde_json::json!(["released-gekko"]);
    write_json(&meta, &forbidden).unwrap();
    let error = run::<B>(
        &c,
        &tmp.path().join("nc-rejected"),
        None,
        Some(&first.join("final")),
        false,
        &device,
    )
    .unwrap_err();
    assert!(error.to_string().contains("provenance"));
    let error = burn_gekko_train::e2e_pilot::run_with_warm_start::<B>(
        &fine_tune,
        &tmp.path().join("nc-warm-rejected"),
        None,
        None,
        Some(&first.join("final")),
        false,
        &device,
    )
    .unwrap_err();
    assert!(error.to_string().contains("noncommercial"));
}
fn fixture_with_seed(root: &Path, seed: u64) -> std::path::PathBuf {
    let dir = root.join("dataset");
    fs::create_dir_all(dir.join("raw")).unwrap();
    let config = CaptureConfig {
        width: 32,
        height: 32,
        seed,
        ..Default::default()
    };
    let binary_sha256 = "fixture-not-a-rendered-dataset".to_string();
    let mut scenes = Vec::new();
    for i in 0..config.scenes() {
        let file = format!("{i:06}.safetensors");
        let seed = config.seed + i as u64;
        let path = dir.join("raw").join(&file);
        chunk(&path, seed);
        scenes.push(SceneEntry {
            file,
            sha256: sha256_file(&path).unwrap(),
            seed,
            split: config.split(i).unwrap(),
        });
    }
    let manifest = DatasetManifest {
        schema: SCHEMA,
        dataset_id: fingerprint(&(SCHEMA, GENERATOR, &binary_sha256, &config)).unwrap(),
        generator: GENERATOR.into(),
        binary_sha256,
        config,
        scenes,
    };
    write_json(&dir.join("manifest.json"), &manifest).unwrap();
    dir
}

#[test]
fn dataset_roundtrip_separates_rooms_and_detects_tampering() {
    let tmp = temp();
    let dir = fixture(tmp.path());
    let manifest = open_dataset(&dir).unwrap();
    assert_eq!(
        manifest
            .scenes
            .iter()
            .filter(|s| s.split == Split::Train)
            .count(),
        2
    );
    let rgb = load_rgb(&dir.join("raw/000000.safetensors")).unwrap();
    assert_eq!(rgb.views.len(), 2);
    let geometry = audit(&dir).unwrap();
    assert_eq!(geometry.visible, 4 * 2 * 32 * 32);
    assert!(geometry.max_self_reprojection_pixels < 1e-4);
    let mut bad = manifest.clone();
    bad.scenes[3].seed = bad.scenes[0].seed;
    write_json(&dir.join("manifest.json"), &bad).unwrap();
    assert!(open_dataset(&dir).is_err());
    write_json(&dir.join("manifest.json"), &manifest).unwrap();
    let file = dir.join("raw/000000.safetensors");
    let mut data = fs::read(&file).unwrap();
    *data.last_mut().unwrap() ^= 1;
    fs::write(file, data).unwrap();
    assert!(open_dataset(&dir).is_err());
}

#[test]
fn bounded_capture_rejects_failure_and_incomplete_success() {
    let tmp = temp();
    let cfg = CaptureConfig::default();
    assert!(generate(&tmp.path().join("failed"), Path::new("/bin/false"), &cfg).is_err());
    assert!(generate(&tmp.path().join("incomplete"), Path::new("/bin/true"), &cfg).is_err());
    let mut oversized = cfg;
    oversized.split_scenes = [32768, 1, 1];
    assert!(oversized.validate().is_err());
    oversized.split_scenes = [8192, 128, 128];
    oversized.camera_baseline = Some(0.25);
    assert!(oversized.validate().is_ok());
    oversized.camera_baseline = Some(f32::NAN);
    assert!(oversized.validate().is_err());
}

#[cfg(unix)]
#[test]
fn complete_capture_is_cached_and_reused_without_restarting_generator() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = temp();
    let source = fixture(tmp.path());
    let script = tmp.path().join("capture.sh");
    // Test the process protocol and cache, using small analytical fixtures instead of invoking a GPU.
    fs::write(
        &script,
        format!(
            "#!/bin/sh\nset -eu\nif [ \"$1\" = '--identity' ]; then printf '%s\\n' '{}'; exit 0; fi\nprintf 'called\\n' >> '{}'/calls\ncp '{}'/raw/* \"$4\"/\n",
            GENERATOR,
            tmp.path().display(),
            source.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
    let config = CaptureConfig {
        width: 32,
        height: 32,
        ..Default::default()
    };
    let first = generate(&tmp.path().join("cache"), &script, &config).unwrap();
    assert_eq!(
        read_config::<CaptureConfig>(&first.join("capture.toml")).unwrap(),
        config
    );
    assert!(!first.join("capture.json").exists());
    let second = generate(&tmp.path().join("cache"), &script, &config).unwrap();
    assert_eq!(first, second);
    assert_eq!(
        fs::read_to_string(tmp.path().join("calls")).unwrap(),
        "called\n"
    );
    fs::write(first.join("raw/000000.safetensors"), b"corrupt").unwrap();
    assert!(generate(&tmp.path().join("cache"), &script, &config).is_err());
    assert_eq!(
        fs::read_to_string(tmp.path().join("calls")).unwrap(),
        "called\n"
    );
}

#[test]
fn checkpoint_restores_optimizer_and_next_sample_exactly() {
    let _rng = TRAINING_RNG.lock().unwrap();
    type B = Autodiff<NdArray<f32>>;
    let tmp = temp();
    let dataset = fixture(tmp.path());
    let device = Default::default();
    let one = TrainConfig {
        steps: 1,
        ..Default::default()
    };
    let first = train::<B>(&dataset, &tmp.path().join("first"), &one, None, &device).unwrap();
    let saved = tmp.path().join("first");
    assert_eq!(
        read_config::<TrainConfig>(&saved.join("config.toml")).unwrap(),
        one
    );
    assert!(!saved.join("config.json").exists());
    let two = TrainConfig {
        steps: 2,
        ..one.clone()
    };
    let resumed = train::<B>(
        &dataset,
        &tmp.path().join("resumed"),
        &two,
        Some(&first.checkpoint),
        &device,
    )
    .unwrap();
    let continuous = train::<B>(
        &dataset,
        &tmp.path().join("continuous"),
        &two,
        None,
        &device,
    )
    .unwrap();
    assert_eq!(resumed.start_step, 1);
    assert_eq!(resumed.train[0].room_seed, continuous.train[1].room_seed);
    assert!((resumed.train[0].total - continuous.train[1].total).abs() < 1e-6);
    assert!((resumed.validation_total - continuous.validation_total).abs() < 1e-6);
    let current_eval = burn_gekko_train::eval::evaluate::<NdArray<f32>>(
        &dataset,
        &saved,
        Split::Validation,
        &device,
        None,
    )
    .unwrap();
    let export_dir = tmp.path().join("annotations");
    let exported = burn_gekko_train::eval::evaluate_with_options::<NdArray<f32>>(
        &dataset,
        &saved,
        Split::Validation,
        &device,
        None,
        &burn_gekko_train::eval::EvaluationOptions {
            export_rooms: 1,
            export_directory: Some(export_dir),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(exported.samples.len(), 1);
    let sample = &exported.samples[0];
    assert_eq!(
        fs::metadata(sample.join("target.f32")).unwrap().len(),
        32 * 32 * 3 * 4
    );
    assert_eq!(
        fs::metadata(sample.join("cross.f32")).unwrap().len(),
        32 * 32 * 3 * 4
    );
    assert_eq!(
        fs::read(sample.join("visibility.u8")).unwrap(),
        vec![1; 32 * 32]
    );
    assert!((exported.mean_total_loss - current_eval.mean_total_loss).abs() < 1e-6);
    // Generalization evaluation must preserve checkpoint identity and prove
    // disjointness from the entire original cache, not merely rename a split.
    let fresh = fixture_with_seed(&tmp.path().join("fresh"), 991000);
    assert!(
        burn_gekko_train::eval::evaluate::<NdArray<f32>>(
            &fresh,
            &saved,
            Split::Test,
            &device,
            None,
        )
        .is_err()
    );
    let generalization_options = burn_gekko_train::eval::EvaluationOptions {
        training_dataset: Some(dataset.clone()),
        ..Default::default()
    };
    let fresh_eval = burn_gekko_train::eval::evaluate_with_options::<NdArray<f32>>(
        &fresh,
        &saved,
        Split::Test,
        &device,
        None,
        &generalization_options,
    )
    .unwrap();
    assert_eq!(fresh_eval.training_dataset_id, current_eval.dataset_id);
    assert_ne!(fresh_eval.dataset_id, current_eval.dataset_id);
    let overlap = fixture_with_seed(
        &tmp.path().join("overlap"),
        CaptureConfig::default().seed + 1,
    );
    let rejected = burn_gekko_train::eval::evaluate_with_options::<NdArray<f32>>(
        &overlap,
        &saved,
        Split::Test,
        &device,
        None,
        &generalization_options,
    )
    .unwrap_err();
    assert!(rejected.to_string().contains("overlap"));
    // An old recorded run remains evaluable without rewriting its configuration.
    write_json(&saved.join("config.json"), &one).unwrap();
    fs::remove_file(saved.join("config.toml")).unwrap();
    let legacy_eval = burn_gekko_train::eval::evaluate::<NdArray<f32>>(
        &dataset,
        &saved,
        Split::Validation,
        &device,
        None,
    )
    .unwrap();
    assert!((legacy_eval.mean_total_loss - current_eval.mean_total_loss).abs() < 1e-6);
    let wrong = TrainConfig { seed: 999, ..two };
    assert!(
        train::<B>(
            &dataset,
            &tmp.path().join("wrong"),
            &wrong,
            Some(&first.checkpoint),
            &device
        )
        .is_err()
    );
}

#[test]
fn pilot_batches_resume_with_identical_fixed_probe_losses() {
    let _rng = TRAINING_RNG.lock().unwrap();
    use burn_gekko_train::pilot::{PilotConfig, run_pilot};
    type B = Autodiff<NdArray<f32>>;
    let tmp = temp();
    let dataset = fixture(tmp.path());
    let device = Default::default();
    let one = PilotConfig {
        training: TrainConfig {
            steps: 1,
            ..Default::default()
        },
        batch_size: 2,
        cache_full_views: true,
        fixed_example: false,
        train_rooms: 2,
        eval_every: 1,
        profile_phases: true,
        max_seconds: 60,
        schedule: Some(burn_gekko_train::pilot::LearningRateSchedule {
            warmup_steps: 1,
            decay_steps: 4,
            min_lr_ratio: 0.1,
        }),
        shuffle_samples: true,
    };
    let first =
        run_pilot::<B>(&dataset, &tmp.path().join("pilot-one"), &one, None, &device).unwrap();
    assert_eq!(first.completed_steps, 1);
    assert!(tmp.path().join("pilot-one/config.toml").is_file());
    assert!(tmp.path().join("pilot-one/pilot-config.toml").is_file());
    let mut two = one;
    two.training.steps = 2;
    let checkpoint = tmp.path().join("pilot-one/checkpoint-000001");
    let resumed = run_pilot::<B>(
        &dataset,
        &tmp.path().join("pilot-resume"),
        &two,
        Some(&checkpoint),
        &device,
    )
    .unwrap();
    let uninterrupted = run_pilot::<B>(
        &dataset,
        &tmp.path().join("pilot-full"),
        &two,
        None,
        &device,
    )
    .unwrap();
    assert!(
        (resumed.probes.last().unwrap().train.total
            - uninterrupted.probes.last().unwrap().train.total)
            .abs()
            < 1e-6
    );
    assert!(
        (resumed.probes.last().unwrap().validation.total
            - uninterrupted.probes.last().unwrap().validation.total)
            .abs()
            < 1e-6
    );
    assert_eq!(resumed.steps[0].samples, uninterrupted.steps[1].samples);
    assert_eq!(
        resumed.steps[0].learning_rate,
        uninterrupted.steps[1].learning_rate
    );
    assert_eq!(
        uninterrupted
            .steps
            .iter()
            .flat_map(|s| s.samples.iter())
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        4
    );
}

#[test]
fn old_generator_cache_keeps_its_original_identity() {
    let tmp = temp();
    let dir = fixture(tmp.path());
    let mut manifest = open_dataset(&dir).unwrap();
    manifest.generator = burn_gekko_data::LEGACY_GENERATOR.into();
    manifest.dataset_id = fingerprint(&(
        SCHEMA,
        &manifest.generator,
        &manifest.binary_sha256,
        &manifest.config,
    ))
    .unwrap();
    write_json(&dir.join("manifest.json"), &manifest).unwrap();
    assert_eq!(
        open_dataset(&dir).unwrap().generator,
        burn_gekko_data::LEGACY_GENERATOR
    );
    manifest.generator = burn_gekko_data::PILOT02_GENERATOR.into();
    manifest.dataset_id = fingerprint(&(
        SCHEMA,
        &manifest.generator,
        &manifest.binary_sha256,
        &manifest.config,
    ))
    .unwrap();
    write_json(&dir.join("manifest.json"), &manifest).unwrap();
    assert_eq!(
        open_dataset(&dir).unwrap().generator,
        burn_gekko_data::PILOT02_GENERATOR
    );
}

#[test]
fn unclamped_position_v2_preserves_world_geometry_and_requires_metadata() {
    let tmp = temp();
    let path = tmp.path().join("outside.safetensors");
    chunk(&path, 5);
    let original = fs::read(&path).unwrap();
    let parsed = safetensors::SafeTensors::deserialize(&original).unwrap();
    let mut tensors: Vec<_> = parsed
        .tensors()
        .into_iter()
        .map(|(name, t)| (name, t.dtype(), t.shape().to_vec(), t.data().to_vec()))
        .collect();
    // Tighten the crop in Z. The same visible plane at world Z=-2 is now outside it.
    for (name, _, _, data) in &mut tensors {
        if name == "aabb" {
            *data = [-4.0f32, -4.0, -1.0, 4.0, 4.0, 1.0]
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect();
        } else if name == "position" {
            for xyz in data.as_chunks_mut::<12>().0 {
                xyz[8..12].copy_from_slice(&(-0.5f32).to_le_bytes());
            }
        }
    }
    let write = |tensors: &[(String, Dtype, Vec<usize>, Vec<u8>)]| {
        let views: Vec<_> = tensors
            .iter()
            .map(|(name, dtype, shape, data)| {
                (
                    name.as_str(),
                    TensorView::new(*dtype, shape.clone(), data).unwrap(),
                )
            })
            .collect();
        fs::write(&path, safetensors::serialize(views, None).unwrap()).unwrap();
    };
    write(&tensors);
    assert!(burn_gekko_data::load_geometry(&path).is_err());
    let metadata = br#"{"capture_engine":"capture-v28;position=2;bounds=2"}"#.to_vec();
    tensors.push((
        "indoor_render_metadata_0".into(),
        Dtype::U8,
        vec![metadata.len()],
        metadata,
    ));
    write(&tensors);
    let geometry = burn_gekko_data::load_geometry(&path).unwrap();
    for view in 0..2 {
        for pixel in 0..32 * 32 {
            let position = &geometry.position[view][pixel * 3..pixel * 3 + 3];
            assert_eq!(position[2], -2.0);
            let projected = burn_gekko_data::project(
                position.try_into().unwrap(),
                &geometry.world_from_view[view],
                geometry.fovy[view],
                32,
                32,
            )
            .unwrap();
            assert!((projected[0] - (pixel % 32) as f32 - 0.5).abs() < 1e-4);
            assert!((projected[1] - (pixel / 32) as f32 - 0.5).abs() < 1e-4);
            assert_eq!(projected[2], geometry.depth[view][pixel]);
        }
    }
}

#[cfg(unix)]
#[test]
fn explicit_capture_recovery_checks_identity_and_lock_without_rerendering() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = temp();
    let source = fixture(tmp.path());
    let script = tmp.path().join("capture.sh");
    fs::write(&script, format!(
        "#!/bin/sh\nset -eu\nif [ \"$1\" = '--identity' ]; then printf '%s\\n' '{}'; exit 0; fi\ncp '{}'/raw/* \"$4\"/\nexit 1\n",
        GENERATOR, source.display()
    )).unwrap();
    fs::set_permissions(&script, fs::Permissions::from_mode(0o755)).unwrap();
    let config = CaptureConfig {
        width: 32,
        height: 32,
        ..Default::default()
    };
    let data_root = tmp.path().join("cache");
    assert!(generate(&data_root, &script, &config).is_err());
    let stage = fs::read_dir(data_root.join("datasets"))
        .unwrap()
        .next()
        .unwrap()
        .unwrap()
        .path();
    assert!(burn_gekko_data::recover_capture(&data_root, Path::new("/bin/true"), &stage).is_err());
    let id = fingerprint(&(SCHEMA, GENERATOR, sha256_file(&script).unwrap(), &config)).unwrap();
    let lock = data_root.join("datasets").join(format!("{id}.lock"));
    fs::write(&lock, b"busy").unwrap();
    assert!(burn_gekko_data::recover_capture(&data_root, &script, &stage).is_err());
    fs::remove_file(lock).unwrap();
    // Recovery also accepts the unchanged snapshots written before the TOML migration.
    write_json(&stage.join("capture.json"), &config).unwrap();
    fs::remove_file(stage.join("capture.toml")).unwrap();
    let final_dir = burn_gekko_data::recover_capture(&data_root, &script, &stage).unwrap();
    assert_eq!(open_dataset(&final_dir).unwrap().dataset_id, id);
    assert!(final_dir.join("recovery.json").is_file());
    assert!(!stage.exists());
}
