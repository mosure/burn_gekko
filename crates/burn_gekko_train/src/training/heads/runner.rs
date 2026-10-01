use super::{
    HeadTrainConfig,
    data::{TrainingTensors, load},
    evaluation,
};
use anyhow::{Result, ensure};
use burn::{
    module::{AutodiffModule, Module},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::{Int, Tensor, TensorData, backend::AutodiffBackend},
};
use burn_gekko::{
    heads::{
        calibration::{CalibrationHead, calibration_loss},
        reconstruction::RgbReconstructionHead,
    },
    tensor::scalar,
};
use burn_gekko_data::{Split, head_cache::HeadCache, sha256_file, write_config, write_json};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde_json::{Value, json};
use std::{fs, io::Write, path::Path, time::Instant};

/// `stop_after` is a bounded interruption for resume qualification, not checkpoint selection.
pub fn run<B: AutodiffBackend>(
    c: &HeadTrainConfig,
    run: &Path,
    resume: Option<&Path>,
    stop_after: Option<usize>,
    device: &B::Device,
) -> Result<()> {
    c.validate()?;
    ensure!(!run.exists(), "head run exists");
    ensure!(
        fs::canonicalize(run.ancestors().find(|p| p.exists()).unwrap())?
            .starts_with(fs::canonicalize(".data")?),
        "head run outside .data"
    );
    let wall = Instant::now();
    let cache = HeadCache::load(&c.cache, &c.cache_sha256)?;
    ensure!(
        cache.checkpoint_sha256 == c.checkpoint_sha256,
        "head foundation checkpoint mismatch"
    );
    let samples = load(&c.cache, &cache)?;
    let resident = TrainingTensors::<B>::new(&samples, &cache, device);
    B::seed(device, c.seed);
    let mut camera = CalibrationHead::<B>::new(cache.camera_width, c.width, device);
    let mut rgb = RgbReconstructionHead::<B>::new(cache.latent_width, c.width, device);
    let mut camera_opt = AdamWConfig::new().with_weight_decay(c.weight_decay).init();
    let mut rgb_opt = AdamWConfig::new().with_weight_decay(c.weight_decay).init();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    let identity = c.identity()?;
    let mut step = 0;
    if let Some(path) = resume {
        let meta: Value = serde_json::from_slice(&fs::read(path.join("metadata.json"))?)?;
        ensure!(
            meta["identity"] == identity,
            "head resume configuration/source changed"
        );
        for name in ["camera", "rgb", "camera-optimizer", "rgb-optimizer"] {
            ensure!(
                meta["files"][name] == sha256_file(&path.join(format!("{name}.mpk")))?,
                "head resume checksum mismatch"
            );
        }
        camera = camera.load_record(recorder.load(path.join("camera"), device)?);
        rgb = rgb.load_record(recorder.load(path.join("rgb"), device)?);
        camera_opt = camera_opt.load_record(recorder.load(path.join("camera-optimizer"), device)?);
        rgb_opt = rgb_opt.load_record(recorder.load(path.join("rgb-optimizer"), device)?);
        step = meta["completed_steps"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("missing resume step"))? as usize;
    }
    ensure!(
        step < c.steps && stop_after.is_none_or(|n| n > step && n <= c.steps),
        "no head updates requested"
    );
    fs::create_dir_all(run)?;
    write_config(&run.join("config.toml"), c)?;
    write_json(
        &run.join("provenance.json"),
        &json!({"identity":identity,"source_sha256":crate::provenance::identity()?,"checkpoint_sha256":c.checkpoint_sha256,"cache_sha256":c.cache_sha256,"frozen_foundation":true,"encoder_gradients":false,"noncommercial_weight_dependencies":[],"input_contract":cache.input_contract,"camera_frame":cache.camera_frame,"backend":std::any::type_name::<B>(),"resume":resume,"training_room_seeds":cache.samples.iter().filter(|r|r.split==Split::Train).map(|r|r.room_seed).collect::<std::collections::BTreeSet<_>>(),"validation_room_seeds":cache.samples.iter().filter(|r|r.split==Split::Validation).map(|r|r.room_seed).collect::<std::collections::BTreeSet<_>>()}),
    )?;
    let evaluate = |camera: &CalibrationHead<B>, rgb: &RgbReconstructionHead<B>, split, out| {
        evaluation::evaluate(
            &camera.valid(),
            &rgb.valid(),
            &samples,
            &cache,
            split,
            out,
            device,
        )
    };
    let initial_train = evaluate(&camera, &rgb, Split::Train, None)?;
    let initial = evaluate(&camera, &rgb, Split::Validation, None)?;
    let constant = evaluation::constant_camera(&samples)?;
    write_json(
        &run.join("initial.json"),
        &json!({"training":initial_train,"validation":initial,"constant_camera_validation":constant}),
    )?;
    let mut history = fs::File::create(run.join("metrics.jsonl"))?;
    let start = step;
    let mut max_camera_gradient = 0f64;
    let mut max_rgb_gradient = 0f64;
    let initial_camera = camera.output.weight.val().inner();
    let initial_rgb = rgb.output.weight.val().inner();
    let mut probes = Vec::new();
    let training_wall = Instant::now();
    while step < c.steps
        && stop_after.is_none_or(|n| step < n)
        && wall.elapsed().as_secs() < c.max_seconds
        && !run.join("STOP").exists()
    {
        let mut rng =
            ChaCha8Rng::seed_from_u64(c.seed ^ (step as u64).wrapping_mul(0x9e3779b97f4a7c15));
        let rgb_ids: Vec<i64> = (0..c.rgb_batch)
            .map(|_| {
                resident.hidden_indices[rng.random_range(0..resident.hidden_indices.len())] as i64
            })
            .collect();
        let camera_ids: Vec<i64> = (0..c.camera_batch)
            .map(|_| rng.random_range(0..resident.train_indices.len()) as i64)
            .collect();
        let indices = |v: Vec<i64>| {
            let n = v.len();
            Tensor::<B, 1, Int>::from_data(TensorData::new(v, [n]), device)
        };
        let r = indices(rgb_ids.clone());
        let k = indices(camera_ids.clone());
        let warm = ((step + 1) as f64 / c.warmup_steps as f64).min(1.);
        let lr = c.learning_rate
            * warm
            * (0.1
                + 0.9 * 0.5 * (1. + (std::f64::consts::PI * step as f64 / c.steps as f64).cos()));
        let camera_loss = calibration_loss(
            camera.forward(resident.camera.clone().select(0, k.clone())),
            resident.labels.clone().select(0, k.clone()),
            resident.translation_valid.clone().select(0, k),
        );
        let camera_value = scalar(camera_loss.clone())?;
        let mut gradients = GradientsParams::from_grads(camera_loss.backward(), &camera);
        let camera_norm = crate::train::clip(&camera, &mut gradients, 1.)?;
        camera = camera_opt.step(lr, camera, gradients);
        let features = Tensor::cat(
            vec![
                resident.cross.clone().select(0, r.clone()),
                resident.mono.clone().select(0, r.clone()),
            ],
            0,
        );
        let y = resident.rgb.clone().select(0, r);
        let y = Tensor::cat(vec![y.clone(), y], 0);
        let loss = (rgb.forward(features) - y.detach()).powf_scalar(2.).mean();
        let rgb_value = scalar(loss.clone())?;
        let mut gradients = GradientsParams::from_grads(loss.backward(), &rgb);
        let rgb_norm = crate::train::clip(&rgb, &mut gradients, 1.)?;
        rgb = rgb_opt.step(lr, rgb, gradients);
        step += 1;
        max_camera_gradient = max_camera_gradient.max(camera_norm);
        max_rgb_gradient = max_rgb_gradient.max(rgb_norm);
        writeln!(
            history,
            "{}",
            json!({"step":step,"camera_loss":camera_value,"rgb_loss":rgb_value,"camera_gradient_norm":camera_norm,"rgb_gradient_norm":rgb_norm,"learning_rate":lr,"camera_samples":camera_ids,"rgb_hidden_indices":rgb_ids})
        )?;
        if step.is_multiple_of(c.eval_every) {
            let scores = evaluate(&camera, &rgb, Split::Validation, None)?;
            eprintln!(
                "head step {step}/{}: RGB PSNR {:?}, rotation {:.2} degrees",
                c.steps, scores.rgb_hidden_psnr_db, scores.rotation_mean_degrees
            );
            probes.push(json!({"step":step,"validation":scores}));
        }
    }
    let train_seconds = training_wall.elapsed().as_secs_f64();
    ensure!(step > start, "no head training completed");
    let camera_delta = scalar(
        (camera.output.weight.val().inner() - initial_camera)
            .abs()
            .max(),
    )?;
    let rgb_delta = scalar((rgb.output.weight.val().inner() - initial_rgb).abs().max())?;
    ensure!(
        camera_delta > 0. && rgb_delta > 0.,
        "head parameters did not update"
    );
    let final_dir = run.join("final");
    fs::create_dir(&final_dir)?;
    camera
        .clone()
        .save_file(final_dir.join("camera"), &recorder)?;
    rgb.clone().save_file(final_dir.join("rgb"), &recorder)?;
    recorder.record(camera_opt.to_record(), final_dir.join("camera-optimizer"))?;
    recorder.record(rgb_opt.to_record(), final_dir.join("rgb-optimizer"))?;
    let files: [(&str, String); 4] =
        ["camera", "rgb", "camera-optimizer", "rgb-optimizer"].map(|name| {
            (
                name,
                sha256_file(&final_dir.join(format!("{name}.mpk"))).unwrap(),
            )
        });
    write_json(
        &final_dir.join("metadata.json"),
        &json!({"schema":1,"identity":identity,"checkpoint_sha256":c.checkpoint_sha256,"completed_steps":step,"files":files.into_iter().collect::<std::collections::BTreeMap<_,_>>()}),
    )?;
    let trained = evaluate(&camera, &rgb, Split::Train, None)?;
    let validation_dir = run.join("validation");
    let validation = evaluate(&camera, &rgb, Split::Validation, Some(&validation_dir))?;
    // Read the just-saved weights back through the production loader and compare every output metric.
    let reloaded_camera = CalibrationHead::<B>::new(cache.camera_width, c.width, device)
        .load_record(recorder.load(final_dir.join("camera"), device)?);
    let reloaded_rgb = RgbReconstructionHead::<B>::new(cache.latent_width, c.width, device)
        .load_record(recorder.load(final_dir.join("rgb"), device)?);
    let replay = evaluate(&reloaded_camera, &reloaded_rgb, Split::Validation, None)?;
    ensure!(
        serde_json::to_value(&validation)? == serde_json::to_value(&replay)?,
        "head checkpoint replay changed outputs"
    );
    let gates = json!({"completed_schedule":step==c.steps,"finite_losses_gradients":true,"camera_parameters_updated":camera_delta>0.,"rgb_parameters_updated":rgb_delta>0.,"camera_training_loss_decreased":trained.camera_regression_loss<initial_train.camera_regression_loss,"rgb_training_mse_decreased":trained.rgb_hidden_mse<initial_train.rgb_hidden_mse,"final_rotation_valid":validation.invalid_rotations==0,"no_focal_clamps":validation.focal_clamps==0,"checkpoint_replay_exact":true});
    write_json(
        &run.join("report.json"),
        &json!({"schema":1,"checkpoint_sha256":c.checkpoint_sha256,"completed_steps":step,"starting_step":start,"seconds":wall.elapsed().as_secs_f64(),"training_seconds":train_seconds,"updates_per_second":(step-start) as f64/train_seconds,"max_camera_gradient_norm":max_camera_gradient,"max_rgb_gradient_norm":max_rgb_gradient,"camera_weight_delta":camera_delta,"rgb_weight_delta":rgb_delta,"initial_training":initial_train,"initial_validation":initial,"training":trained,"validation":validation,"constant_camera_validation":constant,"probes":probes,"stability_gates":gates,"stop_reason":if step==c.steps{"step_limit"}else{"bounded_interruption"},"scope":"Fixed-schedule output-head fitting on a frozen foundation. Validation is development data; stability does not establish geometric accuracy, sharp RGB or SOTA."}),
    )?;
    let mut evidence = serde_json::Map::new();
    for (key, path) in [
        ("report", run.join("report.json")),
        ("provenance", run.join("provenance.json")),
        ("config", run.join("config.toml")),
        ("metadata", final_dir.join("metadata.json")),
        ("camera_weights", final_dir.join("camera.mpk")),
        ("rgb_weights", final_dir.join("rgb.mpk")),
        ("predictions", run.join("validation/predictions.json")),
        ("cache_manifest", c.cache.join("manifest.json")),
        ("steps", run.join("metrics.jsonl")),
    ] {
        evidence.insert(
            key.into(),
            json!({"path":path,"sha256":sha256_file(&path)?}),
        );
    }
    write_json(
        &run.join("evidence.json"),
        &json!({"schema":1,"checkpoint_sha256":c.checkpoint_sha256,"files":evidence}),
    )?;
    Ok(())
}
