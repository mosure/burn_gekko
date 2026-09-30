//! RGB training loop and optimizer/checkpoint transitions.
use super::*;
pub fn run<B: AutodiffBackend>(
    c: &E2eConfig,
    run: &Path,
    split: Option<Split>,
    resume: Option<&Path>,
    unrelated: bool,
    device: &B::Device,
) -> Result<()> {
    run_with_warm_start::<B>(c, run, split, resume, None, unrelated, device)
}

/// An explicit weights-only fine-tune permits a new dataset/objective while
/// recording the parent checkpoint. Exact continuation remains `resume`.
#[allow(clippy::too_many_arguments)]
pub fn run_with_warm_start<B: AutodiffBackend>(
    c: &E2eConfig,
    run: &Path,
    split: Option<Split>,
    resume: Option<&Path>,
    warm_start: Option<&Path>,
    unrelated: bool,
    device: &B::Device,
) -> Result<()> {
    c.validate()?;
    ensure!(
        warm_start.is_none() || (resume.is_none() && split.is_none()),
        "weights-only warm start cannot be combined with continuation or evaluation"
    );
    let wall = Instant::now();
    ensure!(!run.exists(), "choose a fresh run directory");
    let ancestor = run.ancestors().find(|p| p.exists()).unwrap();
    ensure!(
        fs::canonicalize(ancestor)?.starts_with(fs::canonicalize(".data")?),
        "outputs must be in .data"
    );
    ensure!(
        split.is_none() || resume.is_some(),
        "evaluation requires a checkpoint"
    );
    fs::create_dir_all(run)?;
    write_config(&run.join("config.toml"), c)?;
    let manifest = open_dataset(&c.dataset)?;
    ensure!(
        manifest.config.cameras > c.references
            && manifest.config.width <= 512
            && manifest.config.height <= 512,
        "invalid dataset bounds"
    );
    let (mut model, encoder_id) = initialize::<B>(c, device)?;
    let identity = c.identity()?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    let mut enc_opt = AdamWConfig::new()
        .with_weight_decay(c.weight_decay)
        .init::<B, VJepaEncoder<B>>();
    let mut dec_opt = AdamWConfig::new()
        .with_weight_decay(c.weight_decay)
        .init::<B, GekkoDecoder<B>>();
    let mut restored = None;
    let mut lineage = None;
    let mut warm_stage = None;
    let mut transport_initialization = if c.appearance_transport {
        "random"
    } else {
        "disabled"
    };
    let mut matching_initialization = if c.transport_matching {
        "random"
    } else {
        "disabled"
    };
    if let Some(path) = warm_start {
        let metadata: Snapshot = serde_json::from_slice(&fs::read(path.join("metadata.json"))?)?;
        let source: E2eConfig = burn_gekko_data::read_config(
            &path
                .parent()
                .ok_or_else(|| anyhow::anyhow!("checkpoint has no parent run"))?
                .join("config.toml"),
        )?;
        source.validate()?;
        ensure!(
            metadata.noncommercial_weight_dependencies.is_empty()
                && metadata.encoder_id == encoder_id,
            "warm-start provenance mismatch or noncommercial dependency"
        );
        ensure!(metadata.gate.stage <= 2, "invalid warm-start encoder stage");
        ensure!(
            source.decoder_width == c.decoder_width
                && source.decoder_depth == c.decoder_depth
                && source.decoder_heads == c.decoder_heads
                && source.decoder_qk_norm == c.decoder_qk_norm
                && source.appearance_bypass == c.appearance_bypass
                && source.rgb_head == c.rgb_head,
            "warm-start architecture mismatch"
        );
        ensure!(
            !source.appearance_transport
                || (c.appearance_transport
                    && source.transport_max_displacement == c.transport_max_displacement),
            "cannot silently remove or rescale a trained appearance head"
        );
        ensure!(
            !source.transport_matching || c.transport_matching,
            "cannot silently remove a trained RGB matcher"
        );
        ensure!(
            sha256_file(&path.join("model.mpk"))? == metadata.model_sha256,
            "warm-start model checksum mismatch"
        );
        model = model.load_file(path.join("model"), &recorder, device)?;
        if source.appearance_transport {
            transport_initialization = "own_checkpoint_weights";
        }
        if source.transport_matching {
            matching_initialization = "own_checkpoint_weights";
        }
        if c.appearance_transport && !source.appearance_transport {
            B::seed(device, c.seed.wrapping_add(9));
            model.decoder.appearance_head = Some(
                crate::appearance::AppearanceHead::new(
                    c.decoder_width,
                    c.transport_max_displacement,
                    c.transport_loss_weight,
                    device,
                )
                .with_pyramid_loss(c.transport_pyramid_loss)
                .with_coarse_smoothness(c.transport_coarse_smoothness_weight),
            );
            model = model.clone().load_record(model.into_record());
        }
        if c.transport_matching && !source.transport_matching {
            B::seed(device, c.seed.wrapping_add(17));
            model.decoder.appearance_head.as_mut().unwrap().matcher =
                Some(crate::matching::RgbMatcher::new(device));
            model = model.clone().load_record(model.into_record());
        }
        warm_stage = Some(metadata.gate.stage);
        lineage = Some(WarmStart {
            checkpoint: path.to_path_buf(),
            model_sha256: metadata.model_sha256,
            source_metadata_sha256: sha256_file(&path.join("metadata.json"))?,
            source_identity: metadata.identity,
            source_dataset_id: metadata.dataset_id,
            source_completed_steps: metadata.completed_steps,
            optimizer_reset: true,
        });
    }
    if let Some(path) = resume {
        if c.transport_matching {
            matching_initialization = "resumed";
        }
        if c.appearance_transport {
            transport_initialization = "resumed";
        }
        let metadata: Snapshot = serde_json::from_slice(&fs::read(path.join("metadata.json"))?)?;
        ensure!(
            metadata.noncommercial_weight_dependencies.is_empty()
                && metadata.encoder_id == encoder_id
                && metadata.identity == identity,
            "checkpoint provenance/configuration mismatch"
        );
        ensure!(
            metadata.backend == std::any::type_name::<B>(),
            "checkpoint backend mismatch"
        );
        ensure!(metadata.gate.stage <= 2, "invalid checkpoint encoder stage");
        ensure!(
            metadata.dataset_id == manifest.dataset_id,
            "dataset mismatch"
        );
        ensure!(
            sha256_file(&path.join("model.mpk"))? == metadata.model_sha256,
            "checkpoint checksum mismatch"
        );
        model = model.load_file(path.join("model"), &recorder, device)?;
        if split.is_none() {
            ensure!(
                sha256_file(&path.join("encoder-optimizer.mpk"))?
                    == metadata.encoder_optimizer_sha256
                    && sha256_file(&path.join("decoder-optimizer.mpk"))?
                        == metadata.decoder_optimizer_sha256,
                "optimizer checksum mismatch"
            );
            enc_opt = enc_opt.load_record(recorder.load(path.join("encoder-optimizer"), device)?);
            dec_opt = dec_opt.load_record(recorder.load(path.join("decoder-optimizer"), device)?);
        }
        lineage = metadata.warm_start.clone();
        restored = Some(metadata);
    }
    if let Some(head) = model.decoder.appearance_head.as_mut() {
        // Loss settings belong to the destination config, including an explicit
        // own-checkpoint transfer that resets optimizers and changes its loss.
        head.pyramid_loss = c.transport_pyramid_loss;
        head.auxiliary_weight = c.transport_loss_weight;
        head.coarse_smoothness_weight = c.transport_coarse_smoothness_weight;
    }
    let cache_bytes = c.train_rooms as u64
        * manifest.config.cameras as u64
        * manifest.config.width as u64
        * manifest.config.height as u64
        * 3
        * 4;
    let heldout_rooms = manifest
        .scenes
        .iter()
        .filter(|e| e.split == split.unwrap_or(Split::Validation))
        .count();
    let heldout_bytes = heldout_rooms as u64
        * manifest.config.cameras as u64
        * manifest.config.width as u64
        * manifest.config.height as u64
        * 3
        * 4;
    let device_cache_bytes = heldout_bytes
        + if split.is_none() && c.rgb_cache == RgbCache::Device {
            cache_bytes
        } else {
            0
        };
    ensure!(
        device_cache_bytes <= c.rgb_cache_max_mib as u64 * 1024 * 1024,
        "resident evaluation RGB exceeds configured memory ceiling"
    );
    ensure!(
        split.is_some() || cache_bytes <= c.rgb_cache_max_mib as u64 * 1024 * 1024,
        "RGB cache exceeds configured memory ceiling"
    );
    let mut training = if c.rgb_cache == RgbCache::Host {
        TrainingScenes::Host(Vec::new())
    } else {
        TrainingScenes::Device(Vec::new())
    };
    let mut validation = Vec::new();
    for entry in &manifest.scenes {
        let wanted = if let Some(s) = split {
            entry.split == s
        } else {
            entry.split == Split::Validation
                || (entry.split == Split::Train && training.len() < c.train_rooms)
        };
        if !wanted {
            continue;
        }
        let scene = load_rgb(&c.dataset.join("raw").join(&entry.file))?;
        if split.is_none()
            && entry.split == Split::Train
            && let TrainingScenes::Host(scenes) = &mut training
        {
            scenes.push(scene);
            continue;
        }
        let resident = Scene {
            seed: scene.seed,
            rgb: (0..scene.views.len())
                .map(|v| image_tensor::<B::InnerBackend>(&scene, v, device))
                .collect(),
        };
        if split.is_none() && entry.split == Split::Train {
            if let TrainingScenes::Device(scenes) = &mut training {
                scenes.push(resident);
            }
        } else {
            validation.push(resident);
        }
    }
    write_json(
        &run.join("provenance.json"),
        &serde_json::json!({"dataset_id":manifest.dataset_id,"encoder_id":encoder_id,"identity":identity,"backend":std::any::type_name::<B>(),"initialization":c.initialization,"fusion_initialization":if lineage.is_some(){"own_checkpoint_weights"}else{"random"},"appearance_transport":c.appearance_transport,"transport_initialization":transport_initialization,"matching_initialization":matching_initialization,"warm_start":lineage,"noncommercial_weight_dependencies":[],"teacher":null,"synthetic_reference_probability":c.synthetic_reference_probability,"cached_feature_bytes":0,"rgb_cache":c.rgb_cache,"rgb_cache_bytes":cache_bytes,"encoder_license":if matches!(c.initialization,Initialization::Scratch{..}) {"no pretrained weights"}else{"MIT"},"resume":resume,"evaluation_split":split}),
    )?;
    if split.is_some() {
        let mse = evaluate(
            &model.valid(),
            &validation,
            c,
            &run.join("samples"),
            validation.len(),
            unrelated,
            true,
        )?;
        write_json(
            &run.join("report.json"),
            &serde_json::json!({"mean_hidden_rgb_mse":mse,"seconds":wall.elapsed().as_secs_f64()}),
        )?;
        return Ok(());
    }
    ensure!(
        training.len() == c.train_rooms && !validation.is_empty(),
        "insufficient scenes"
    );
    let train_probe: Vec<_> = (0..training.len().min(c.probe_rooms))
        .map(|s| {
            let (target, refs) = training.batch(&[(s, 0)], manifest.config.cameras - 1, device);
            Scene {
                seed: training.seed(s),
                rgb: std::iter::once(target).chain(refs).collect(),
            }
        })
        .collect();
    let initial = evaluate(
        &model.valid(),
        &validation,
        c,
        &run.join("step-initial"),
        c.probe_rooms,
        false,
        false,
    )?;
    let synthetic_probe: Vec<Scene<B::InnerBackend>> = if c.synthetic_reference_probability > 0. {
        validation
            .iter()
            .take(c.probe_rooms)
            .map(|scene| {
                let target = scene.rgb[0].clone();
                let (refs, _) = crate::curriculum::references(
                    target.clone(),
                    c.references,
                    scene.seed ^ 0x7afe,
                    0,
                );
                Scene {
                    seed: scene.seed,
                    rgb: std::iter::once(target).chain(refs).collect(),
                }
            })
            .collect()
    } else {
        Vec::new()
    };
    if !synthetic_probe.is_empty() {
        evaluate_synthetic(
            &model.valid(),
            &synthetic_probe,
            c,
            &run.join("synthetic-step-initial"),
        )?;
    }
    let (start, mut gate) = restored.map(|m| (m.completed_steps, m.gate)).unwrap_or((
        0,
        UnfreezeGate::new(
            initial,
            matches!(c.initialization, Initialization::Scratch { .. }),
        ),
    ));
    if let Some(stage) = warm_stage {
        gate.stage = stage;
    }
    ensure!(start < c.steps, "no additional steps requested");
    let stage_blocks = |stage: usize, depth: usize| match stage {
        0 => 0,
        1 => depth.min(2),
        _ => depth + 1,
    };
    let depth = model.encoder.blocks.len();
    model = model.train_encoder(stage_blocks(gate.stage, depth));
    let first_before = model.encoder.blocks[0].attn.qkv.weight.val().detach();
    let last_before = model.encoder.blocks[depth - 1]
        .attn
        .qkv
        .weight
        .val()
        .detach();
    let stem_before = model.encoder.image_patch_embed.proj.weight.val().detach();
    let appearance_before = model
        .decoder
        .appearance_head
        .as_ref()
        .map(|head| head.flow.weight.val().detach());
    let matcher_before = model
        .decoder
        .appearance_head
        .as_ref()
        .and_then(|head| head.matcher.as_ref())
        .map(|matcher| matcher.first.weight.val().detach());
    let mut sampler = SampleSchedule::new(training.len(), manifest.config.cameras, c.seed, true);
    let mut log = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(run.join("metrics.jsonl"))?;
    let mut probes = vec![serde_json::json!({"step":start,"mse":initial,"stage":gate.stage})];
    let mut completed = start;
    let mut stage_steps = [0usize; 3];
    let mut times = Vec::new();
    let prepare_seconds = wall.elapsed().as_secs_f64();
    for step in start..c.steps {
        if wall.elapsed().as_secs() >= c.max_seconds || run.join("STOP").exists() {
            break;
        }
        let tick = Instant::now();
        let samples: Vec<_> = (0..c.batch_size)
            .map(|j| sampler.sample(step * c.batch_size + j, c.fixed_example))
            .collect();
        let mask = visible_mask(
            manifest.config.width / 16 * (manifest.config.height / 16),
            c.mask_ratio,
            c.seed,
            if c.fixed_example { 0 } else { step },
        )?;
        let (rgb, mut refs) = training.batch(&samples, c.references, device);
        let synthetic =
            crate::curriculum::use_synthetic(c.seed, step, c.synthetic_reference_probability);
        let synthetic_warps = if synthetic {
            let (views, parameters) =
                crate::curriculum::references(rgb.clone(), c.references, c.seed, step);
            refs = views;
            Some(parameters)
        } else {
            None
        };
        let (loss, ri) = model.training_loss_with_ri(
            Tensor::from_inner(rgb),
            &refs.into_iter().map(Tensor::from_inner).collect::<Vec<_>>(),
            &mask,
            c.edge_loss_weight,
            c.gradient_energy_weight,
            if step >= c.ri_start_step {
                c.ri_weight
            } else {
                0.
            },
        )?;
        let value = scalar(loss.clone())?;
        let ri = scalar(ri)?;
        ensure!(
            value.is_finite() && ri.is_finite(),
            "nonfinite training loss at step {step}"
        );
        let mut grads = GradientsParams::from_grads(loss.backward(), &model);
        let norm = clip(&model, &mut grads, 1.)?;
        let enc_grads = take_gradients(&model.encoder, &mut grads);
        let encoder_gradient_tensors = enc_grads.len();
        ensure!(
            (gate.stage == 0) == enc_grads.is_empty(),
            "encoder gradient stage contract failed"
        );
        let lr = crate::pilot::LearningRateSchedule {
            warmup_steps: c.warmup_steps,
            decay_steps: c.decay_steps,
            min_lr_ratio: 0.1,
        }
        .learning_rate(c.learning_rate, step);
        if !enc_grads.is_empty() {
            model.encoder = enc_opt.step(lr * c.encoder_lr_ratio, model.encoder, enc_grads);
        }
        model.decoder = dec_opt.step(lr, model.decoder, grads);
        B::sync(device).map_err(|e| anyhow::anyhow!("{e}"))?;
        completed = step + 1;
        stage_steps[gate.stage] += 1;
        times.push(tick.elapsed().as_secs_f64());
        writeln!(
            log,
            "{}",
            serde_json::json!({"step":completed,"loss":value,"ri_loss":ri,"gradient_norm":norm,"learning_rate":lr,"encoder_learning_rate":if gate.stage==0{0.}else{lr*c.encoder_lr_ratio},"stage":gate.stage,"encoder_gradient_tensors":encoder_gradient_tensors,"seconds":times.last(),"synthetic_warps":synthetic_warps,"samples":samples.iter().map(|&(s,v)|(training.seed(s),v)).collect::<Vec<_>>()})
        )?;
        log.flush()?;
        if completed.is_multiple_of(50) {
            eprintln!(
                "e2e step {completed}/{} stage {} loss {value:.6} {:.1}ms",
                c.steps,
                gate.stage,
                times.last().unwrap() * 1000.
            );
        }
        if completed.is_multiple_of(c.eval_every) {
            let mse = evaluate(
                &model.valid(),
                &validation,
                c,
                &run.join(format!("step-{completed:06}")),
                c.probe_rooms,
                false,
                false,
            )?;
            if !synthetic_probe.is_empty() {
                evaluate_synthetic(
                    &model.valid(),
                    &synthetic_probe,
                    c,
                    &run.join(format!("synthetic-step-{completed:06}")),
                )?;
            }
            let transitioned = c.unfreeze
                && gate.observe(
                    completed,
                    mse,
                    c.unfreeze_min_steps,
                    c.unfreeze_min_improvement,
                );
            let train_mse = evaluate(
                &model.valid(),
                &train_probe,
                c,
                &run.join(format!("train-step-{completed:06}")),
                train_probe.len(),
                false,
                false,
            )?;
            let first_delta = scalar(
                (model.encoder.blocks[0].attn.qkv.weight.val().detach() - first_before.clone())
                    .abs()
                    .max(),
            )?;
            let last_delta = scalar(
                (model.encoder.blocks[depth - 1]
                    .attn
                    .qkv
                    .weight
                    .val()
                    .detach()
                    - last_before.clone())
                .abs()
                .max(),
            )?;
            let appearance_delta = appearance_before
                .as_ref()
                .zip(model.decoder.appearance_head.as_ref())
                .map(|(before, head)| {
                    scalar(
                        (head.flow.weight.val().detach() - before.clone())
                            .abs()
                            .max(),
                    )
                })
                .transpose()?;
            probes.push(serde_json::json!({"step":completed,"mse":mse,"training_mse":train_mse,"stage":gate.stage,"transitioned":transitioned,"first_encoder_delta":first_delta,"last_encoder_delta":last_delta,"appearance_head_delta":appearance_delta}));
            write_json(&run.join("probes.json"), &probes)?;
            eprintln!(
                "validation step {completed}: {mse:.6}; training probe {train_mse:.6}; stage {} transition {transitioned}",
                gate.stage
            );
            if transitioned {
                model = model.train_encoder(stage_blocks(gate.stage, depth));
            }
        }
        if c.checkpoint_every > 0 && completed.is_multiple_of(c.checkpoint_every) {
            checkpoint(
                &model,
                &enc_opt,
                &dec_opt,
                &run.join(format!("checkpoint-{completed:06}")),
                Snapshot {
                    identity: identity.clone(),
                    dataset_id: manifest.dataset_id.clone(),
                    encoder_id: encoder_id.clone(),
                    completed_steps: completed,
                    gate: gate.clone(),
                    model_sha256: String::new(),
                    encoder_optimizer_sha256: String::new(),
                    decoder_optimizer_sha256: String::new(),
                    noncommercial_weight_dependencies: Vec::new(),
                    backend: std::any::type_name::<B>().into(),
                    warm_start: lineage.clone(),
                },
            )?;
            write_json(
                &run.join("latest-checkpoint.json"),
                &serde_json::json!({"path":format!("checkpoint-{completed:06}"),"step":completed}),
            )?;
        }
    }
    ensure!(completed > start, "no training updates completed");
    let first_delta = scalar(
        (model.encoder.blocks[0].attn.qkv.weight.val().detach() - first_before)
            .abs()
            .max(),
    )?;
    let last_delta = scalar(
        (model.encoder.blocks[depth - 1]
            .attn
            .qkv
            .weight
            .val()
            .detach()
            - last_before)
            .abs()
            .max(),
    )?;
    let stem_delta = scalar(
        (model.encoder.image_patch_embed.proj.weight.val().detach() - stem_before)
            .abs()
            .max(),
    )?;
    ensure!(
        stage_steps[2] == 0 || (first_delta > 0. && stem_delta > 0.),
        "full encoder/stem did not update"
    );
    ensure!(
        stage_steps[1] + stage_steps[2] == 0 || last_delta > 0.,
        "encoder tail did not update"
    );
    ensure!(
        stage_steps[1] + stage_steps[2] > 0
            || (first_delta == 0. && last_delta == 0. && stem_delta == 0.),
        "frozen encoder changed"
    );
    let appearance_delta = appearance_before
        .as_ref()
        .zip(model.decoder.appearance_head.as_ref())
        .map(|(before, head)| {
            scalar(
                (head.flow.weight.val().detach() - before.clone())
                    .abs()
                    .max(),
            )
        })
        .transpose()?;
    ensure!(
        completed == start || !c.appearance_transport || appearance_delta.is_some_and(|x| x > 0.),
        "appearance head did not update"
    );
    let matcher_delta = matcher_before
        .as_ref()
        .zip(
            model
                .decoder
                .appearance_head
                .as_ref()
                .and_then(|h| h.matcher.as_ref()),
        )
        .map(|(before, matcher)| {
            scalar(
                (matcher.first.weight.val().detach() - before.clone())
                    .abs()
                    .max(),
            )
        })
        .transpose()?;
    ensure!(
        !c.transport_matching || matcher_delta.is_some_and(|x| x > 0.),
        "RGB correspondence matcher did not update"
    );
    checkpoint(
        &model,
        &enc_opt,
        &dec_opt,
        &run.join("final"),
        Snapshot {
            identity,
            dataset_id: manifest.dataset_id,
            encoder_id,
            completed_steps: completed,
            gate,
            model_sha256: String::new(),
            encoder_optimizer_sha256: String::new(),
            decoder_optimizer_sha256: String::new(),
            noncommercial_weight_dependencies: Vec::new(),
            backend: std::any::type_name::<B>().into(),
            warm_start: lineage,
        },
    )?;
    let mse = evaluate(
        &model.valid(),
        &validation,
        c,
        &run.join("validation"),
        validation.len(),
        false,
        true,
    )?;
    let wrong = evaluate(
        &model.valid(),
        &validation,
        c,
        &run.join("unrelated"),
        validation.len(),
        true,
        false,
    )?;
    let warm = &times[times.len().min(50)..];
    write_json(
        &run.join("report.json"),
        &serde_json::json!({"completed_steps":completed,"start_step":start,"seconds":wall.elapsed().as_secs_f64(),"prepare_seconds":prepare_seconds,"stage_steps":stage_steps,"first_encoder_max_abs_delta":first_delta,"last_encoder_max_abs_delta":last_delta,"image_stem_max_abs_delta":stem_delta,"appearance_head_max_abs_delta":appearance_delta,"matcher_max_abs_delta":matcher_delta,"validation_mse":mse,"unrelated_mse":wrong,"warm_targets_per_second":if warm.is_empty(){0.}else{warm.len() as f64*c.batch_size as f64/warm.iter().sum::<f64>()},"probes":probes,"cached_feature_bytes":0,"stop_reason":if completed==c.steps{"step_limit"}else if run.join("STOP").exists(){"stop_file"}else{"wall_limit"}}),
    )?;
    Ok(())
}

/// Mark derived image tasks so they cannot be mistaken for captured-room quality.
fn evaluate_synthetic<B: Backend>(
    model: &ReconstructionModel<B>,
    scenes: &[Scene<B>],
    c: &E2eConfig,
    output: &Path,
) -> Result<()> {
    evaluate(model, scenes, c, output, scenes.len(), false, false)?;
    for entry in fs::read_dir(output)? {
        let path = entry?.path();
        let path = if path.is_dir() {
            path.join("sample.json")
        } else {
            path
        };
        if path.extension().is_some_and(|e| e == "json") {
            let mut row: serde_json::Value = serde_json::from_slice(&fs::read(&path)?)?;
            row["diagnostic_only"] = serde_json::json!(true);
            row["synthetic_reference_augmentation"] = serde_json::json!("projective_rgb_v1");
            write_json(&path, &row)?;
        }
    }
    Ok(())
}
