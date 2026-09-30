//! Bounded latent studies with a fixed teacher and separately scheduled student.
use crate::{
    batch::SampleSchedule,
    e2e::{UnfreezeGate, take_gradients},
    e2e_pilot::host_batch,
    encoder::normalize,
    latent::{LatentFusion, LatentModel, latent_loss, normalize_teacher},
    latent_eval::{evaluate, values},
    masking::{MaskPattern, mask},
    model::DecoderConfig,
    train::{EncoderSource, clip, load_encoder, scalar},
};
use anyhow::{Result, ensure};
use burn::{
    module::{AutodiffModule, Module},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    record::{FullPrecisionSettings, NamedMpkFileRecorder, Recorder},
    tensor::{Tensor, backend::AutodiffBackend},
};
use burn_gekko_data::{
    RgbScene, SceneEntry, Split, fingerprint, load_dataset_rgb, read_dataset_manifest, sha256_file,
    write_config, write_json,
};
use burn_vjepa::VJepaEncoder;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};

mod config;
mod equivariance;
pub use config::{LatentConfig, WeightAncestor};
pub(crate) use config::{Snapshot, audited_checkpoint};

fn save_checkpoint<
    B: AutodiffBackend,
    E: Optimizer<VJepaEncoder<B>, B>,
    F: Optimizer<LatentFusion<B>, B>,
>(
    model: &LatentModel<B>,
    enc: &E,
    fusion: &F,
    path: &Path,
    mut meta: Snapshot,
) -> Result<()> {
    ensure!(!path.exists(), "checkpoint exists");
    let stage = path.with_extension("partial");
    ensure!(!stage.exists(), "partial checkpoint exists");
    fs::create_dir(&stage)?;
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    model.clone().save_file(stage.join("model"), &recorder)?;
    recorder.record(enc.to_record(), stage.join("encoder-optimizer"))?;
    recorder.record(fusion.to_record(), stage.join("fusion-optimizer"))?;
    meta.model_sha256 = sha256_file(&stage.join("model.mpk"))?;
    meta.encoder_optimizer_sha256 = sha256_file(&stage.join("encoder-optimizer.mpk"))?;
    meta.fusion_optimizer_sha256 = sha256_file(&stage.join("fusion-optimizer.mpk"))?;
    write_json(&stage.join("metadata.json"), &meta)?;
    if meta.teacher_id == crate::e2e_pilot::REVIEWED_VJEPA_ID {
        fs::write(
            stage.join("LICENSE.vjepa.txt"),
            burn_vjepa::provenance::WEIGHTS_LICENSE,
        )?;
        fs::write(
            stage.join("NOTICE.md"),
            "Student initialized from Meta V-JEPA 2.1 MIT image-encoder weights. Teacher is the fixed original package, not a released Gekko. Fusion, latent and RI heads initialized randomly. Retain LICENSE.vjepa.txt. Original weights: https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt\n",
        )?;
    }
    fs::rename(stage, path)?;
    Ok(())
}

pub fn run<B: AutodiffBackend>(
    c: &LatentConfig,
    run: &Path,
    resume: Option<&Path>,
    device: &B::Device,
) -> Result<()> {
    c.validate()?;
    ensure!(
        !(c.stable_attention && std::any::type_name::<B>().contains("Fusion")),
        "float64 stable attention is unsupported by Burn CUDA Fusion in full evaluation; use float32"
    );
    ensure!(!run.exists(), "choose a fresh output directory");
    let ancestor = run.ancestors().find(|p| p.exists()).unwrap();
    ensure!(
        fs::canonicalize(ancestor)?.starts_with(fs::canonicalize(".data")?),
        "output must be under .data"
    );
    let wall = Instant::now();
    fs::create_dir_all(run)?;
    write_config(&run.join("config.toml"), c)?;
    eprintln!("validating immutable dataset and teacher provenance");
    let manifest = read_dataset_manifest(&c.dataset)?;
    ensure!(
        manifest.config.cameras > c.references
            && manifest.config.width <= 512
            && manifest.config.height <= 512,
        "invalid dataset shape"
    );
    let load = |split, count| -> Result<(Vec<RgbScene>, Vec<SceneEntry>)> {
        let entries: Vec<_> = manifest
            .scenes
            .iter()
            .filter(|s| s.split == split)
            .take(count)
            .cloned()
            .collect();
        ensure!(entries.len() == count, "insufficient rooms");
        let scenes = entries
            .iter()
            .map(|e| load_dataset_rgb(&c.dataset, &manifest, e))
            .collect::<Result<Vec<_>>>()?;
        Ok((scenes, entries))
    };
    let (training, train_entries) = load(Split::Train, c.train_rooms)?;
    let (validation, val_entries) = load(Split::Validation, c.validation_rooms)?;
    let (teacher, ec, teacher_id) = load_encoder::<B::InnerBackend>(&c.teacher, c.seed, device)?;
    let affinity_layers: Vec<_> = c
        .fusion_auxiliary
        .teacher_layers
        .iter()
        .map(|x| x - 1)
        .collect();
    ensure!(
        affinity_layers
            .iter()
            .all(|x| ec.encoder.hierarchical_layers().contains(x)),
        "affinity targets must use trained hierarchical encoder outputs"
    );
    if !matches!(c.teacher, EncoderSource::DiagnosticTiny) {
        ensure!(
            teacher_id == crate::e2e_pilot::REVIEWED_VJEPA_ID,
            "unreviewed teacher package"
        );
    }
    // The teacher's backend has no autodiff tape and it is never handed to an optimizer.
    let teacher_first = teacher.blocks[0].attn.qkv.weight.val();
    let (encoder, student_config, student_id) = load_encoder::<B>(&c.teacher, c.seed, device)?;
    ensure!(
        student_id == teacher_id,
        "student/teacher initialization mismatch"
    );
    B::seed(device, c.seed.wrapping_add(31));
    let mut model = LatentModel::new(
        encoder,
        student_config,
        &DecoderConfig {
            encoder_dim: ec.encoder.embed_dim,
            width: c.decoder_width,
            depth: c.decoder_depth,
            heads: c.decoder_heads,
            patch: 16,
        },
        device,
    )?;
    let mut enc_opt = AdamWConfig::new()
        .with_weight_decay(c.weight_decay)
        .init::<B, VJepaEncoder<B>>();
    let mut fusion_opt = AdamWConfig::new()
        .with_weight_decay(c.weight_decay)
        .init::<B, LatentFusion<B>>();
    let identity = c.identity()?;
    let backend = std::any::type_name::<B>().to_string();
    let recorder = NamedMpkFileRecorder::<FullPrecisionSettings>::default();
    let mut restored = None;
    if resume.is_none()
        && let Some(parent) = &c.warm_start
    {
        audited_checkpoint(parent, &teacher_id)?;
        let pc: LatentConfig =
            burn_gekko_data::read_config(&parent.checkpoint.parent().unwrap().join("config.toml"))?;
        ensure!(
            pc.decoder_width == c.decoder_width
                && pc.decoder_depth == c.decoder_depth
                && pc.decoder_heads == c.decoder_heads,
            "warm-start architecture mismatch"
        );
        ensure!(
            pc.spatial_descriptor.is_none() || c.spatial_descriptor.is_some(),
            "cannot silently discard a trained spatial head"
        );
        model = model
            .prepare_spatial_descriptor(pc.spatial_descriptor.as_ref())?
            .load_checked_record(recorder.load(parent.checkpoint.join("model"), device)?)?;
    }
    if let Some(path) = resume {
        let m: Snapshot = serde_json::from_slice(&fs::read(path.join("metadata.json"))?)?;
        audited_checkpoint(
            &WeightAncestor {
                checkpoint: path.into(),
                model_sha256: m.model_sha256.clone(),
            },
            &teacher_id,
        )?;
        ensure!(
            m.identity == identity
                && m.dataset_id == manifest.dataset_id
                && m.teacher_id == teacher_id
                && m.backend == backend,
            "resume identity/backend mismatch"
        );
        ensure!(
            m.noncommercial_weight_dependencies.is_empty(),
            "noncommercial dependency prohibited"
        );
        ensure!(
            sha256_file(&path.join("model.mpk"))? == m.model_sha256
                && sha256_file(&path.join("encoder-optimizer.mpk"))? == m.encoder_optimizer_sha256
                && sha256_file(&path.join("fusion-optimizer.mpk"))? == m.fusion_optimizer_sha256,
            "checkpoint checksum mismatch"
        );
        model = model
            .prepare_spatial_descriptor(c.spatial_descriptor.as_ref())?
            .load_checked_record(recorder.load(path.join("model"), device)?)?;
        enc_opt = enc_opt.load_record(recorder.load(path.join("encoder-optimizer"), device)?);
        fusion_opt = fusion_opt.load_record(recorder.load(path.join("fusion-optimizer"), device)?);
        restored = Some(m);
    }
    model = model
        .with_spatial_input_layer(c.spatial_input_layer, resume.is_none())?
        .with_spatial_input_scale(c.spatial_input_scale)?;
    if model.fusion.spatial_descriptor.is_none() && c.spatial_descriptor.is_some() {
        model = model.prepare_spatial_descriptor(c.spatial_descriptor.as_ref())?;
    }
    if let (Some(head), Some(config)) =
        (&mut model.fusion.spatial_descriptor, &c.spatial_descriptor)
    {
        head.residual_radius = config.residual_radius;
    }
    model.validate_spatial_descriptor()?;
    // Numerical policy is not a learned parameter and must be set after loading.
    model.fusion.decoder = model
        .fusion
        .decoder
        .with_stable_attention(c.stable_attention)
        .with_cross_view_rope(c.cross_view_rope);
    // Always reload the same ancestor even on exact optimizer resume. The
    // affinity teacher must not silently advance to the resumed student.
    let affinity_encoder = if c.fusion_auxiliary.enabled() && c.fusion_auxiliary.anchor_warm_start {
        Some(
            crate::latent_assess::load_assessed_model::<B::InnerBackend>(
                c.warm_start.as_ref().unwrap(),
                device,
            )?
            .model
            .encoder,
        )
    } else {
        None
    };
    let grid = [manifest.config.height / 16, manifest.config.width / 16];
    // Position-conditioned constant predictor fitted exclusively on the first 16 training rooms.
    let mut train_mean =
        Tensor::<B::InnerBackend, 3>::zeros([1, grid[0] * grid[1], ec.encoder.embed_dim], device);
    let mean_rooms = training.len().min(16);
    for s in 0..mean_rooms {
        for v in 0..manifest.config.cameras {
            let (rgb, _) =
                host_batch::<B::InnerBackend>(&training, &[(s, v)], c.references, device);
            train_mean = train_mean
                + normalize_teacher(teacher.forward_image(normalize(rgb, &ec), None).tokens);
        }
    }
    train_mean = train_mean / (mean_rooms * manifest.config.cameras) as f64;
    fs::write(
        run.join("train-position-mean.f32"),
        values(train_mean.clone())?
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    let student_initialization = if resume.is_some() {
        "exact checkpoint resume; model, both optimizers and gate restored"
    } else if c.warm_start.is_some() {
        "audited own latent checkpoint; both optimizers and gate reset for new phase"
    } else {
        "same package as teacher; random fusion and prediction heads"
    };
    write_json(
        &run.join("provenance.json"),
        &serde_json::json!({"task":"fixed_vjepa21_latent_prediction","identity":identity,"dataset_id":manifest.dataset_id,"teacher_id":teacher_id,"backend":backend,"student_initialization":student_initialization,"warm_start":c.warm_start,"teacher_update":"never","dense_teacher_input":"loss only","feature_cache":"none; online teacher and student forwards","rgb_source":"immutable disk capture; selected RGB resident on host","noncommercial_weight_dependencies":[],"mean_baseline_training_rooms":mean_rooms,"resume":resume,"training_room_seeds":train_entries.iter().map(|e|e.seed).collect::<Vec<_>>(),"validation_room_seeds":val_entries.iter().map(|e|e.seed).collect::<Vec<_>>()}),
    )?;
    eprintln!("initial latent validation");
    let initial = evaluate(
        &model.valid(),
        &teacher,
        &validation,
        &val_entries,
        c,
        train_mean.clone(),
        &run.join("initial"),
        false,
        false,
    )?;
    let initial_mse = initial["mean_cross_mse"].as_f64().unwrap();
    let (start, mut gate) = restored
        .map(|m| (m.completed_steps, m.gate))
        .unwrap_or_else(|| {
            let mut gate = UnfreezeGate::new(initial_mse, false);
            gate.stage = c.initial_encoder_stage;
            (0, gate)
        });
    ensure!(start < c.steps, "no additional updates requested");
    let blocks = |stage, depth: usize| match stage {
        0 => 0,
        1 => depth.min(2),
        _ => depth + 1,
    };
    let depth = model.encoder.blocks.len();
    model = model.train_encoder(blocks(gate.stage, depth));
    let first_before = model.encoder.blocks[0].attn.qkv.weight.val().detach();
    let last_before = model.encoder.blocks[depth - 1]
        .attn
        .qkv
        .weight
        .val()
        .detach();
    let head_before = model.fusion.prediction.weight.val().detach();
    let mut sampler = SampleSchedule::new(training.len(), manifest.config.cameras, c.seed, true);
    let mut log = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(run.join("metrics.jsonl"))?;
    let mut probes =
        vec![serde_json::json!({"step":start,"cross_mse":initial_mse,"stage":gate.stage})];
    let mut completed = start;
    let mut times = Vec::new();
    let mut stage_steps = [0usize; 3];
    let prepare_seconds = wall.elapsed().as_secs_f64();
    let metadata = |step, gate: UnfreezeGate| Snapshot {
        identity: identity.clone(),
        dataset_id: manifest.dataset_id.clone(),
        teacher_id: teacher_id.clone(),
        completed_steps: step,
        gate,
        backend: backend.clone(),
        model_sha256: String::new(),
        encoder_optimizer_sha256: String::new(),
        fusion_optimizer_sha256: String::new(),
        noncommercial_weight_dependencies: Vec::new(),
        weight_ancestors: c.warm_start.iter().cloned().collect(),
    };
    for step in start..c.steps {
        if wall.elapsed().as_secs() >= c.max_seconds || run.join("STOP").exists() {
            break;
        }
        let tick = Instant::now();
        let samples: Vec<_> = (0..c.batch_size)
            .map(|j| sampler.sample(step * c.batch_size + j, false))
            .collect();
        let mask = mask(grid, c.mask_ratio, c.seed, step, c.train_mask)?;
        let augmentation_start = Instant::now();
        let warped = c
            .equivariance
            .as_ref()
            .map(|config| {
                crate::data::augmentation::batch::<B::InnerBackend>(
                    &training,
                    &samples,
                    config,
                    c.seed,
                    step,
                    ec.patch_size,
                    device,
                )
            })
            .transpose()?;
        let augmentation_seconds = augmentation_start.elapsed().as_secs_f64();
        let warp_valid_fraction = warped.as_ref().map(|b| b.valid_fraction);
        let (rgb, refs) = host_batch::<B::InnerBackend>(&training, &samples, c.references, device);
        let targets = teacher
            .forward_image(normalize(rgb.clone(), &ec), None)
            .tokens;
        let auxiliary_reference = step % c.references;
        let auxiliary_teacher = c.fusion_auxiliary.enabled().then(|| {
            if affinity_encoder.is_some() || !affinity_layers.is_empty() {
                let output = affinity_encoder
                    .as_ref()
                    .unwrap_or(&teacher)
                    .forward_image_capture_layers(
                        normalize(
                            Tensor::cat(vec![rgb.clone(), refs[auxiliary_reference].clone()], 0),
                            &ec,
                        ),
                        None,
                        &affinity_layers,
                    );
                let tokens = output.tokens;
                (
                    tokens.clone().slice_dim(0, 0..c.batch_size),
                    tokens.slice_dim(0, c.batch_size..2 * c.batch_size),
                    output
                        .hierarchical
                        .into_iter()
                        .map(|tokens| {
                            (
                                tokens.clone().slice_dim(0, 0..c.batch_size),
                                tokens.slice_dim(0, c.batch_size..2 * c.batch_size),
                            )
                        })
                        .collect::<Vec<_>>(),
                )
            } else {
                (
                    targets.clone(),
                    teacher
                        .forward_image_capture_layers(
                            normalize(refs[auxiliary_reference].clone(), &ec),
                            None,
                            &[],
                        )
                        .tokens,
                    Vec::new(),
                )
            }
        });
        let target = Tensor::<B, 4>::from_inner(rgb);
        let references: Vec<_> = refs.into_iter().map(Tensor::<B, 4>::from_inner).collect();
        let features = model.encode_references(&references);
        let prediction = model.predict_encoded(
            model.encode(target.clone(), Some(&mask)),
            features.clone(),
            &mask,
            grid,
        )?;
        let dense_target = (c.fusion_auxiliary.enabled()
            || c.equivariance.is_some()
            || (c.ri_weight > 0. && step >= c.ri_start_step))
            .then(|| model.encode(target, None));
        let (aux_attention, aux_dense, aux_descriptor) =
            if let Some((target_teacher, reference_teacher, hierarchy)) = auxiliary_teacher {
                let pair = model.fusion.decoder.pair_training(
                    dense_target.as_ref().unwrap().clone(),
                    features[auxiliary_reference].clone(),
                    grid,
                )?;
                let affinity = if hierarchy.is_empty() {
                    crate::fusion_objective::teacher_affinity(
                        target_teacher.clone(),
                        reference_teacher.clone(),
                        c.fusion_auxiliary.teacher_temperature,
                    )
                } else {
                    crate::fusion_objective::hierarchical_affinity(
                        hierarchy.clone(),
                        c.fusion_auxiliary.teacher_temperature,
                    )
                };
                let mut attention = crate::fusion_objective::attention_kl(
                    pair.logits,
                    Tensor::from_inner(affinity.clone()),
                );
                let mut dense = (model.fusion.prediction.forward(pair.features.clone())
                    - Tensor::from_inner(normalize_teacher(target_teacher.clone())))
                .powf_scalar(2.)
                .mean();
                let descriptor = if c.fusion_auxiliary.bidirectional {
                    let reverse = model.fusion.decoder.pair_training(
                        features[auxiliary_reference].clone(),
                        dense_target.as_ref().unwrap().clone(),
                        grid,
                    )?;
                    let reverse_affinity = if hierarchy.is_empty() {
                        crate::fusion_objective::teacher_affinity(
                            reference_teacher.clone(),
                            target_teacher,
                            c.fusion_auxiliary.teacher_temperature,
                        )
                    } else {
                        crate::fusion_objective::hierarchical_affinity(
                            hierarchy.into_iter().map(|(a, b)| (b, a)).collect(),
                            c.fusion_auxiliary.teacher_temperature,
                        )
                    };
                    attention = (attention
                        + crate::fusion_objective::attention_kl(
                            reverse.logits,
                            Tensor::from_inner(reverse_affinity.clone()),
                        ))
                        * 0.5;
                    dense = (dense
                        + (model.fusion.prediction.forward(reverse.features.clone())
                            - Tensor::from_inner(normalize_teacher(reference_teacher)))
                        .powf_scalar(2.)
                        .mean())
                        * 0.5;
                    if c.fusion_auxiliary.descriptor_weight > 0. {
                        crate::fusion_objective::descriptor_kl(
                            model
                                .spatial_descriptor(
                                    dense_target.as_ref().unwrap().clone(),
                                    pair.features.clone(),
                                )
                                .unwrap_or(pair.features),
                            model
                                .spatial_descriptor(
                                    features[auxiliary_reference].clone(),
                                    reverse.features.clone(),
                                )
                                .unwrap_or(reverse.features),
                            Tensor::from_inner(affinity),
                            Tensor::from_inner(reverse_affinity),
                            c.fusion_auxiliary
                                .descriptor_temperature
                                .unwrap_or(c.fusion_auxiliary.teacher_temperature),
                        )
                    } else {
                        Tensor::zeros([1], device)
                    }
                } else {
                    Tensor::zeros([1], device)
                };
                (attention, dense, descriptor)
            } else {
                (
                    Tensor::zeros([1], device),
                    Tensor::zeros([1], device),
                    Tensor::zeros([1], device),
                )
            };
        let (warp_pair, warp_self) = if let Some(batch) = warped {
            equivariance::losses(
                &model,
                dense_target.as_ref().unwrap().clone(),
                batch,
                c.equivariance.as_ref().unwrap(),
                grid,
            )?
        } else {
            (Tensor::zeros([1], device), Tensor::zeros([1], device))
        };
        let ri = if c.ri_weight > 0. && step >= c.ri_start_step {
            Some(model.predict_improvement(dense_target.unwrap(), features, grid)?)
        } else {
            None
        };
        let mut loss = latent_loss(
            prediction,
            Tensor::from_inner(targets),
            ri,
            &mask,
            c.visible_loss_weight,
            c.ri_weight,
        );
        loss.total = loss.total
            + aux_attention.clone() * c.fusion_auxiliary.attention_weight
            + aux_dense.clone() * c.fusion_auxiliary.dense_weight
            + aux_descriptor.clone() * c.fusion_auxiliary.descriptor_weight;
        if let Some(config) = &c.equivariance {
            loss.total = loss.total
                + (warp_pair.clone() + warp_self.clone() * config.self_weight)
                    * (config.weight / (1. + config.self_weight));
        }
        // Logging must not build an unused differentiable branch from the loss.
        let readings = values(Tensor::<B::InnerBackend, 1>::cat(
            vec![
                loss.total.clone().inner(),
                loss.cross.inner(),
                loss.monocular.inner(),
                loss.visible.inner(),
                loss.improvement.inner(),
                aux_attention.inner(),
                aux_dense.inner(),
                aux_descriptor.inner(),
                warp_pair.inner(),
                warp_self.inner(),
            ],
            0,
        ))?;
        let mut grads = GradientsParams::from_grads(loss.total.backward(), &model);
        let norm = clip(&model, &mut grads, 1.)?;
        let encoder_grads = take_gradients(&model.encoder, &mut grads);
        let encoder_gradient_tensors = encoder_grads.len();
        ensure!(
            (gate.stage == 0) == encoder_grads.is_empty(),
            "encoder stage gradient contract failed"
        );
        let lr = crate::pilot::LearningRateSchedule {
            warmup_steps: c.warmup_steps,
            decay_steps: c.decay_steps,
            min_lr_ratio: 0.1,
        }
        .learning_rate(c.learning_rate, step);
        if !encoder_grads.is_empty() {
            model.encoder = enc_opt.step(lr * c.encoder_lr_ratio, model.encoder, encoder_grads);
        }
        model.fusion = fusion_opt.step(lr, model.fusion, grads);
        B::sync(device).map_err(|e| anyhow::anyhow!("{e}"))?;
        completed = step + 1;
        stage_steps[gate.stage] += 1;
        times.push(tick.elapsed().as_secs_f64());
        writeln!(
            log,
            "{}",
            serde_json::json!({"step":completed,"total":readings[0],"cross":readings[1],"monocular":readings[2],"visible":readings[3],"ri":readings[4],"attention_kl":readings[5],"dense_latent_mse":readings[6],"descriptor_kl":readings[7],"warp_pair_nll":readings[8],"warp_self_nll":readings[9],"warp_valid_fraction":warp_valid_fraction,"augmentation_seconds":augmentation_seconds,"gradient_norm":norm,"learning_rate":lr,"encoder_gradient_tensors":encoder_gradient_tensors,"stage":gate.stage,"seconds":times.last(),"samples":samples.iter().map(|&(s,v)|(training[s].seed,v)).collect::<Vec<_>>()})
        )?;
        log.flush()?;
        if completed.is_multiple_of(50) {
            eprintln!(
                "latent step {completed}/{} stage {} cross {:.5} mono {:.5} {:.1}ms",
                c.steps,
                gate.stage,
                readings[1],
                readings[2],
                times.last().unwrap() * 1000.
            );
        }
        if completed.is_multiple_of(c.eval_every) {
            let eval = evaluate(
                &model.valid(),
                &teacher,
                &validation,
                &val_entries,
                c,
                train_mean.clone(),
                &run.join(format!("step-{completed:06}")),
                false,
                false,
            )?;
            let mse = eval["mean_cross_mse"].as_f64().unwrap();
            let transition = c.unfreeze
                && gate.stage < c.encoder_stage_cap
                && gate.observe(
                    completed,
                    mse,
                    c.unfreeze_min_steps,
                    c.unfreeze_min_improvement,
                );
            probes.push(serde_json::json!({"step":completed,"cross_mse":mse,"monocular_mse":eval["mean_monocular_mse"],"cross_cosine":eval["mean_cross_cosine"],"variance_ratio":eval["mean_spatial_variance_ratio"],"stage":gate.stage,"transitioned":transition}));
            write_json(&run.join("probes.json"), &probes)?;
            if transition {
                model = model.train_encoder(blocks(gate.stage, depth));
            }
            eprintln!(
                "latent validation {completed}: {mse:.5}; stage {} transition {transition}",
                gate.stage
            );
        }
        if c.checkpoint_every > 0 && completed.is_multiple_of(c.checkpoint_every) {
            save_checkpoint(
                &model,
                &enc_opt,
                &fusion_opt,
                &run.join(format!("checkpoint-{completed:06}")),
                metadata(completed, gate.clone()),
            )?;
        }
    }
    ensure!(completed > start, "no updates completed within limit");
    save_checkpoint(
        &model,
        &enc_opt,
        &fusion_opt,
        &run.join("final"),
        metadata(completed, gate.clone()),
    )?;
    let final_eval = evaluate(
        &model.valid(),
        &teacher,
        &validation,
        &val_entries,
        c,
        train_mean.clone(),
        &run.join("validation"),
        true,
        true,
    )?;
    let train_n = training.len().min(c.validation_rooms);
    let train_eval = evaluate(
        &model.valid(),
        &teacher,
        &training[..train_n],
        &train_entries[..train_n],
        c,
        train_mean,
        &run.join("training-probe"),
        false,
        false,
    )?;
    let teacher_delta = scalar(
        (teacher.blocks[0].attn.qkv.weight.val() - teacher_first)
            .abs()
            .max(),
    )?;
    ensure!(teacher_delta == 0., "teacher changed");
    let mut warm = times
        .iter()
        .skip(10.min(times.len() / 2))
        .copied()
        .collect::<Vec<_>>();
    warm.sort_by(f64::total_cmp);
    let median = warm[warm.len() / 2];
    let head_delta = scalar(
        (model.fusion.prediction.weight.val().detach() - head_before)
            .abs()
            .max(),
    )?;
    ensure!(head_delta > 0., "latent head did not update");
    write_json(
        &run.join("report.json"),
        &serde_json::json!({"schema":1,"status":"completed_diagnostic","task":"fixed_vjepa21_latent_prediction","rgb_blur_resolved":false,"completed_steps":completed,"starting_step":start,"stop_reason":if completed==c.steps{"step_limit"}else if run.join("STOP").exists(){"stop_file"}else{"wall_limit"},"prepare_seconds":prepare_seconds,"run_seconds":wall.elapsed().as_secs_f64(),"median_update_seconds":median,"warm_targets_per_second":c.batch_size as f64/median,"stage_steps":stage_steps,"teacher_max_abs_delta":teacher_delta,"prediction_head_max_abs_delta":head_delta,"first_encoder_max_abs_delta":scalar((model.encoder.blocks[0].attn.qkv.weight.val().detach()-first_before).abs().max())?,"last_encoder_max_abs_delta":scalar((model.encoder.blocks[depth-1].attn.qkv.weight.val().detach()-last_before).abs().max())?,"initial_validation_cross_mse":initial_mse,"final_validation":final_eval,"training_probe":train_eval,"probes":probes}),
    )?;
    Ok(())
}
