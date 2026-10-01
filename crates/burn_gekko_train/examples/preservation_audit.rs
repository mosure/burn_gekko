//! Native CUDA audit of identical-weight feature targets across batch layouts.
#[cfg(feature = "cuda")]
mod audit {
    use anyhow::{Result, ensure};
    use burn::tensor::Tensor;
    use burn_gekko::objectives::preservation::relative_feature_mse;
    use burn_gekko_data::{
        Split, load_dataset_rgb, read_config, read_dataset_manifest, sha256_file, write_json,
    };
    use burn_gekko_train::{
        batch::SampleSchedule, e2e_pilot::host_batch, encoder::normalize,
        latent_assess::load_assessed_model, latent_pilot::LatentConfig, masking::mask,
        train::scalar,
    };
    use std::{collections::BTreeMap, path::Path};
    type I = burn::backend::Cuda<f32, i32>;
    type A = burn::backend::Autodiff<I>;

    pub fn run() -> Result<()> {
        let args = std::env::args().collect::<Vec<_>>();
        ensure!(
            args.len() == 3,
            "usage: preservation_audit CONFIG.toml OUTPUT.json"
        );
        let output = Path::new(&args[2]);
        ensure!(
            output.starts_with(".data") && !output.exists(),
            "choose a new .data output"
        );
        let c: LatentConfig = read_config(Path::new(&args[1]))?;
        let anchor = &c
            .encoder_preservation
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing preservation config"))?
            .anchor;
        ensure!(
            c.warm_start.as_ref() == Some(anchor),
            "audit requires identical starting weights"
        );
        let device = Default::default();
        let loaded = load_assessed_model::<A>(anchor, &device)?;
        let depth = loaded.model.encoder.blocks.len();
        let model = loaded.model.train_encoder(depth + 1);
        let frozen = load_assessed_model::<I>(anchor, &device)?.model;
        ensure!(
            scalar(
                (model.encoder.blocks[0].attn.qkv.weight.val().inner()
                    - frozen.encoder.blocks[0].attn.qkv.weight.val())
                .abs()
                .max()
            )? == 0.,
            "initial encoder weights differ"
        );
        let manifest = read_dataset_manifest(&c.dataset)?;
        let entries = manifest
            .scenes
            .iter()
            .filter(|s| s.split == Split::Train)
            .take(c.train_rooms)
            .collect::<Vec<_>>();
        let mut sampler = SampleSchedule::new(c.train_rooms, manifest.config.cameras, c.seed, true);
        let samples = (0..c.batch_size)
            .map(|j| sampler.sample(j, false))
            .collect::<Vec<_>>();
        let rooms = samples
            .iter()
            .map(|&(s, _)| load_dataset_rgb(&c.dataset, &manifest, entries[s]))
            .collect::<Result<Vec<_>>>()?;
        let local = samples
            .iter()
            .enumerate()
            .map(|(j, &(_, v))| (j, v))
            .collect::<Vec<_>>();
        let (rgb, refs) = host_batch::<I>(&rooms, &local, c.references, &device);
        let grid = [manifest.config.height / 16, manifest.config.width / 16];
        let mask = mask(grid, c.mask_ratio, c.seed, 0, c.train_mask)?;
        let ad_refs = refs
            .iter()
            .cloned()
            .map(Tensor::<A, 4>::from_inner)
            .collect::<Vec<_>>();
        let reference_features = model.encode_references(&ad_refs);
        let sparse_features = model.encode(Tensor::from_inner(rgb.clone()), Some(&mask));
        let prediction = model.predict_encoded(
            sparse_features.clone(),
            reference_features.clone(),
            &mask,
            grid,
        )?;
        let student_refs = reference_features
            .into_iter()
            .map(|x| model.final_encoder_features(x))
            .collect::<Vec<_>>();
        let student_sparse = model.final_encoder_features(sparse_features);
        let student_dense =
            model.final_encoder_features(model.encode(Tensor::from_inner(rgb.clone()), None));
        let mut results = BTreeMap::new();
        for capture in [false, true] {
            let levels = if capture {
                vec![c.spatial_input_layer.unwrap() - 1]
            } else {
                vec![]
            };
            let encode = |x, m| {
                frozen
                    .encoder
                    .forward_image_capture_layers(normalize(x, &frozen.encoder_config), m, &levels)
                    .tokens
            };
            let sparse = encode(rgb.clone(), Some(&mask));
            let dense = encode(rgb.clone(), None);
            let references = encode(Tensor::cat(refs.clone(), 0), None);
            let mut views = vec![rgb.clone()];
            views.extend(refs.clone());
            let joint = encode(Tensor::cat(views, 0), None);
            let mut row = BTreeMap::new();
            let b = c.batch_size;
            row.insert(
                "sparse_matched".to_string(),
                scalar(relative_feature_mse(
                    student_sparse.clone(),
                    Tensor::from_inner(sparse),
                ))?,
            );
            row.insert(
                "target_matched".into(),
                scalar(relative_feature_mse(
                    student_dense.clone(),
                    Tensor::from_inner(dense),
                ))?,
            );
            row.insert(
                "target_joint".into(),
                scalar(relative_feature_mse(
                    student_dense.clone(),
                    Tensor::from_inner(joint.clone().slice_dim(0, 0..b)),
                ))?,
            );
            for (i, student) in student_refs.iter().enumerate() {
                row.insert(
                    format!("reference_{i}_matched"),
                    scalar(relative_feature_mse(
                        student.clone(),
                        Tensor::from_inner(references.clone().slice_dim(0, i * b..(i + 1) * b)),
                    ))?,
                );
                row.insert(
                    format!("reference_{i}_joint"),
                    scalar(relative_feature_mse(
                        student.clone(),
                        Tensor::from_inner(joint.clone().slice_dim(0, (i + 1) * b..(i + 2) * b)),
                    ))?,
                );
            }
            results.insert(
                if capture {
                    "with_intermediate_capture"
                } else {
                    "final_only_capture"
                },
                row,
            );
        }
        write_json(
            output,
            &serde_json::json!({"schema":1,"config_sha256":sha256_file(Path::new(&args[1]))?,"checkpoint":anchor,"batch_size":c.batch_size,"samples":samples,"relative_feature_mse":results,"prediction_cross_mean":scalar(prediction.cross.mean())?,"scope":"First registered minibatch; no optimizer or backward step. Decoder forward occurs before full-target encoding; student features retain their autodiff graphs and the feature loss is differentiable. Matched layouts use B sparse, B full target and R*B references; joint uses (R+1)*B full views. This is a numerical diagnostic, not a training result."}),
        )?;
        Ok(())
    }
}

fn main() -> anyhow::Result<()> {
    #[cfg(feature = "cuda")]
    return audit::run();
    #[cfg(not(feature = "cuda"))]
    anyhow::bail!("build preservation_audit with --features cuda")
}
