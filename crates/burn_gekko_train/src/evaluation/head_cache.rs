//! One frozen RGB inference pass; camera/RGB supervision is attached afterwards.
use crate::{latent_assess::load_assessed_model, latent_pilot::WeightAncestor};
use anyhow::{Result, ensure};
use burn::tensor::{Tensor, backend::Backend};
use burn_gekko_data::{
    Split,
    head_cache::{ArrayFile, CAMERA_FRAME, HeadCache, HeadSample, camera_target},
    load_dataset_rgb, load_geometry, read_dataset_manifest, sha256_file, write_config, write_json,
};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CacheConfig {
    pub dataset: PathBuf,
    pub weights: WeightAncestor,
    pub train_rooms: usize,
    pub validation_rooms: usize,
    pub seed: u64,
    pub mask_ratio: f32,
    pub references: usize,
}

pub fn pool<B: Backend>(x: Tensor<B, 3>, grid: [usize; 2]) -> Tensor<B, 3> {
    let [b, n, d] = x.dims();
    assert_eq!(n, grid[0] * grid[1]);
    assert!(grid.iter().all(|n| n.is_multiple_of(4)));
    x.reshape([b, 4, grid[0] / 4, 4, grid[1] / 4, d])
        .mean_dim(2)
        .mean_dim(4)
        .reshape([b, 16, d])
}

fn save<B: Backend, const D: usize>(
    root: &Path,
    file: String,
    x: Tensor<B, D>,
) -> Result<ArrayFile> {
    let data = burn_gekko::tensor::values(x)?;
    fs::write(
        root.join(&file),
        data.iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )?;
    Ok(ArrayFile {
        sha256: sha256_file(&root.join(&file))?,
        file,
        values: data.len(),
    })
}

pub fn run<B: Backend>(c: &CacheConfig, out: &Path, device: &B::Device) -> Result<()> {
    ensure!(
        !out.exists()
            && (1..=128).contains(&c.train_rooms)
            && (1..=64).contains(&c.validation_rooms)
            && c.mask_ratio > 0.
            && c.mask_ratio < 1.,
        "invalid/existing head cache"
    );
    ensure!(
        fs::canonicalize(out.ancestors().find(|p| p.exists()).unwrap())?
            .starts_with(fs::canonicalize(".data")?),
        "cache outside .data"
    );
    let manifest = read_dataset_manifest(&c.dataset)?;
    ensure!(
        c.references > 0 && c.references < manifest.config.cameras,
        "invalid reference count"
    );
    let grid = [manifest.config.height / 16, manifest.config.width / 16];
    ensure!(
        grid.iter().all(|n| n.is_multiple_of(4)),
        "head cache requires grid divisible by four"
    );
    let model = load_assessed_model::<B>(&c.weights, device)?.model;
    let latent_width = model.encoder_config.encoder.embed_dim;
    let camera_width = model.fusion.prediction.weight.dims()[0];
    let mask = crate::masking::mask(
        grid,
        c.mask_ratio,
        c.seed,
        0,
        crate::masking::MaskPattern::Random,
    )?;
    fs::create_dir_all(out)?;
    write_config(&out.join("config.toml"), c)?;
    let mut samples = Vec::new();
    for (split, count) in [
        (Split::Train, c.train_rooms),
        (Split::Validation, c.validation_rooms),
    ] {
        let entries: Vec<_> = manifest
            .scenes
            .iter()
            .filter(|s| s.split == split)
            .take(count)
            .collect();
        ensure!(entries.len() == count, "insufficient head-cache rooms");
        for entry in entries {
            let scene = load_dataset_rgb(&c.dataset, &manifest, entry)?;
            let rgb: Vec<_> = (0..scene.views.len())
                .map(|v| crate::encoder::image_tensor(&scene, v, device))
                .collect();
            let encoded = model.encode_references(&rgb);
            let mut features = Vec::new();
            for v in 0..rgb.len() {
                let r = (v + 1) % rgb.len();
                let refs = (1..=c.references)
                    .map(|offset| encoded[(v + offset) % rgb.len()].clone())
                    .collect();
                let predicted = model.predict_encoded(
                    model.encode(rgb[v].clone(), Some(&mask)),
                    refs,
                    &mask,
                    grid,
                )?;
                let a = model.fusion.decoder.pair_features(
                    encoded[v].clone(),
                    encoded[r].clone(),
                    grid,
                )?;
                let b = model.fusion.decoder.pair_features(
                    encoded[r].clone(),
                    encoded[v].clone(),
                    grid,
                )?;
                let pair = Tensor::cat(vec![pool(a, grid), pool(b, grid)], 1);
                let prefix = format!("{}-{v}", entry.seed);
                features.push((
                    v,
                    r,
                    save(out, format!("{prefix}-cross.f32"), predicted.cross)?,
                    save(out, format!("{prefix}-mono.f32"), predicted.monocular)?,
                    save(out, format!("{prefix}-camera.f32"), pair)?,
                ));
            }
            // Labels are loaded only after all model outputs for the room exist.
            let geometry = load_geometry(&c.dataset.join("raw").join(&entry.file))?;
            for (v, r, completion, monocular, camera_features) in features {
                samples.push(HeadSample {
                    room_seed: entry.seed,
                    split,
                    target_view: v,
                    reference_view: r,
                    hidden_tokens: (0..grid[0] * grid[1])
                        .filter(|i| !mask.indices().contains(i))
                        .collect(),
                    completion,
                    monocular,
                    camera_features,
                    rgb: save(
                        out,
                        format!("{}-{v}-rgb.f32", entry.seed),
                        burn_gekko::loss::rgb_patches(rgb[v].clone(), 16, false),
                    )?,
                    camera: camera_target(
                        &geometry.world_from_view[v],
                        &geometry.world_from_view[r],
                        geometry.fovy[v],
                        scene.width,
                        scene.height,
                    )?,
                });
            }
            eprintln!("cached {:?} room {}", split, entry.seed);
        }
    }
    let cache=HeadCache {
        schema:1, checkpoint_sha256:c.weights.model_sha256.clone(),dataset_id:manifest.dataset_id,
        dataset_manifest_sha256:sha256_file(&c.dataset.join("manifest.json"))?,config_sha256:sha256_file(&out.join("config.toml"))?,
        source_sha256:crate::provenance::identity()?,grid,latent_width,camera_width,camera_frame:CAMERA_FRAME.into(),
        input_contract:"RGB decoder: sparse-target completion latents with real references, no target/teacher pixels. Camera: separate dense RGB pair fusion, both directions, pooled 4x4 spatial cells. No calibration/geometry labels enter either inference route. Validation is disjoint by room but reused development for the frozen foundation checkpoint.".into(),
        samples,
    };
    cache.validate()?;
    write_json(&out.join("manifest.json"), &cache)
}
