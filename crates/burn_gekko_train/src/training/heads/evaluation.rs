use super::data::Sample;
use anyhow::{Result, ensure};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn_gekko::{
    heads::{calibration::CalibrationHead, reconstruction::RgbReconstructionHead},
    tensor::values,
};
use burn_gekko_data::{Split, head_cache::HeadCache, sha256_file, write_json};
use burn_gekko_eval::heads::{
    HeadRow as Row, HeadScores as Scores, RgbScore, camera_score, rgb_score, summarize,
};
use std::{fs, path::Path};

pub fn evaluate<B: Backend>(
    camera: &CalibrationHead<B>,
    rgb: &RgbReconstructionHead<B>,
    samples: &[Sample],
    cache: &HeadCache,
    split: Split,
    out: Option<&Path>,
    device: &B::Device,
) -> Result<Scores> {
    if let Some(out) = out {
        ensure!(!out.exists(), "head export exists");
        fs::create_dir_all(out)?;
    }
    let n = cache.grid.iter().product::<usize>();
    let tensor = |v: &Vec<f32>, shape| Tensor::from_data(TensorData::new(v.clone(), shape), device);
    let mut rows = Vec::new();
    let mut exports = Vec::new();
    for s in samples.iter().filter(|s| s.meta.split == split) {
        let cp = values(camera.forward(tensor(&s.camera, [1, 32, cache.camera_width])))?;
        let p = values(rgb.forward(tensor(&s.cross, [1, n, cache.latent_width])))?;
        let m = values(rgb.forward(tensor(&s.mono, [1, n, cache.latent_width])))?;
        let row = Row {
            room_seed: s.meta.room_seed,
            target_view: s.meta.target_view,
            reference_view: s.meta.reference_view,
            camera: camera_score(&cp, &s.meta.camera)?,
            rgb: rgb_score(&p, &s.rgb, &s.meta.hidden_tokens)?,
            monocular: rgb_score(&m, &s.rgb, &s.meta.hidden_tokens)?,
            camera_prediction: cp,
        };
        if let Some(out) = out {
            let mut files = std::collections::BTreeMap::new();
            for (name, data) in [("prediction", &p), ("monocular", &m), ("target", &s.rgb)] {
                let file = out.join(format!(
                    "{}-{}-{name}.f32",
                    s.meta.room_seed, s.meta.target_view
                ));
                fs::write(
                    &file,
                    data.iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>(),
                )?;
                files.insert(
                    name,
                    serde_json::json!({"path":file,"sha256":sha256_file(&file)?}),
                );
            }
            exports.push(serde_json::json!({"room_seed":s.meta.room_seed,"target_view":s.meta.target_view,"hidden_tokens":s.meta.hidden_tokens,"files":files,"metrics":row,"camera_target":s.meta.camera}));
        }
        rows.push(row);
    }
    let scores = summarize(rows)?;
    if let Some(out) = out {
        write_json(
            &out.join("predictions.json"),
            &serde_json::json!({"schema":1,"grid":cache.grid,"checkpoint_sha256":cache.checkpoint_sha256,"samples":exports,"scores":scores}),
        )?;
    }
    Ok(scores)
}

/// A label-only constant fitted on training rooms; no validation fitting.
pub fn constant_camera(samples: &[Sample]) -> Result<Scores> {
    let train: Vec<_> = samples
        .iter()
        .filter(|s| s.meta.split == Split::Train)
        .collect();
    ensure!(!train.is_empty(), "missing camera baseline training set");
    let mean: Vec<_> = (0..11)
        .map(|i| {
            train
                .iter()
                .map(|s| s.meta.camera.regression()[i] as f64)
                .sum::<f64>() as f32
                / train.len() as f32
        })
        .collect();
    let rows = samples
        .iter()
        .filter(|s| s.meta.split == Split::Validation)
        .map(|s| {
            Ok(Row {
                room_seed: s.meta.room_seed,
                target_view: s.meta.target_view,
                reference_view: s.meta.reference_view,
                camera: camera_score(&mean, &s.meta.camera)?,
                camera_prediction: mean.clone(),
                rgb: RgbScore {
                    mse: 0.,
                    psnr_db: None,
                },
                monocular: RgbScore {
                    mse: 0.,
                    psnr_db: None,
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    summarize(rows)
}
