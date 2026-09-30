//! Bound CUDA upload layout cost independently of learned-model quality.
use anyhow::{Result, ensure};
use burn::tensor::backend::Backend;
use burn_gekko_train::encoder::{RgbUploadLayout, normalize, upload_rgb_layout};
use clap::Parser;
use serde::{Deserialize, Serialize};
use std::{path::PathBuf, time::Instant};

#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    output: PathBuf,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    batches: Vec<usize>,
    image_size: usize,
    warmup: usize,
    repetitions: usize,
}
fn run<B: Backend>(c: &Config, out: &std::path::Path, device: &B::Device) -> Result<()> {
    ensure!(!out.exists(), "preserve benchmark outputs");
    ensure!(
        out.parent()
            .unwrap()
            .canonicalize()?
            .starts_with(std::fs::canonicalize(".data")?),
        "output outside .data"
    );
    ensure!(
        c.batches.iter().all(|&b| b > 0 && b <= 32)
            && !c.batches.is_empty()
            && c.image_size <= 512
            && c.image_size > 0
            && c.repetitions > 0
            && c.repetitions <= 100
            && c.warmup <= 10,
        "invalid benchmark bound"
    );
    let encoder = burn_vjepa::VJepaConfig::tiny_for_tests();
    let mut results = Vec::new();
    let modes = [
        RgbUploadLayout::LegacyNhwc,
        RgbUploadLayout::FlatNhwc,
        RgbUploadLayout::PlanarNchw,
    ];
    for &batch in &c.batches {
        let shape = [batch, c.image_size, c.image_size];
        let data: Vec<_> = (0..batch * c.image_size * c.image_size * 3)
            .map(|i| (i % 997) as f32 / 997.)
            .collect();
        let mut reference = None;
        for &mode in &modes {
            let tensor = normalize(
                upload_rgb_layout::<B>(data.clone(), shape, mode, device),
                &encoder,
            );
            let values = burn_gekko_train::latent_eval::values(tensor)?;
            if let Some(old) = &reference {
                ensure!(old == &values, "layout changed normalized RGB values");
            } else {
                reference = Some(values);
            }
        }
        let mut times = [Vec::new(), Vec::new(), Vec::new()];
        // Rotate order to reduce systematic clock/temperature drift.
        for repeat in 0..c.warmup + c.repetitions {
            for offset in 0..3 {
                let index = (repeat + offset) % 3;
                let start = Instant::now();
                let tensor = normalize(
                    upload_rgb_layout::<B>(data.clone(), shape, modes[index], device),
                    &encoder,
                );
                std::hint::black_box(&tensor);
                B::sync(device).map_err(|e| anyhow::anyhow!("{e}"))?;
                if repeat >= c.warmup {
                    times[index].push(start.elapsed().as_secs_f64());
                }
            }
        }
        for (layout, mut seconds) in modes.into_iter().zip(times) {
            seconds.sort_by(f64::total_cmp);
            let median = seconds[seconds.len() / 2];
            eprintln!("batch {batch}: {layout:?} {median:.6} s");
            results.push(serde_json::json!({"batch":batch,"layout":layout,"median_seconds":median,"seconds":seconds,"rgb_bytes":data.len()*4,"normalized_values_bitwise_equal":true}));
        }
    }
    burn_gekko_data::write_json(
        out,
        &serde_json::json!({"config":c,"backend":std::any::type_name::<B>(),"scope":"host clone/packing, upload and normalization with device synchronization; alternating layout order; no optimizer or quality claim","results":results}),
    )
}
fn main() -> Result<()> {
    let a = Args::parse();
    let c = burn_gekko_data::read_config(&a.config)?;
    #[cfg(feature = "cuda")]
    type B = burn::backend::Cuda<f32, i32>;
    #[cfg(not(feature = "cuda"))]
    type B = burn::backend::NdArray<f32>;
    run::<B>(&c, &a.output, &Default::default())
}
