//! Isolate unused hierarchical branches during partial encoder adaptation.
//! Synthetic feature targets are a memory diagnostic, not a quality experiment.
use anyhow::{Result, ensure};
use burn::{
    module::Module,
    optim::{AdamWConfig, GradientsParams, Optimizer},
    tensor::{Tensor, TensorData, backend::AutodiffBackend},
};
use burn_gekko_data::{
    DatasetManifest, Split, fingerprint, load_rgb, read_config, sha256_file, write_json,
};
use burn_gekko_train::{
    encoder::{image_tensor, normalize},
    latent_eval::values,
    latent_pilot::LatentConfig,
    train::load_encoder,
};
use clap::Parser;
use std::{fs, io::Write, path::PathBuf, time::Instant};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value_t = 200)]
    steps: usize,
    #[arg(long, default_value_t = 48)]
    batch: usize,
    #[arg(long)]
    legacy: bool,
}
fn run<B: AutodiffBackend>(a: Args) -> Result<()> {
    ensure!(
        (1..=200).contains(&a.steps) && (1..=48).contains(&a.batch),
        "diagnostic bounds exceeded"
    );
    ensure!(!a.output.exists(), "output exists");
    let parent = a.output.ancestors().find(|p| p.exists()).unwrap();
    ensure!(
        fs::canonicalize(parent)?.starts_with(fs::canonicalize(".data")?),
        "output outside .data"
    );
    fs::create_dir_all(&a.output)?;
    let c: LatentConfig = read_config(&a.config)?;
    let manifest: DatasetManifest =
        serde_json::from_slice(&fs::read(c.dataset.join("manifest.json"))?)?;
    ensure!(
        manifest.dataset_id
            == fingerprint(&(
                manifest.schema,
                &manifest.generator,
                &manifest.binary_sha256,
                &manifest.config
            ))?,
        "dataset identity mismatch"
    );
    let d = Default::default();
    let (encoder, ec, id) = load_encoder::<B>(&c.teacher, c.seed, &d)?;
    ensure!(
        id == "c408f68dd18a38824d0fa1d615e6f9f9f04f111d71a6f7c846f41dc187a8795f",
        "unreviewed encoder"
    );
    let mut encoder = encoder.no_grad().with_last_blocks_require_grad(2, true);
    let mut views = Vec::new();
    for entry in manifest.scenes.iter().filter(|e| e.split == Split::Train) {
        ensure!(
            std::path::Path::new(&entry.file).components().count() == 1,
            "invalid shard filename"
        );
        let p = c.dataset.join("raw").join(&entry.file);
        ensure!(sha256_file(&p)? == entry.sha256, "shard checksum mismatch");
        let scene = load_rgb(&p)?;
        for view in 0..scene.views.len() {
            if views.len() < a.batch {
                views.push(image_tensor::<B>(&scene, view, &d));
            }
        }
        if views.len() == a.batch {
            break;
        }
    }
    ensure!(views.len() == a.batch, "insufficient RGB views");
    let rgb = normalize(Tensor::cat(views, 0), &ec);
    let n = manifest.config.height / 16 * (manifest.config.width / 16);
    let target = Tensor::<B, 3>::from_data(
        TensorData::new(
            (0..a.batch * n * ec.encoder.embed_dim)
                .map(|i| (i as f32 * 0.017).sin())
                .collect(),
            [a.batch, n, ec.encoder.embed_dim],
        ),
        &d,
    );
    let mut opt = AdamWConfig::new().with_weight_decay(0.).init();
    let mut log = fs::File::create(a.output.join("metrics.jsonl"))?;
    let wall = Instant::now();
    let mut losses = Vec::new();
    for step in 0..a.steps {
        let tick = Instant::now();
        let tokens = if a.legacy {
            encoder.forward_image(rgb.clone(), None).tokens
        } else {
            encoder
                .forward_image_capture_layers(rgb.clone(), None, &[])
                .tokens
        };
        let loss = (tokens - target.clone()).powf_scalar(2.).mean();
        let reading = values(loss.clone().inner())?[0];
        let gradients = GradientsParams::from_grads(loss.backward(), &encoder);
        let count = gradients.len();
        ensure!(count == 26, "unexpected partial encoder gradient count");
        encoder = opt.step(1e-6, encoder, gradients);
        B::sync(&d).map_err(|e| anyhow::anyhow!("{e}"))?;
        writeln!(
            log,
            "{}",
            serde_json::json!({"step":step+1,"loss":reading,"gradient_tensors":count,"seconds":tick.elapsed().as_secs_f64()})
        )?;
        log.flush()?;
        losses.push(reading);
        if (step + 1) % 25 == 0 {
            eprintln!("memory audit step {}/{}", step + 1, a.steps);
        }
    }
    write_json(
        &a.output.join("report.json"),
        &serde_json::json!({"status":"memory_diagnostic_only","legacy_hierarchical_capture":a.legacy,
        "steps":a.steps,"batch":a.batch,"first_loss":losses[0],"last_loss":losses.last(),"seconds":wall.elapsed().as_secs_f64(),
        "teacher_id":id,"noncommercial_weight_dependencies":[],"target":"deterministic synthetic feature vectors; not quality training","cleanup_calls":0}),
    )?;
    Ok(())
}
fn main() -> Result<()> {
    #[cfg(feature = "cuda")]
    type B = burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>;
    #[cfg(not(feature = "cuda"))]
    type B = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
    run::<B>(Args::parse())
}
