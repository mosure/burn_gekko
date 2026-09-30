use anyhow::{Result, ensure};
use burn::tensor::{Tensor, TensorData, backend::Backend};
use burn_gekko_train::released;
use burn_vjepa::SparseTokenMask;
use clap::Parser;
use std::{
    fs,
    path::{Path, PathBuf},
};
#[derive(Parser)]
struct Args {
    #[arg(long)]
    weights: PathBuf,
    #[arg(long)]
    fixture: PathBuf,
    #[arg(long)]
    output: PathBuf,
    #[arg(long, default_value = "cuda")]
    backend: String,
}
fn export<B: Backend, const D: usize>(path: &Path, t: Tensor<B, D>) -> Result<()> {
    let values = t.into_data().convert::<f32>().to_vec::<f32>()?;
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(path, bytes)?;
    Ok(())
}
fn run<B: Backend>(args: &Args) -> Result<()> {
    let d = Default::default();
    let (encoder, decoder) = released::load::<B>(&args.weights, &d)?;
    let meta: serde_json::Value =
        serde_json::from_slice(&fs::read(args.fixture.join("fixture.json"))?)?;
    let h = meta["height"].as_u64().unwrap() as usize;
    let w = meta["width"].as_u64().unwrap() as usize;
    let ids: Vec<usize> = meta["visible"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    let mask = SparseTokenMask::new(ids, h * w / 256)?;
    let read = |name: &str| -> Result<Tensor<B, 4>> {
        let bytes = fs::read(args.fixture.join(name))?;
        let values: Vec<f32> = bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        Ok(
            Tensor::<B, 4>::from_data(TensorData::new(values, [1, h, w, 3]), &d)
                .permute([0, 3, 1, 2]),
        )
    };
    let target = read("target.f32")?;
    let reference = read("reference.f32")?;
    let sparse = encoder.forward(target.clone(), Some(&mask));
    let full = encoder.forward(target, None);
    let reference = encoder.forward(reference, None);
    fs::create_dir_all(&args.output)?;
    export(&args.output.join("sparse.f32"), sparse.clone())?;
    export(&args.output.join("full.f32"), full.clone())?;
    export(&args.output.join("reference.f32"), reference.clone())?;
    let cross = decoder.features(
        sparse.clone(),
        Some(reference.clone()),
        Some(&mask),
        [h / 16, w / 16],
        false,
    );
    let mae = decoder.features(sparse, None, Some(&mask), [h / 16, w / 16], true);
    let dense = decoder.features(full, Some(reference), None, [h / 16, w / 16], false);
    export(&args.output.join("cross-features.f32"), cross.clone())?;
    export(
        &args.output.join("cross.f32"),
        decoder.cross_head.forward(cross),
    )?;
    export(&args.output.join("mae.f32"), decoder.mae_head.forward(mae))?;
    export(
        &args.output.join("dense.f32"),
        decoder.cross_head.forward(dense),
    )?;
    let changed = encoder.forward(read("intervened.f32")?, Some(&mask));
    let clean = encoder.forward(read("target.f32")?, Some(&mask));
    let delta = burn_gekko_train::train::scalar((changed - clean).abs().max())?;
    ensure!(
        delta == 0.,
        "hidden target intervention leaked into sparse encoder"
    );
    burn_gekko_data::write_json(
        &args.output.join("audit.json"),
        &serde_json::json!({"tensors_loaded":712,"hidden_intervention_max_abs":delta,"backend":B::name(&d)}),
    )?;
    Ok(())
}
fn main() -> Result<()> {
    let args = Args::parse();
    match args.backend.as_str() {
        "cpu" => run::<burn::backend::NdArray<f32>>(&args),
        #[cfg(feature = "cuda")]
        "cuda" => run::<burn::backend::Cuda<f32, i32>>(&args),
        _ => anyhow::bail!("unsupported backend"),
    }
}
