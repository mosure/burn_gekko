//! Export a single-checkpoint known-transform diagnostic using native Burn.
use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    output: PathBuf,
}
fn main() -> Result<()> {
    let a = Args::parse();
    #[cfg(feature = "cuda")]
    type B = burn::backend::Cuda<f32, i32>;
    #[cfg(not(feature = "cuda"))]
    type B = burn::backend::NdArray<f32>;
    burn_gekko_train::evaluation::equivariance::run::<B>(
        &burn_gekko_data::read_config(&a.config)?,
        &a.output,
        &Default::default(),
    )
}
