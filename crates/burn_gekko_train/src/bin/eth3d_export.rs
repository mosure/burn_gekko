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
    let c = burn_gekko_data::read_config(&a.config)?;
    #[cfg(feature = "cuda")]
    type B = burn::backend::Cuda<f32, i32>;
    #[cfg(not(feature = "cuda"))]
    type B = burn::backend::NdArray<f32>;
    burn_gekko_train::eth3d::run::<B>(&c, &a.output, &Default::default())
}
