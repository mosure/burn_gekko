use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    run: PathBuf,
    /// Exact continuation, including both optimizers and the student unfreeze gate.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
}
fn main() -> Result<()> {
    let a = Args::parse();
    let c = burn_gekko_data::read_config(&a.config)?;
    #[cfg(feature = "cuda")]
    type B = burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>;
    #[cfg(not(feature = "cuda"))]
    type B = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
    burn_gekko_train::latent_pilot::run::<B>(
        &c,
        &a.run,
        a.checkpoint.as_deref(),
        &Default::default(),
    )
}
