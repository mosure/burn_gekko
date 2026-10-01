use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    run: PathBuf,
    #[arg(long)]
    resume: Option<PathBuf>,
    #[arg(long)]
    stop_after: Option<usize>,
}
fn main() -> Result<()> {
    let a = Args::parse();
    let c = burn_gekko_data::read_config(&a.config)?;
    // Deliberately CPU: fitting cached small heads needs no renewed GPU allowance.
    burn_gekko_train::training::heads::run::<burn::backend::Autodiff<burn::backend::NdArray<f32>>>(
        &c,
        &a.run,
        a.resume.as_deref(),
        a.stop_after,
        &Default::default(),
    )
}
