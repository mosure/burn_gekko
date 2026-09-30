use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    run: PathBuf,
    #[arg(long, value_parser=["validation","test"])]
    evaluate: Option<String>,
    #[arg(long)]
    unrelated: bool,
}
fn main() -> Result<()> {
    let args = Args::parse();
    let config = burn_gekko_data::read_config(&args.config)?;
    let split = args.evaluate.map(|s| {
        if s == "test" {
            burn_gekko_data::Split::Test
        } else {
            burn_gekko_data::Split::Validation
        }
    });
    #[cfg(feature = "cuda")]
    burn_gekko_train::transport_pilot::run::<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>(
        &config,
        &args.run,
        split,
        args.unrelated,
        &Default::default(),
    )?;
    #[cfg(not(feature = "cuda"))]
    burn_gekko_train::transport_pilot::run::<burn::backend::Autodiff<burn::backend::NdArray<f32>>>(
        &config,
        &args.run,
        split,
        args.unrelated,
        &Default::default(),
    )?;
    Ok(())
}
