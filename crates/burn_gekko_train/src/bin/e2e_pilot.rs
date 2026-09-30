use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    run: PathBuf,
    #[arg(long,value_parser=["validation","test"])]
    evaluate: Option<String>,
    /// End-to-end checkpoint directory; resumes optimizer state unless evaluating.
    #[arg(long)]
    checkpoint: Option<PathBuf>,
    /// Explicit weights-only fine-tune of an audited own checkpoint; resets AdamW.
    #[arg(long, conflicts_with_all=["checkpoint", "evaluate"])]
    warm_start: Option<PathBuf>,
    #[arg(long)]
    unrelated: bool,
}
fn main() -> Result<()> {
    let a = Args::parse();
    let c = burn_gekko_data::read_config(&a.config)?;
    let split = a.evaluate.map(|s| {
        if s == "test" {
            burn_gekko_data::Split::Test
        } else {
            burn_gekko_data::Split::Validation
        }
    });
    #[cfg(feature = "cuda")]
    type B = burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>;
    #[cfg(not(feature = "cuda"))]
    type B = burn::backend::Autodiff<burn::backend::NdArray<f32>>;
    burn_gekko_train::e2e_pilot::run_with_warm_start::<B>(
        &c,
        &a.run,
        split,
        a.checkpoint.as_deref(),
        a.warm_start.as_deref(),
        a.unrelated,
        &Default::default(),
    )
}
