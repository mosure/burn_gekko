use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
#[derive(Parser)]
#[command(about = "Build one private project page and paper from one verified experiment")]
struct Args {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Build {
        #[arg(long)]
        experiment: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long)]
        pdf: bool,
    },
    Validate {
        #[arg(long)]
        bundle: PathBuf,
    },
}
fn main() -> Result<()> {
    match Args::parse().command {
        Command::Build {
            experiment,
            output,
            pdf,
        } => burn_gekko_report::build(&experiment, &output, pdf),
        Command::Validate { bundle } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&burn_gekko_report::validate::validate(&bundle)?)?
            );
            Ok(())
        }
    }
}
