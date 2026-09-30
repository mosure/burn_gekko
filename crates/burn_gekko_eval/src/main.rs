use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;
#[derive(Parser)]
#[command(about = "Native single-checkpoint evaluation, without Python or GPU inference")]
struct Args {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    Activity {
        #[arg(long)]
        config: PathBuf,
    },
    Training {
        #[arg(long)]
        config: PathBuf,
    },
    Score {
        #[arg(long)]
        config: PathBuf,
    },
    Camera {
        #[arg(long)]
        config: PathBuf,
    },
    References {
        #[arg(long)]
        config: PathBuf,
    },
}
fn main() -> Result<()> {
    match Args::parse().command {
        Command::Activity { config } => {
            burn_gekko_eval::process_activity::summarize(&burn_gekko_data::read_config(&config)?)?;
        }
        Command::Training { config } => {
            burn_gekko_eval::training_export::summarize(&burn_gekko_data::read_config(&config)?)?;
        }
        Command::References { config } => {
            burn_gekko_eval::reference_count::score(&burn_gekko_data::read_config(&config)?)?
        }
        Command::Camera { config } => {
            burn_gekko_eval::camera_export::score(&burn_gekko_data::read_config(&config)?)?
        }
        Command::Score { config } => {
            let c = burn_gekko_data::read_config(&config)?;
            let result = burn_gekko_eval::benchmark::score(&c)?;
            println!("{}", serde_json::to_string_pretty(&result["methods"])?);
        }
    }
    Ok(())
}
