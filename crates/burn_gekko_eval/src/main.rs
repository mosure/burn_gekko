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
    /// Diagnose hard versus subpatch localization and camera consensus on cached RGB matches.
    PoseLocalization {
        #[arg(long)]
        config: PathBuf,
        /// Verify a completed report and emit its checkpoint-bound publication capability.
        #[arg(long)]
        report_only: bool,
    },
    /// Paired room-level uncertainty across a fixed synthetic pose solver-seed panel.
    CompareSyntheticPose {
        #[arg(long)]
        config: PathBuf,
    },
    /// Validate target-cache identity and coverage for every room and directed pair.
    AuditViewTargets {
        #[arg(long)]
        config: PathBuf,
    },
    /// Score synthetic pose retention from unfiltered RGB matches across fixed solver seeds.
    SyntheticPose {
        #[arg(long)]
        config: PathBuf,
    },
    /// Verify a fixed latent-export prefix after exporter-only changes.
    LatentReplay {
        #[arg(long)]
        config: PathBuf,
    },
    /// Repeat frozen pose predictions under declared solver seeds on CPU.
    PoseStability {
        #[arg(long)]
        config: PathBuf,
    },
    /// Diagnose completion amplitude, spatial structure and reference gains on CPU.
    LatentDetail {
        #[arg(long)]
        config: PathBuf,
    },
    /// Select a fixed renderer-supervised auxiliary screen using synthetic validation only.
    SelectViewGeometry {
        #[arg(long)]
        config: PathBuf,
    },
    /// Cache renderer-supervised cross-view targets on CPU, separate from RGB inputs.
    PrepareViewTargets {
        #[arg(long)]
        config: PathBuf,
    },
    /// Select a registered final-feature preservation screen against matched controls.
    SelectPreservation {
        #[arg(long)]
        config: PathBuf,
    },
    /// Analyze pinned Nsight GPU/API traces without inferring SM occupancy.
    Dispatch {
        #[arg(long)]
        config: PathBuf,
    },
    /// Check correspondence preservation after exporter engineering.
    PoseReplay {
        #[arg(long)]
        config: PathBuf,
    },
    /// Select between matched encoder adaptation screens using validation only.
    SelectAdaptation {
        #[arg(long)]
        config: PathBuf,
    },
    /// Check numerical preservation of a fixed training-update prefix.
    Replay {
        #[arg(long)]
        config: PathBuf,
    },
    /// Cache a fixed TUM Freiburg 3 RGB cohort with separate camera labels.
    PrepareTum {
        #[arg(long)]
        config: PathBuf,
    },
    /// Fit calibrated poses after RGB inference; failures remain in the metrics.
    Pose {
        #[arg(long)]
        config: PathBuf,
    },
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
        Command::PoseLocalization {
            config,
            report_only,
        } => {
            let config = burn_gekko_data::read_config(&config)?;
            let result = if report_only {
                burn_gekko_eval::pose::localization::publish(&config)?
            } else {
                burn_gekko_eval::pose::localization::analyze(&config)?
            };
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::CompareSyntheticPose { config } => {
            let result = burn_gekko_eval::pose::synthetic_comparison::compare(
                &burn_gekko_data::read_config(&config)?,
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::AuditViewTargets { config } => {
            let result =
                burn_gekko_eval::target_audit::audit(&burn_gekko_data::read_config(&config)?)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::SyntheticPose { config } => {
            let result =
                burn_gekko_eval::pose::synthetic::score(&burn_gekko_data::read_config(&config)?)?;
            println!("{}", serde_json::to_string_pretty(&result["methods"])?);
        }
        Command::LatentReplay { config } => {
            let result =
                burn_gekko_eval::latent_replay::verify(&burn_gekko_data::read_config(&config)?)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::PoseStability { config } => {
            let result =
                burn_gekko_eval::pose::stability::analyze(&burn_gekko_data::read_config(&config)?)?;
            println!("{}", serde_json::to_string_pretty(&result["contrasts"])?);
        }
        Command::LatentDetail { config } => {
            let result =
                burn_gekko_eval::latent_detail::analyze(&burn_gekko_data::read_config(&config)?)?;
            println!("{}", serde_json::to_string_pretty(&result["methods"])?);
        }
        Command::SelectViewGeometry { config } => {
            let result = burn_gekko_eval::adaptation::view_geometry::select(
                &burn_gekko_data::read_config(&config)?,
            )?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::PrepareViewTargets { config } => {
            burn_gekko_data::view_targets::prepare(&burn_gekko_data::read_config(&config)?)?;
        }
        Command::SelectPreservation { config } => {
            let r = burn_gekko_eval::adaptation::preservation::select(
                &burn_gekko_data::read_config(&config)?,
            )?;
            println!("{}", serde_json::to_string_pretty(&r)?);
        }
        Command::Dispatch { config } => {
            burn_gekko_eval::dispatch::summarize(&burn_gekko_data::read_config(&config)?)?;
        }
        Command::PoseReplay { config } => {
            let r = burn_gekko_eval::pose::replay::verify(&burn_gekko_data::read_config(&config)?)?;
            println!("{}", serde_json::to_string_pretty(&r)?);
        }
        Command::SelectAdaptation { config } => {
            let r = burn_gekko_eval::adaptation::select(&burn_gekko_data::read_config(&config)?)?;
            println!("{}", serde_json::to_string_pretty(&r)?);
        }
        Command::Replay { config } => {
            let r = burn_gekko_eval::replay::verify(&burn_gekko_data::read_config(&config)?)?;
            println!("{}", serde_json::to_string_pretty(&r)?);
        }
        Command::PrepareTum { config } => {
            burn_gekko_data::tum::prepare(&burn_gekko_data::read_config(&config)?)?;
        }
        Command::Pose { config } => {
            let r =
                burn_gekko_eval::pose::benchmark::score(&burn_gekko_data::read_config(&config)?)?;
            println!("{}", serde_json::to_string_pretty(&r.methods)?);
        }
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
