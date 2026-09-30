use anyhow::{Result, ensure};
use burn_gekko_data::{
    CaptureConfig, audit, config_toml, generate, open_dataset, read_config, write_json,
};
use burn_gekko_train::eval::{EvaluationOptions, evaluate_with_options};
use burn_gekko_train::pilot::{PilotConfig, run_pilot};
use burn_gekko_train::train::{TrainConfig, train};
use clap::{Parser, Subcommand, ValueEnum};
use std::path::{Component, Path, PathBuf};

#[derive(Parser)]
#[command(
    about = "Burn multi-view training, capture and assessment. Runtime artifacts go under ./.data/"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}
#[derive(Subcommand)]
enum Command {
    /// Train fixed-teacher latent completion, with audited initialization and a time ceiling.
    TrainLatent {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        run: PathBuf,
        #[arg(long)]
        resume: Option<PathBuf>,
        #[arg(long, value_enum, default_value = "cpu")]
        backend: Backend,
    },
    /// Export latent completion, co-visibility and optional geometric diagnostics.
    AssessLatent {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, value_enum, default_value = "cpu")]
        backend: Backend,
    },
    /// Check known image-transform correspondences for one audited checkpoint.
    AssessEquivariance {
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        output: PathBuf,
        #[arg(long, value_enum, default_value = "cpu")]
        backend: Backend,
    },
    /// Cache a tiny published-Zeroverse dataset, or reuse the identical verified capture.
    Capture {
        #[arg(long, default_value = "configs/data/capture-preflight.toml")]
        config: PathBuf,
        #[arg(
            long,
            default_value = "tools/zeroverse_capture/target/debug/gekko_zeroverse_capture"
        )]
        binary: PathBuf,
        #[arg(long)]
        dry_run: bool,
    },
    /// Verify cache hashes, scene splits, and RGB contracts without a GPU.
    VerifyDataset {
        #[arg(long)]
        dataset: PathBuf,
    },
    /// Explicitly revalidate complete staged captures after an adapter repair; never re-renders.
    RecoverCapture {
        #[arg(long)]
        staging: PathBuf,
        #[arg(
            long,
            default_value = "tools/zeroverse_capture/target/debug/gekko_zeroverse_capture"
        )]
        binary: PathBuf,
    },
    /// Verify camera/geometry conventions and compute directional co-visibility counts.
    AuditGeometry {
        #[arg(long)]
        dataset: PathBuf,
    },
    /// Tiny RGB preflight for pipeline contracts; use train-latent for primary studies.
    TrainPreflight {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long, default_value = "configs/train/train-preflight.toml")]
        config: PathBuf,
        #[arg(long)]
        run: String,
        #[arg(long, value_enum, default_value = "cpu")]
        backend: Backend,
        #[arg(long)]
        resume: Option<PathBuf>,
    },
    /// Report held-out directional co-visibility ranking; preflight scores are diagnostics only.
    EvalPreflight {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        run: PathBuf,
        #[arg(long, value_enum, default_value = "validation")]
        split: EvalSplit,
        #[arg(long, value_enum, default_value = "cpu")]
        backend: Backend,
        #[arg(long)]
        step: Option<usize>,
        /// Optional TOML settings for reference controls and annotated sample exports.
        #[arg(long)]
        options: Option<PathBuf>,
    },
    /// Bounded small-set study with batching, phase timing, fixed probes, and periodic checkpoints.
    TrainPilot {
        #[arg(long)]
        dataset: PathBuf,
        #[arg(long)]
        config: PathBuf,
        #[arg(long)]
        run: String,
        #[arg(long, value_enum, default_value = "cuda")]
        backend: Backend,
        #[arg(long)]
        resume: Option<PathBuf>,
    },
}
#[derive(Clone, Copy, Debug, ValueEnum)]
enum EvalSplit {
    Validation,
    Test,
    /// Requires diagnostic_training=true in the options TOML.
    Train,
}
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Backend {
    Cpu,
    Cuda,
    Wgpu,
}

fn inside_data(path: &Path) -> Result<PathBuf> {
    let root = std::fs::canonicalize(".data")?;
    let path = std::fs::canonicalize(path)?;
    ensure!(
        path.starts_with(root),
        "datasets/checkpoints must be under ./.data/"
    );
    Ok(path)
}
fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::TrainLatent {
            config,
            run,
            resume,
            backend,
        } => {
            let config = read_config(&config)?;
            match backend {
                #[cfg(feature = "ndarray")]
                Backend::Cpu => {
                    burn_gekko_train::training::latent::run::<
                        burn::backend::Autodiff<burn::backend::NdArray<f32>>,
                    >(&config, &run, resume.as_deref(), &Default::default())?
                }
                #[cfg(not(feature = "ndarray"))]
                Backend::Cpu => anyhow::bail!("rebuild with feature ndarray"),
                #[cfg(feature = "cuda")]
                Backend::Cuda => {
                    burn_gekko_train::training::latent::run::<
                        burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>,
                    >(&config, &run, resume.as_deref(), &Default::default())?
                }
                #[cfg(not(feature = "cuda"))]
                Backend::Cuda => anyhow::bail!("rebuild with feature cuda"),
                #[cfg(feature = "wgpu")]
                Backend::Wgpu => {
                    burn_gekko_train::training::latent::run::<
                        burn::backend::Autodiff<burn::backend::Wgpu<f32, i32>>,
                    >(&config, &run, resume.as_deref(), &Default::default())?
                }
                #[cfg(not(feature = "wgpu"))]
                Backend::Wgpu => anyhow::bail!("rebuild with feature wgpu"),
            }
        }
        Command::AssessLatent {
            config,
            output,
            backend,
        } => {
            let config = read_config(&config)?;
            match backend {
                #[cfg(feature = "ndarray")]
                Backend::Cpu => burn_gekko_train::evaluation::assessment::run::<
                    burn::backend::NdArray<f32>,
                >(&config, &output, &Default::default())?,
                #[cfg(not(feature = "ndarray"))]
                Backend::Cpu => anyhow::bail!("rebuild with feature ndarray"),
                #[cfg(feature = "cuda")]
                Backend::Cuda => burn_gekko_train::evaluation::assessment::run::<
                    burn::backend::Cuda<f32, i32>,
                >(&config, &output, &Default::default())?,
                #[cfg(not(feature = "cuda"))]
                Backend::Cuda => anyhow::bail!("rebuild with feature cuda"),
                #[cfg(feature = "wgpu")]
                Backend::Wgpu => burn_gekko_train::evaluation::assessment::run::<
                    burn::backend::Wgpu<f32, i32>,
                >(&config, &output, &Default::default())?,
                #[cfg(not(feature = "wgpu"))]
                Backend::Wgpu => anyhow::bail!("rebuild with feature wgpu"),
            }
        }
        Command::AssessEquivariance {
            config,
            output,
            backend,
        } => {
            let config = read_config(&config)?;
            match backend {
                #[cfg(feature = "ndarray")]
                Backend::Cpu => burn_gekko_train::evaluation::equivariance::run::<
                    burn::backend::NdArray<f32>,
                >(&config, &output, &Default::default())?,
                #[cfg(not(feature = "ndarray"))]
                Backend::Cpu => anyhow::bail!("rebuild with feature ndarray"),
                #[cfg(feature = "cuda")]
                Backend::Cuda => burn_gekko_train::evaluation::equivariance::run::<
                    burn::backend::Cuda<f32, i32>,
                >(&config, &output, &Default::default())?,
                #[cfg(not(feature = "cuda"))]
                Backend::Cuda => anyhow::bail!("rebuild with feature cuda"),
                #[cfg(feature = "wgpu")]
                Backend::Wgpu => burn_gekko_train::evaluation::equivariance::run::<
                    burn::backend::Wgpu<f32, i32>,
                >(&config, &output, &Default::default())?,
                #[cfg(not(feature = "wgpu"))]
                Backend::Wgpu => anyhow::bail!("rebuild with feature wgpu"),
            }
        }
        Command::RecoverCapture { staging, binary } => {
            let staging = inside_data(&staging)?;
            println!(
                "{}",
                burn_gekko_data::recover_capture(Path::new(".data"), &binary, &staging)?.display()
            );
        }
        Command::TrainPilot {
            dataset,
            config,
            run,
            backend,
            resume,
        } => {
            let dataset = inside_data(&dataset)?;
            let mut it = Path::new(&run).components();
            ensure!(
                matches!(it.next(), Some(Component::Normal(_))) && it.next().is_none(),
                "run must be a directory name"
            );
            let run = PathBuf::from(".data/runs").join(run);
            let config: PilotConfig = read_config(&config)?;
            let resume = resume.map(|p| inside_data(&p)).transpose()?;
            let report = match backend {
                #[cfg(feature = "ndarray")]
                Backend::Cpu => run_pilot::<burn::backend::Autodiff<burn::backend::NdArray<f32>>>(
                    &dataset,
                    &run,
                    &config,
                    resume.as_deref(),
                    &Default::default(),
                )?,
                #[cfg(not(feature = "ndarray"))]
                Backend::Cpu => anyhow::bail!("rebuild with feature ndarray"),
                #[cfg(feature = "cuda")]
                Backend::Cuda => {
                    run_pilot::<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>(
                        &dataset,
                        &run,
                        &config,
                        resume.as_deref(),
                        &Default::default(),
                    )?
                }
                #[cfg(not(feature = "cuda"))]
                Backend::Cuda => anyhow::bail!("rebuild with --features cuda"),
                #[cfg(feature = "wgpu")]
                Backend::Wgpu => {
                    run_pilot::<burn::backend::Autodiff<burn::backend::Wgpu<f32, i32>>>(
                        &dataset,
                        &run,
                        &config,
                        resume.as_deref(),
                        &Default::default(),
                    )?
                }
                #[cfg(not(feature = "wgpu"))]
                Backend::Wgpu => anyhow::bail!("rebuild with --features wgpu"),
            };
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &serde_json::json!({"run":run,"completed_steps":report.completed_steps,
                "stop_reason":report.stop_reason,"warm_examples_per_second":report.warm_examples_per_second,
                "fixed_train_loss_reduction":report.fixed_train_loss_reduction,"validation_loss_reduction":report.validation_loss_reduction})
                )?
            );
        }
        Command::EvalPreflight {
            dataset,
            run,
            split,
            backend,
            step,
            options,
        } => {
            let dataset = inside_data(&dataset)?;
            let run = inside_data(&run)?;
            let options: EvaluationOptions = options
                .map(|p| read_config(&p))
                .transpose()?
                .unwrap_or_default();
            let split = match split {
                EvalSplit::Validation => burn_gekko_data::Split::Validation,
                EvalSplit::Test => burn_gekko_data::Split::Test,
                EvalSplit::Train => burn_gekko_data::Split::Train,
            };
            let report = match backend {
                #[cfg(feature = "ndarray")]
                Backend::Cpu => evaluate_with_options::<burn::backend::NdArray<f32>>(
                    &dataset,
                    &run,
                    split,
                    &Default::default(),
                    step,
                    &options,
                )?,
                #[cfg(not(feature = "ndarray"))]
                Backend::Cpu => anyhow::bail!("rebuild with feature ndarray"),
                #[cfg(feature = "cuda")]
                Backend::Cuda => evaluate_with_options::<burn::backend::Cuda<f32, i32>>(
                    &dataset,
                    &run,
                    split,
                    &Default::default(),
                    step,
                    &options,
                )?,
                #[cfg(not(feature = "cuda"))]
                Backend::Cuda => anyhow::bail!("rebuild with --features cuda"),
                #[cfg(feature = "wgpu")]
                Backend::Wgpu => evaluate_with_options::<burn::backend::Wgpu<f32, i32>>(
                    &dataset,
                    &run,
                    split,
                    &Default::default(),
                    step,
                    &options,
                )?,
                #[cfg(not(feature = "wgpu"))]
                Backend::Wgpu => anyhow::bail!("rebuild with --features wgpu"),
            };
            let name = if split == burn_gekko_data::Split::Validation {
                "evaluation-validation.json"
            } else if split == burn_gekko_data::Split::Train {
                "evaluation-training-diagnostic.json"
            } else {
                "evaluation-test.json"
            };
            let name = step
                .map(|s| format!("step-{s:06}-{name}"))
                .unwrap_or_else(|| name.into());
            let name = if options.unrelated_references {
                format!("unrelated-{name}")
            } else {
                name
            };
            write_json(&run.join(name), &report)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Capture {
            config,
            binary,
            dry_run,
        } => {
            let config: CaptureConfig = read_config(&config)?;
            config.validate()?;
            if dry_run {
                print!("{}", config_toml(&config)?);
                eprintln!(
                    "generator: {}\nbinary: {}\ncache: ./.data/datasets/<sha256>",
                    burn_gekko_data::GENERATOR,
                    binary.display()
                );
            } else {
                println!(
                    "{}",
                    generate(Path::new(".data"), &binary, &config)?.display()
                );
            }
        }
        Command::VerifyDataset { dataset } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&open_dataset(&inside_data(&dataset)?)?)?
            );
        }
        Command::AuditGeometry { dataset } => {
            let report = audit(&inside_data(&dataset)?)?;
            std::fs::create_dir_all(".data/reports")?;
            write_json(
                &PathBuf::from(".data/reports")
                    .join(format!("{}-geometry.json", report.dataset_id)),
                &report,
            )?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::TrainPreflight {
            dataset,
            config,
            run,
            backend,
            resume,
        } => {
            let dataset = inside_data(&dataset)?;
            let mut it = Path::new(&run).components();
            ensure!(
                matches!(it.next(), Some(Component::Normal(_))) && it.next().is_none(),
                "run must be a single directory name"
            );
            let run = PathBuf::from(".data/runs").join(run);
            let config: TrainConfig = read_config(&config)?;
            let resume = resume.map(|p| inside_data(&p)).transpose()?;
            let report = match backend {
                #[cfg(feature = "ndarray")]
                Backend::Cpu => train::<burn::backend::Autodiff<burn::backend::NdArray<f32>>>(
                    &dataset,
                    &run,
                    &config,
                    resume.as_deref(),
                    &Default::default(),
                )?,
                #[cfg(not(feature = "ndarray"))]
                Backend::Cpu => anyhow::bail!("rebuild with feature ndarray"),
                #[cfg(feature = "cuda")]
                Backend::Cuda => train::<burn::backend::Autodiff<burn::backend::Cuda<f32, i32>>>(
                    &dataset,
                    &run,
                    &config,
                    resume.as_deref(),
                    &Default::default(),
                )?,
                #[cfg(not(feature = "cuda"))]
                Backend::Cuda => anyhow::bail!("rebuild with --features cuda"),
                #[cfg(feature = "wgpu")]
                Backend::Wgpu => train::<burn::backend::Autodiff<burn::backend::Wgpu<f32, i32>>>(
                    &dataset,
                    &run,
                    &config,
                    resume.as_deref(),
                    &Default::default(),
                )?,
                #[cfg(not(feature = "wgpu"))]
                Backend::Wgpu => anyhow::bail!("rebuild with --features wgpu"),
            };
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
    }
    Ok(())
}
