use anyhow::{Result, ensure};
use bevy_zeroverse::{
    app::OvoxelMode,
    render::RenderMode,
    scene::{
        ZeroverseSceneType,
        procedural_indoor::{IndoorQuality, layout::IndoorLayout},
    },
};
use bevy_zeroverse_burn::{
    chunk::ColorCodec,
    compression::Compression,
    generator::{GenConfig, WriteMode, run_chunk_generation, validate_gen_config},
};
use burn_gekko_data::{CaptureConfig, read_config};
use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
struct Args {
    #[arg(long)]
    config: Option<PathBuf>,
    #[arg(long)]
    output: Option<PathBuf>,
    #[arg(long)]
    identity: bool,
    /// Validate and print the resolved configuration without starting the renderer.
    #[arg(long)]
    dry_run: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.identity {
        println!("{}", burn_gekko_data::GENERATOR);
        return Ok(());
    }
    let config_path = args
        .config
        .ok_or_else(|| anyhow::anyhow!("--config required"))?;
    let output = args
        .output
        .ok_or_else(|| anyhow::anyhow!("--output required"))?;
    let config: CaptureConfig = read_config(&config_path)?;
    config.validate()?;
    let capture = GenConfig {
        output: output.clone(),
        workers: 1,
        chunk_size: 1,
        samples: config.scenes(),
        playback_step: 0.0,
        playback_steps: 1,
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        compression: Compression::Zstd { level: 3 },
        color_codec: ColorCodec::Raw,
        render_modes: vec![RenderMode::Color, RenderMode::Depth, RenderMode::Position],
        timeout_secs: config.timeout_secs,
        width: config.width as u32,
        height: config.height as u32,
        seed: Some(config.seed),
        indoor_layout: IndoorLayout::Mixed,
        indoor_density: config.density,
        indoor_human_density: 0.0,
        indoor_gi_rays: 64,
        indoor_quality: IndoorQuality::Portable,
        indoor_camera: Some(if let Some(baseline) = config.camera_baseline {
            format!(
                r#"{{"primary_room":true,"path_length_min":0.0,"path_length_max":0.0,"long_path_fraction":0.0,"baseline":{baseline}}}"#
            )
        } else if config.multiview == Some(true) {
            r#"{"primary_room":true,"path_length_min":0.0,"path_length_max":0.0,"long_path_fraction":0.0,"multiview":{}}"#.into()
        } else {
            r#"{"primary_room":true,"path_length_min":0.0,"path_length_max":0.0,"long_path_fraction":0.0,"multiview":null}"#.into()
        }),
        cameras: config.cameras,
        enable_ui: false,
        write_mode: WriteMode::Chunk,
        ov_mode: OvoxelMode::Disabled,
        export_ovoxel: false,
        main_thread_app: true,
        ..GenConfig::default()
    };
    validate_gen_config(&capture)?;
    if args.dry_run {
        println!("{}\n{capture:#?}", burn_gekko_data::GENERATOR);
        return Ok(());
    }
    ensure!(
        !output.exists() || std::fs::read_dir(&output)?.next().is_none(),
        "capture output must be empty"
    );
    run_chunk_generation(capture)
}
