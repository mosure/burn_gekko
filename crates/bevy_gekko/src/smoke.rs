//! Opt-in native renderer/inference qualification; never a replacement for inference.
use crate::state::Demo;
use bevy::{
    app::AppExit,
    prelude::*,
    render::view::screenshot::{Screenshot, ScreenshotCaptured},
};
use std::{
    path::PathBuf,
    time::{Duration, Instant},
};

#[derive(Resource)]
pub struct Smoke {
    directory: Option<PathBuf>,
    saving: bool,
    started: Instant,
}
pub fn configure(app: &mut App) {
    let args: Vec<_> = std::env::args().collect();
    let directory = args
        .windows(2)
        .find(|p| p[0] == "--smoke-output")
        .map(|p| PathBuf::from(&p[1]));
    if directory.is_some() {
        app.insert_resource(Smoke {
            directory,
            saving: false,
            started: Instant::now(),
        })
        .add_systems(Update, receipt);
    }
}
fn receipt(
    mut commands: Commands,
    demo: Res<Demo>,
    mut smoke: ResMut<Smoke>,
    mut exit: MessageWriter<AppExit>,
) {
    if smoke.started.elapsed() > Duration::from_secs(240) {
        error!("Native demo smoke timed out: {}", demo.status);
        exit.write(AppExit::error());
        return;
    }
    let Some(result) = &demo.result else {
        return;
    };
    if smoke.saving {
        return;
    }
    smoke.saving = true;
    let directory = smoke.directory.as_ref().unwrap();
    let save = || -> anyhow::Result<()> {
        std::fs::create_dir_all(directory)?;
        std::fs::write(
            directory.join("receipt.json"),
            serde_json::to_vec_pretty(&serde_json::json!({
                "result": result, "camera_score": demo.camera_score, "camera_truth": demo.camera_truth,
                "elapsed_seconds": smoke.started.elapsed().as_secs_f64(), "input_revision": demo.revision
            }))?,
        )?;
        for (i, input) in demo.images.iter().enumerate() {
            image::save_buffer(
                directory.join(format!("input-{i}.png")),
                &input.pixels,
                input.size as u32,
                input.size as u32,
                image::ColorType::Rgb8,
            )?;
        }
        image::save_buffer(
            directory.join("predicted.png"),
            &result.rgb,
            result.size as u32,
            result.size as u32,
            image::ColorType::Rgb8,
        )?;
        Ok(())
    };
    if let Err(error) = save() {
        error!("Native smoke receipt: {error}");
        exit.write(AppExit::error());
        return;
    }
    commands
        .spawn(Screenshot::primary_window())
        .observe(finished);
}
fn finished(event: On<ScreenshotCaptured>, smoke: Res<Smoke>, mut exit: MessageWriter<AppExit>) {
    let result = event
        .image
        .clone()
        .try_into_dynamic()
        .map_err(|e| format!("{e}"))
        .and_then(|image| {
            image
                .save(smoke.directory.as_ref().unwrap().join("viewer.png"))
                .map_err(|e| e.to_string())
        });
    match result {
        Ok(()) => {
            info!("GEKKO_NATIVE_SMOKE_PASS");
            exit.write(AppExit::Success);
        }
        Err(e) => {
            error!("Screenshot failed: {e}");
            exit.write(AppExit::error());
        }
    }
}
