use crate::{
    controls,
    state::{Demo, Mode},
    ui,
    worker::{Event, Worker},
};
use bevy::prelude::*;
use bevy_zeroverse::{
    app::{BevyZeroverseConfig, viewer_app},
    camera::{DefaultZeroverseCamera, PlaybackMode},
    scene::{ZeroverseSceneType, procedural_indoor::IndoorQuality},
};

pub fn run() {
    #[cfg(not(target_arch = "wasm32"))]
    let (root, auto, images) = {
        let args: Vec<_> = std::env::args().collect();
        let root = args
            .windows(2)
            .find(|p| p[0] == "--model")
            .map(|p| p[1].clone())
            .unwrap_or_else(|| "https://mosure.github.io/burn_gekko/demo/models".into());
        (
            root,
            args.iter().any(|x| x == "--auto-infer"),
            args.iter().any(|x| x == "--images" || x == "--image"),
        )
    };
    #[cfg(target_arch = "wasm32")]
    let (root, auto, images) = {
        let query = web_sys::window()
            .and_then(|w| w.location().search().ok())
            .unwrap_or_default();
        (
            "models".into(),
            query.contains("auto_infer=1"),
            query.contains("mode=images"),
        )
    };
    let config = BevyZeroverseConfig {
        name: "bevy_gekko — Multi-view inference".into(),
        width: 1440.,
        height: 900.,
        editor: false,
        num_cameras: 3,
        camera_grid: false,
        gizmos: true,
        regenerate_ms: 0,
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        indoor_seed: Some(5),
        indoor_human_density: 0.,
        indoor_quality: IndoorQuality::Portable,
        playback_mode: PlaybackMode::Still,
        playback_speed: 0.,
        animated: false,
        rotation_augmentation: false,
        keybinds: false,
        press_esc_close: !cfg!(target_arch = "wasm32"),
        image_copiers: false,
        initialize_scene: !images,
        ..default()
    };
    let mut app = viewer_app(None, Some(config));
    app.insert_resource(bevy_egui::EguiGlobalSettings {
        auto_create_primary_context: false,
        ..default()
    })
    .add_plugins(bevy_egui::EguiPlugin::default());
    let mut demo = Demo::new(root, auto);
    if images {
        demo.mode = Mode::Images;
        demo.status = "Upload two to four photographs of one scene.".into();
    }
    demo.load = auto;
    let worker = Worker::new();
    #[cfg(not(target_arch = "wasm32"))]
    {
        let args: Vec<_> = std::env::args().collect();
        let paths: Vec<_> = args
            .windows(2)
            .filter(|p| p[0] == "--image")
            .map(|p| p[1].clone())
            .collect();
        if !paths.is_empty() {
            let images: anyhow::Result<Vec<_>> = paths
                .iter()
                .map(|p| burn_gekko_inference::RgbInput::decode(p.clone(), &std::fs::read(p)?, 256))
                .collect();
            let event = match images {
                Ok(images) if (2..=4).contains(&images.len()) => Event::Uploaded { images },
                Ok(_) => Event::Error {
                    message: "Pass two to four --image paths.".into(),
                },
                Err(e) => Event::Error {
                    message: e.to_string(),
                },
            };
            let _ = worker.output.try_send(event);
        }
    }
    app.insert_resource(demo)
        .insert_resource(worker)
        .insert_resource(DefaultZeroverseCamera {
            resolution: Some(UVec2::splat(256)),
        })
        .insert_resource(bevy::winit::WinitSettings {
            focused_mode: bevy::winit::UpdateMode::reactive(std::time::Duration::from_millis(100)),
            unfocused_mode: bevy::winit::UpdateMode::reactive_low_power(
                std::time::Duration::from_millis(500),
            ),
        })
        .add_systems(
            Update,
            (
                events,
                controls::keyboard,
                controls::scene,
                controls::capture,
            )
                .chain(),
        )
        .add_systems(bevy_egui::EguiPrimaryContextPass, ui::panel);
    #[cfg(target_arch = "wasm32")]
    {
        let world = app.world_mut();
        let mut windows = world.query::<&mut Window>();
        for mut window in windows.iter_mut(world) {
            window.fit_canvas_to_parent = true;
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    crate::smoke::configure(&mut app);
    app.run();
}

fn events(mut demo: ResMut<Demo>, worker: Res<Worker>) {
    while let Ok(event) = worker.events.try_recv() {
        match event {
            Event::Progress { message } => demo.status = message,
            Event::Ready => {
                demo.model_ready = true;
                demo.model_loading = false;
                demo.status = "Model ready. Press Space to capture and infer.".into();
                if demo.auto_infer {
                    demo.request = true;
                    demo.auto_infer = false;
                }
            }
            Event::Error { message } => {
                demo.busy = false;
                demo.model_loading = false;
                demo.pending = None;
                demo.status = format!("Error: {message}");
                error!("{}", demo.status);
            }
            Event::Output { result } => {
                demo.busy = false;
                if result.revision == demo.revision {
                    demo.camera_score = demo.camera_truth.as_ref().and_then(|t| {
                        burn_gekko_metrics::heads::camera_score(&result.camera, t).ok()
                    });
                    demo.status =
                        "Inference complete. Results describe this captured revision.".into();
                    info!(
                        "GEKKO_INFERENCE revision={} psnr={:?}",
                        result.revision, result.rgb_score.psnr_db
                    );
                    demo.result = Some(*result);
                    demo.texture_revision = None;
                } else {
                    demo.status =
                        "Scene changed during inference. Press Space for fresh annotations.".into();
                }
            }
            Event::Uploaded { images } => {
                demo.invalidate("Images loaded. Press Space for inference.");
                demo.mode = Mode::Images;
                demo.images = images;
                demo.target = 0;
            }
            Event::UploadCancelled => demo.status = "Image selection cancelled.".into(),
            Event::UploadError { message } => {
                demo.status = format!("Image selection failed: {message}")
            }
        }
    }
    if demo.load && !demo.model_loading && !demo.model_ready {
        demo.load = false;
        demo.model_loading = true;
        demo.status = "Loading model…".into();
        worker.send(crate::worker::Work::Load {
            root: demo.model_root.clone(),
        });
    }
    if demo.upload {
        demo.upload = false;
        worker.upload();
    }
}
