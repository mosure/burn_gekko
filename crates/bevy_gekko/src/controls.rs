use crate::{
    state::{CaptureStamp, Demo, Mode, Snapshot},
    worker::{Work, Worker},
};
use bevy::{
    camera::RenderTarget,
    prelude::*,
    render::view::screenshot::{Screenshot, ScreenshotCaptured},
};
use bevy_egui::EguiContexts;
use bevy_panorbit_camera::PanOrbitCamera;
use bevy_zeroverse::{
    app::BevyZeroverseConfig,
    camera::{CaptureCameraIndex, EditorCameraMarker, ZeroverseCamera},
    scene::{RegenerateSceneEvent, SceneLoadedEvent, procedural_indoor::layout::IndoorManifest},
};
use burn_gekko_inference::Request;

pub fn keyboard(
    keys: Res<ButtonInput<KeyCode>>,
    mut demo: ResMut<Demo>,
    mut contexts: EguiContexts,
) {
    if contexts
        .ctx_mut()
        .is_ok_and(|ctx| ctx.egui_wants_keyboard_input())
    {
        return;
    }
    for (i, key) in [
        KeyCode::Digit1,
        KeyCode::Digit2,
        KeyCode::Digit3,
        KeyCode::Digit4,
    ]
    .into_iter()
    .enumerate()
    {
        if keys.just_pressed(key) {
            demo.select = Some(i);
            if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
                demo.place = true;
            }
        }
    }
    if keys.just_pressed(KeyCode::Digit0) {
        demo.overview = true;
    }
    if keys.just_pressed(KeyCode::Space) || keys.just_pressed(KeyCode::KeyI) {
        demo.request = true;
    }
    if keys.just_pressed(KeyCode::KeyC) {
        demo.place = true;
    }
    if keys.just_pressed(KeyCode::KeyN) {
        demo.regenerate = true;
    }
    if keys.just_pressed(KeyCode::KeyU) && !demo.busy {
        demo.upload = true;
    }
    if keys.just_pressed(KeyCode::KeyL) {
        demo.load = true;
    }
}
fn jump(camera: &Transform, transform: &mut Transform, pan: &mut PanOrbitCamera, radius: f32) {
    *transform = *camera;
    let back = camera.back();
    let yaw = back.x.atan2(back.z);
    // PanOrbit measures elevation; its X rotation is the negative of pitch.
    let pitch = back.y.clamp(-1., 1.).asin();
    pan.focus = camera.translation + camera.forward() * radius;
    pan.target_focus = pan.focus;
    pan.radius = Some(radius);
    pan.target_radius = radius;
    pan.yaw = Some(yaw);
    pan.target_yaw = yaw;
    pan.pitch = Some(pitch);
    pan.target_pitch = pitch;
    pan.force_update = true;
}
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
pub fn scene(
    mut demo: ResMut<Demo>,
    mut config: ResMut<BevyZeroverseConfig>,
    manifest: Option<Res<IndoorManifest>>,
    mut cameras: Query<(
        &CaptureCameraIndex,
        &mut ZeroverseCamera,
        &mut Transform,
        &Projection,
        &mut Camera,
    )>,
    mut editor: Query<
        (&mut Transform, &mut PanOrbitCamera, &mut Camera),
        (With<EditorCameraMarker>, Without<ZeroverseCamera>),
    >,
    mut regenerated: MessageReader<SceneLoadedEvent>,
    mut regenerate: MessageWriter<RegenerateSceneEvent>,
) {
    if !regenerated.is_empty() {
        regenerated.clear();
        demo.invalidate("Scene ready. Select a camera, then press Space.");
        demo.initialized = false;
        demo.scene_frames = 0;
    }
    demo.scene_frames = demo.scene_frames.saturating_add(1);
    if demo.regenerate {
        demo.regenerate = false;
        demo.mode = Mode::Scene;
        demo.invalidate("Generating a new room…");
        demo.images.clear();
        demo.last_signature.clear();
        demo.initialized = false;
        config.indoor_seed = Some(config.indoor_seed.unwrap_or(5) + 1);
        regenerate.write(RegenerateSceneEvent);
    }
    for (_, _, _, _, mut camera) in &mut cameras {
        camera.is_active = demo.mode == Mode::Scene;
    }
    if let Ok((_, _, mut camera)) = editor.single_mut() {
        camera.is_active = demo.mode == Mode::Scene;
    }
    if demo.mode == Mode::Images {
        if let Some(index) = demo.select.take()
            && index < demo.images.len()
        {
            demo.target = index;
            demo.invalidate("Target changed. Press Space to infer.");
        }
        demo.place = false;
        demo.overview = false;
        return;
    }
    if !demo.initialized && cameras.iter().count() >= 3 && editor.single_mut().is_ok() {
        demo.select = Some(0);
        demo.initialized = true;
    }
    // Read editor pose before changing selection so Shift+number stores the current view.
    let pose = editor.single_mut().ok().map(|(t, _, _)| *t);
    if let Some(index) = demo.select.take() {
        if index < cameras.iter().count() {
            demo.target = index;
            demo.invalidate("Target changed. Press Space to infer.");
            if !demo.place
                && let Some((_, _, camera, _, _)) =
                    cameras.iter().find(|(i, _, _, _, _)| i.0 == index)
                && let Ok((mut transform, mut pan, _)) = editor.single_mut()
            {
                jump(camera, &mut transform, &mut pan, 2.);
            }
        } else {
            demo.place = false;
        }
    }
    if demo.place {
        demo.place = false;
        if let Some(pose) = pose {
            for (index, mut camera, mut transform, _, _) in &mut cameras {
                if index.0 == demo.target {
                    camera.override_transform = Some(pose);
                    *transform = pose;
                }
            }
            demo.invalidate("Camera moved to editor pose. Press Space for fresh annotations.");
        }
    }
    if demo.overview {
        demo.overview = false;
        if let Some(manifest) = manifest.as_ref()
            && let Some(view) = manifest.cameras.first()
            && let Ok((mut transform, mut pan, _)) = editor.single_mut()
        {
            // Use Zeroverse's collision-checked interior home view. A position
            // above the room sees the opaque roof instead of its contents.
            let rotation = Quat::from_rotation_y(manifest.world_yaw);
            let center = rotation * view.target;
            let pose =
                Transform::from_translation(rotation * view.start).looking_at(center, Vec3::Y);
            jump(
                &pose,
                &mut transform,
                &mut pan,
                pose.translation.distance(center),
            );
        }
    }
    let mut rows: Vec<_> = cameras
        .iter()
        .map(|(i, _, t, p, _)| {
            let fov = match p {
                Projection::Perspective(p) => p.fov,
                _ => 0.,
            };
            (i.0, t.to_matrix().to_cols_array(), fov)
        })
        .collect();
    rows.sort_by_key(|r| r.0);
    let signature: Vec<_> = rows
        .into_iter()
        .flat_map(|r| r.1.into_iter().chain([r.2]))
        .collect();
    if signature != demo.last_signature {
        let changed = !demo.last_signature.is_empty();
        demo.last_signature = signature;
        demo.scene_frames = 0;
        if changed {
            demo.invalidate("Camera settings changed. Press Space to refresh annotations.");
        }
    }
}

#[allow(clippy::type_complexity)]
pub fn capture(
    mut commands: Commands,
    mut demo: ResMut<Demo>,
    worker: Res<Worker>,
    cameras: Query<
        (
            &CaptureCameraIndex,
            &RenderTarget,
            &GlobalTransform,
            &Projection,
        ),
        With<ZeroverseCamera>,
    >,
) {
    if let Some(snapshot) = demo.pending.as_ref()
        && snapshot.images.iter().all(Option::is_some)
    {
        let snapshot = demo.pending.take().unwrap();
        demo.camera_truth = snapshot.truth;
        demo.images = snapshot.images.into_iter().map(Option::unwrap).collect();
        demo.status = "Predicting RGB, calibration, matches and visibility scores…".into();
        worker.send(Work::Infer {
            request: Request {
                revision: demo.revision,
                target: demo.target,
                images: demo.images.clone(),
            },
        });
    }
    if !demo.request {
        return;
    }
    demo.request = false;
    if !demo.model_ready {
        demo.status = "Load the model before running inference.".into();
        return;
    }
    if demo.busy {
        return;
    }
    if demo.mode == Mode::Images {
        if demo.images.len() < 2 {
            demo.status = "Upload two to four views first.".into();
            return;
        }
        demo.busy = true;
        demo.status = "Predicting from uploaded images…".into();
        worker.send(Work::Infer {
            request: Request {
                revision: demo.revision,
                target: demo.target,
                images: demo.images.clone(),
            },
        });
        return;
    }
    if demo.scene_frames < 30 {
        demo.request = true;
        demo.status = "Waiting for the scene and rendering to settle…".into();
        return;
    }
    let mut views: Vec<_> = cameras.iter().collect();
    views.sort_by_key(|v| v.0.0);
    if views.len() != 3
        || demo.target >= views.len()
        || views.iter().any(|v| !matches!(v.1, RenderTarget::Image(_)))
    {
        demo.status = "Waiting for all three scene cameras.".into();
        return;
    }
    let target = views[demo.target];
    let reference = views[(demo.target + 1) % views.len()];
    let fov = match target.3 {
        Projection::Perspective(p) => p.fov,
        _ => 0.,
    };
    let truth = burn_gekko_metrics::calibration::camera_target(
        &target.2.to_matrix().to_cols_array(),
        &reference.2.to_matrix().to_cols_array(),
        fov,
        256,
        256,
    )
    .ok();
    let revision = demo.revision;
    demo.pending = Some(Snapshot {
        revision,
        images: vec![None; views.len()],
        truth,
    });
    demo.busy = true;
    demo.status = "Capturing the current three camera views…".into();
    for (index, (_, target, _, _)) in views.into_iter().enumerate() {
        if let RenderTarget::Image(image) = target {
            commands
                .spawn((
                    Screenshot::image(image.handle.clone()),
                    CaptureStamp {
                        revision,
                        view: index,
                    },
                ))
                .observe(captured);
        }
    }
}
fn captured(event: On<ScreenshotCaptured>, stamps: Query<&CaptureStamp>, mut demo: ResMut<Demo>) {
    let Ok(stamp) = stamps.get(event.entity) else {
        return;
    };
    let Some(pending) = demo.pending.as_mut() else {
        return;
    };
    if stamp.revision != pending.revision {
        return;
    }
    let result = crate::capture_image::input(&event.image, format!("Camera {}", stamp.view + 1));
    match result {
        Ok(image) => pending.images[stamp.view] = Some(image),
        Err(e) => {
            demo.pending = None;
            demo.busy = false;
            demo.status = format!("Capture failed: {e}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn editor_jump_keeps_the_look_direction_and_overview_focus() {
        let center = Vec3::new(1., 1., -3.);
        let pose =
            Transform::from_translation(center + Vec3::new(4., 7., 8.)).looking_at(center, Vec3::Y);
        let mut transform = Transform::default();
        let mut pan = PanOrbitCamera::default();
        jump(
            &pose,
            &mut transform,
            &mut pan,
            pose.translation.distance(center),
        );
        let rotation =
            Quat::from_rotation_y(pan.target_yaw) * Quat::from_rotation_x(-pan.target_pitch);
        assert!(pan.focus.distance(center) < 1e-5);
        assert!(
            (pan.focus + rotation * Vec3::Z * pan.target_radius).distance(pose.translation) < 1e-5
        );
        assert!((rotation * Vec3::NEG_Z).dot(*pose.forward()) > 0.99999);
    }
}
