use crate::state::{Demo, Mode};
use bevy::{camera::RenderTarget, prelude::*};
use bevy_egui::{EguiContexts, EguiTextureHandle, egui};
use bevy_zeroverse::camera::{CaptureCameraIndex, EditorCameraMarker, ZeroverseCamera};

fn texture(ctx: &egui::Context, name: &str, size: usize, pixels: &[u8]) -> egui::TextureHandle {
    ctx.load_texture(
        name,
        egui::ColorImage::from_rgb([size, size], pixels),
        egui::TextureOptions::NEAREST,
    )
}
fn textures(ctx: &egui::Context, demo: &mut Demo) {
    if demo.texture_revision == Some(demo.revision) {
        return;
    }
    demo.textures.clear();
    for (i, image) in demo.images.iter().enumerate() {
        demo.textures.push(texture(
            ctx,
            &format!("input-{i}"),
            image.size,
            &image.pixels,
        ));
    }
    if let Some(result) = &demo.result {
        demo.textures.push(texture(
            ctx,
            "masked",
            result.size,
            &demo.images[result.target].masked(&result.visible),
        ));
        demo.textures
            .push(texture(ctx, "reconstruction", result.size, &result.rgb));
        demo.textures.push(texture(
            ctx,
            "monocular",
            result.size,
            &result.monocular_rgb,
        ));
        let n = result.size / 16;
        let mut colors = Vec::new();
        for y in 0..result.size {
            for x in 0..result.size {
                let v = result.improvement[y / 16 * n + x / 16].clamp(0., 1.);
                colors.extend([
                    (25. + 210. * v) as u8,
                    (35. + 160. * v) as u8,
                    (90. - 60. * v) as u8,
                ]);
            }
        }
        demo.textures
            .push(texture(ctx, "relative-improvement", result.size, &colors));
    }
    demo.texture_revision = Some(demo.revision);
}
fn image(ui: &mut egui::Ui, texture: &egui::TextureHandle, label: &str, width: f32) {
    ui.vertical(|ui| {
        ui.label(label);
        ui.image((texture.id(), egui::vec2(width, width)));
    });
}
pub fn panel(
    mut contexts: EguiContexts,
    mut demo: ResMut<Demo>,
    cameras: Query<(&CaptureCameraIndex, &RenderTarget), With<ZeroverseCamera>>,
    mut editor: Query<&mut Camera, With<EditorCameraMarker>>,
) {
    let mut live = Vec::new();
    for (index, target) in &cameras {
        if let RenderTarget::Image(image) = target {
            live.push((
                index.0,
                contexts.add_image(EguiTextureHandle::Strong(image.handle.clone())),
            ));
        }
    }
    live.sort_by_key(|x| x.0);
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    textures(ctx, &mut demo);
    let mut viewport = egui::Ui::new(
        ctx.clone(),
        "viewport".into(),
        egui::UiBuilder::new()
            .layer_id(egui::LayerId::background())
            .max_rect(ctx.viewport_rect()),
    );
    egui::Panel::top("gekko-heading").show_inside(&mut viewport, |ui| {
        ui.horizontal(|ui| {
            ui.heading("bevy_gekko");
            ui.label("RGB-only multi-view inference");
            ui.hyperlink_to(
                "Project & paper",
                "https://mosure.github.io/burn_gekko/project/",
            );
        });
    });
    egui::Panel::right("gekko-controls").default_size(465.).min_size(340.).resizable(true).show_inside(&mut viewport,|ui|{
        egui::ScrollArea::vertical().show(ui,|ui|{
            ui.heading("Explore a scene");
            ui.horizontal(|ui|{
                if ui.selectable_label(demo.mode==Mode::Scene,"Zeroverse scene").clicked() && demo.mode!=Mode::Scene {demo.mode=Mode::Scene;demo.initialized=false;demo.invalidate("Scene mode. Press N if no room has been generated.");demo.images.clear();if live.is_empty(){demo.regenerate=true;}}
                if ui.selectable_label(demo.mode==Mode::Images,"Your images").clicked() && demo.mode!=Mode::Images {demo.mode=Mode::Images;demo.invalidate("Upload two to four images of one scene.");demo.images.clear();}
            });
            if demo.mode==Mode::Scene {
                ui.label("1–3: select / visit camera · 0: editor overview");
                ui.label("Drag: orbit · right drag: pan · wheel: zoom");
                ui.label("C or Shift+1–3: place camera at editor pose");
                ui.horizontal(|ui|{if ui.button("New room  N").clicked(){demo.regenerate=true;}
                if ui.button("Place target here  C").clicked(){demo.place=true;}});
                ui.horizontal_wrapped(|ui|{
                    let width=(ui.available_width()/3.-10.).clamp(72.,144.);
                    for (index,id) in &live {ui.vertical(|ui|{
                        if ui.add(egui::Button::image(egui::Image::new((*id,egui::vec2(width,width)))).selected(demo.target==*index)).clicked(){demo.select=Some(*index);}
                        ui.label(format!("{} {}",index+1,if demo.target==*index{"target"}else{"reference"}));
                    });}
                });
            } else {
                ui.label("Two to four photographs of one scene. The first selected view is the target; use number keys to change it.");
                if ui.add_enabled(!demo.busy,egui::Button::new("Upload images…  U")).clicked(){demo.upload=true;}
                ui.small("PNG/JPEG · local processing · center crop to 256 × 256 · camera ground truth unavailable");
                ui.horizontal_wrapped(|ui|{for i in 0..demo.images.len(){if ui.selectable_label(demo.target==i,format!("Target {}",i+1)).clicked(){demo.select=Some(i);}}});
            }
            ui.separator();
            if !demo.model_ready {
                if ui.add_enabled(!demo.model_loading,egui::Button::new("Load trained model (~393 MiB)  L")).clicked(){demo.load=true;}
                ui.small("Weights stay resident for subsequent inferences. The scene works before downloading them.");
            }
            if ui.add_enabled(demo.model_ready&&!demo.busy,egui::Button::new("Run inference  Space / I")).clicked(){demo.request=true;}
            if demo.busy||demo.model_loading {ui.spinner();}
            ui.label(&demo.status);
            ui.small(format!("Input revision {} · annotations are invalidated after edits",demo.revision));
            if let Some(r)=&demo.result {
                let inputs=demo.images.len();
                ui.separator();ui.heading("Cross-view RGB completion");
                ui.label(format!("Hidden-pixel PSNR: {:.2} dB · no references: {:.2} dB",r.rgb_score.psnr_db.unwrap_or(f64::INFINITY),r.monocular_score.psnr_db.unwrap_or(f64::INFINITY)));
                let width=(ui.available_width()/2.-10.).min(256.);
                ui.horizontal(|ui|{image(ui,&demo.textures[inputs],"Sparse target (10%)",width);image(ui,&demo.textures[inputs+1],"Predicted RGB",width);});
                ui.horizontal(|ui|{image(ui,&demo.textures[r.target],"Target: score only",width);image(ui,&demo.textures[inputs+2],"References disabled",width);});
                ui.small("Only originally hidden pixels contribute to PSNR, sRGB range 1. The reconstruction is not filled with target pixels.");
                ui.separator();ui.heading("Predicted camera calibration");
                ui.label(format!("Target {} relative to reference {}",r.target+1,r.reference+1));
                ui.label(format!("Focal fx / fy: {:.1} / {:.1} px at 256 × 256",r.focal[0]*256.,r.focal[1]*256.));
                if let Some(s)=&demo.camera_score {
                    ui.label(format!("Rotation error {:.2}° · direction error {:.2}°",s.rotation_degrees,s.translation_degrees.unwrap_or(f64::NAN)));
                    ui.label(format!("Focal error {:.2}%",100.*s.focal_relative_error));
                } else {ui.label("Pose errors unavailable for uploaded photographs.");}
                ui.small("Centered principal point. Translation direction has no metric scale. Camera prediction uses a separate dense RGB pair.");
                camera_plot(ui,r.rotation,demo.camera_truth.as_ref().map(|t|t.rotation));
                ui.separator();ui.heading("Coarse feature matches");
                ui.small("Mutual nearest patch descriptors, 16-pixel grid; at most 24 lines. These are predicted matches, not known correspondences.");
                ui.horizontal(|ui|{
                    let a=ui.image((demo.textures[r.target].id(),egui::vec2(width,width))).rect;
                    let b=ui.image((demo.textures[r.reference].id(),egui::vec2(width,width))).rect;
                    for m in &r.matches {ui.painter().line_segment([a.min+egui::vec2(m.target[0],m.target[1])*width/256.,b.min+egui::vec2(m.reference[0],m.reference[1])*width/256.],egui::Stroke::new(1_f32,egui::Color32::from_rgb(30,190,155)));}
                });
                ui.separator();ui.heading("Relative-improvement score");
                image(ui,&demo.textures[inputs+3],"Dark = low · yellow = high (display clamped 0–1)",width);
                ui.small("Separate full-target RI branch. This learned visibility-related score is not a calibrated probability or geometric visibility truth.");
            }
            ui.separator();ui.colored_label(egui::Color32::from_rgb(224,167,78),"Research demo: RGB remains blurry and camera calibration overfits. These are live predictions, not benchmark claims.");
        });
    });
    if let Ok(mut camera) = editor.single_mut() {
        let rect = viewport.available_rect_before_wrap();
        let scale = ctx.pixels_per_point();
        camera.viewport = Some(bevy::camera::Viewport {
            physical_position: UVec2::new(
                (rect.min.x * scale).round() as u32,
                (rect.min.y * scale).round() as u32,
            ),
            physical_size: UVec2::new(
                (rect.width() * scale).floor().max(1.) as u32,
                (rect.height() * scale).floor().max(1.) as u32,
            ),
            ..default()
        });
    }
    if demo.mode == Mode::Images {
        egui::CentralPanel::default().show_inside(&mut viewport, |ui| {
            ui.heading("Your scene, viewed from different cameras");
            if demo.images.is_empty() {
                ui.label(
                    "Upload images using the panel on the right. Nothing is sent to a server.",
                );
            }
            let width = (ui.available_width() / 2. - 15.).min(384.);
            egui::Grid::new("uploads").num_columns(2).show(ui, |ui| {
                for (i, img) in demo.images.iter().enumerate() {
                    image(ui, &demo.textures[i], &img.name, width);
                    if i % 2 == 1 {
                        ui.end_row();
                    }
                }
            });
        });
    }
    #[cfg(target_arch = "wasm32")]
    diagnostics(&demo, live.len());
}
fn camera_plot(ui: &mut egui::Ui, pred: Option<[[f64; 3]; 3]>, truth: Option<[[f64; 3]; 3]>) {
    let (rect, _) =
        ui.allocate_exact_size(egui::vec2(ui.available_width(), 130.), egui::Sense::hover());
    let center = rect.center();
    for (r, color) in [
        (truth, egui::Color32::from_rgb(30, 185, 145)),
        (pred, egui::Color32::from_rgb(230, 160, 60)),
    ] {
        if let Some(r) = r {
            for ((x, y), z) in r[0].into_iter().zip(r[1]).zip(r[2]) {
                ui.painter().arrow(
                    center,
                    egui::vec2((x - 0.45 * z) as f32, -(y + 0.3 * z) as f32) * 50.,
                    egui::Stroke::new(2_f32, color),
                );
            }
        }
    }
    ui.small("Orientation axes: amber prediction; green renderer truth when available.");
}
#[cfg(target_arch = "wasm32")]
fn diagnostics(demo: &Demo, cameras: usize) {
    use wasm_bindgen::JsValue;
    let v = serde_json::json!({"status":demo.status,"revision":demo.revision,"result_revision":demo.result.as_ref().map(|r|r.revision),"target":demo.target,"model_ready":demo.model_ready,"busy":demo.busy,"mode":format!("{:?}",demo.mode),"cameras":cameras,"psnr":demo.result.as_ref().and_then(|r|r.rgb_score.psnr_db),"matches":demo.result.as_ref().map(|r|r.matches.len()),"camera":demo.result.as_ref().map(|r|r.camera.clone()),"has_camera_truth":demo.camera_truth.is_some(),"camera_signature":demo.last_signature});
    if let Some(w) = web_sys::window()
        && let Ok(v) =
            serde::Serialize::serialize(&v, &serde_wasm_bindgen::Serializer::json_compatible())
    {
        let _ = js_sys::Reflect::set(&w, &JsValue::from_str("gekkoDemo"), &v);
    }
}
