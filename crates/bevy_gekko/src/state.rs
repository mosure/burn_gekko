use bevy::prelude::*;
use bevy_egui::egui;
use burn_gekko_inference::{InferenceOutput, RgbInput};
use burn_gekko_metrics::{calibration::CameraTarget, heads::CameraScore};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Scene,
    Images,
}

#[derive(Resource)]
pub struct Demo {
    pub mode: Mode,
    pub revision: u64,
    pub target: usize,
    pub model_ready: bool,
    pub model_loading: bool,
    pub busy: bool,
    pub status: String,
    pub model_root: String,
    pub images: Vec<RgbInput>,
    pub result: Option<InferenceOutput>,
    pub camera_truth: Option<CameraTarget>,
    pub camera_score: Option<CameraScore>,
    pub textures: Vec<egui::TextureHandle>,
    pub texture_revision: Option<u64>,
    pub pending: Option<Snapshot>,
    pub select: Option<usize>,
    pub place: bool,
    pub overview: bool,
    pub request: bool,
    pub regenerate: bool,
    pub upload: bool,
    pub load: bool,
    pub initialized: bool,
    pub auto_infer: bool,
    pub last_signature: Vec<f32>,
    pub scene_frames: usize,
}
impl Demo {
    pub fn new(model_root: String, auto_infer: bool) -> Self {
        Self {
            mode: Mode::Scene,
            revision: 0,
            target: 0,
            model_ready: false,
            model_loading: false,
            busy: false,
            status: "Explore the scene, then load the model to run inference.".into(),
            model_root,
            images: vec![],
            result: None,
            camera_truth: None,
            camera_score: None,
            textures: vec![],
            texture_revision: None,
            pending: None,
            select: None,
            place: false,
            overview: false,
            request: false,
            regenerate: false,
            upload: false,
            load: false,
            initialized: false,
            auto_infer,
            last_signature: vec![],
            scene_frames: 0,
        }
    }
    pub fn invalidate(&mut self, reason: &str) {
        self.revision += 1;
        self.result = None;
        self.camera_score = None;
        self.camera_truth = None;
        self.texture_revision = None;
        if self.pending.is_some() {
            self.busy = false;
        }
        self.pending = None;
        self.status = reason.into();
    }
}
pub struct Snapshot {
    pub revision: u64,
    pub images: Vec<Option<RgbInput>>,
    pub truth: Option<CameraTarget>,
}
#[derive(Component)]
pub struct CaptureStamp {
    pub revision: u64,
    pub view: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn edit_invalidates_pending_capture_but_preserves_worker_busy_state() {
        let mut demo = Demo::new("unused".into(), false);
        demo.busy = true;
        demo.invalidate("camera moved during prediction");
        assert!(demo.busy);
        assert_eq!(demo.revision, 1);
        demo.pending = Some(Snapshot {
            revision: 1,
            images: vec![None; 3],
            truth: None,
        });
        demo.invalidate("camera moved during capture");
        assert!(!demo.busy);
        assert!(demo.pending.is_none());
        assert_eq!(demo.revision, 2);
    }
}
