//! CPU scoring of immutable Burn prediction exports. No Python or GPU required.
//! Input geometry is evaluated after inference; it is never passed to a model here.
pub mod adaptation;
pub mod benchmark;
pub mod camera;
pub mod camera_export;
pub mod contrasts;
pub mod dispatch;
pub mod efficiency;
pub mod heads;
pub mod latent_detail;
pub mod latent_replay;
pub mod metrics;
mod npy;
pub mod pose;
pub mod process_activity;
#[doc(hidden)]
pub mod provenance;
pub mod ranking;
pub mod reference_count;
pub mod refinement;
pub mod replay;
pub mod schema;
pub mod statistics;
pub mod target_audit;
pub mod training;
pub mod training_export;
pub mod warp;
