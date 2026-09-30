//! CPU scoring of immutable Burn prediction exports. No Python or GPU required.
//! Input geometry is evaluated after inference; it is never passed to a model here.
pub mod benchmark;
pub mod camera;
pub mod camera_export;
pub mod contrasts;
pub mod efficiency;
pub mod metrics;
mod npy;
pub mod process_activity;
#[doc(hidden)]
pub mod provenance;
pub mod ranking;
pub mod reference_count;
pub mod schema;
pub mod statistics;
pub mod training;
pub mod training_export;
pub mod warp;
