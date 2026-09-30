//! Burn models for sparse-view latent completion and multi-view fusion.
//! Training, datasets, scoring and publication live in separate workspace crates.
pub mod encoder;
pub mod fusion;
pub mod heads;
pub mod models;
pub mod objectives;
#[doc(hidden)]
pub mod provenance;
pub mod sparse;
pub mod tensor;
// Stable model names retained for checkpoint consumers.
pub use fusion::decoder as model;
pub(crate) use fusion::rotary;
pub use heads::{appearance, matching, transport};
pub use models::{
    latent, legacy_hybrid as hybrid, legacy_released as released, reconstruction as e2e,
};
pub use objectives::{fusion as fusion_objective, rgb as loss};
pub use sparse::{curriculum, masking};
