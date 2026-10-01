//! The same inference contract on native and WASM, without a filesystem requirement.
pub mod bundle;
pub mod pixels;
pub mod runtime;
pub use bundle::{Bundle, ModelFile};
pub use pixels::RgbInput;
pub use runtime::{Inference, InferenceOutput, Match, Request};
#[cfg(all(feature = "web", target_arch = "wasm32"))]
mod web;
