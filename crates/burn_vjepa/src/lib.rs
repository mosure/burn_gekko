//! Minimal, pinned V-JEPA encoder import. See UPSTREAM.json and UPSTREAM.md.
#![allow(clippy::too_many_arguments, clippy::type_complexity)]
mod config;
#[allow(clippy::needless_range_loop)] // Preserve the pinned upstream file byte for byte.
mod model;
mod package;
mod positional;
#[allow(clippy::chunks_exact_to_as_chunks)] // Rust 1.98 lint on pinned upstream code.
mod safetensors_io;
mod sparse_patchify;
mod tokens;

#[doc(hidden)]
pub mod provenance;
pub use config::*;
pub use model::*;
pub use package::*;
pub use positional::*;
pub use safetensors_io::*;
pub use sparse_patchify::*;
pub use tokens::*;
