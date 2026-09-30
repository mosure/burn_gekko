//! Build provenance for checkpoint identity; no cross-crate filesystem access.

#[doc(hidden)]
pub const SOURCES: &[(&str, &str)] = &[
    ("config.rs", include_str!("config.rs")),
    ("lib.rs", include_str!("lib.rs")),
    ("model.rs", include_str!("model.rs")),
    ("package.rs", include_str!("package.rs")),
    ("positional.rs", include_str!("positional.rs")),
    ("provenance.rs", include_str!("provenance.rs")),
    ("safetensors_io.rs", include_str!("safetensors_io.rs")),
    ("sparse_patchify.rs", include_str!("sparse_patchify.rs")),
    ("tokens.rs", include_str!("tokens.rs")),
];

#[doc(hidden)]
pub const MANIFEST: &str = include_str!("../Cargo.toml");

/// Original upstream file hashes and revision.
pub const UPSTREAM: &str = include_str!("../UPSTREAM.json");

/// License notice retained with checkpoints using the audited V-JEPA weights.
pub const WEIGHTS_LICENSE: &str = include_str!("../VJEPA21_WEIGHTS_LICENSE");
