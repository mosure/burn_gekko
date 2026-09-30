//! Build provenance for checkpoint identity; no cross-crate filesystem access.

#[doc(hidden)]
pub const SOURCES: &[(&str, &str)] = &[
    ("cache.rs", include_str!("cache.rs")),
    ("chunk.rs", include_str!("chunk.rs")),
    ("config.rs", include_str!("config.rs")),
    ("config_file.rs", include_str!("config_file.rs")),
    ("geometry.rs", include_str!("geometry.rs")),
    ("image_transform.rs", include_str!("image_transform.rs")),
    ("lib.rs", include_str!("lib.rs")),
    ("provenance.rs", include_str!("provenance.rs")),
];

#[doc(hidden)]
pub const MANIFEST: &str = include_str!("../Cargo.toml");
