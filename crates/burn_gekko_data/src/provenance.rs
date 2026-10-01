//! Build provenance for checkpoint identity; no cross-crate filesystem access.

#[doc(hidden)]
pub const SOURCES: &[(&str, &str)] = &[
    ("head_cache.rs", include_str!("head_cache.rs")),
    ("cache.rs", include_str!("cache.rs")),
    ("chunk.rs", include_str!("chunk.rs")),
    ("config.rs", include_str!("config.rs")),
    ("config_file.rs", include_str!("config_file.rs")),
    ("geometry.rs", include_str!("geometry.rs")),
    ("view_targets.rs", include_str!("view_targets.rs")),
    (
        "view_targets/cache.rs",
        include_str!("view_targets/cache.rs"),
    ),
    ("image_transform.rs", include_str!("image_transform.rs")),
    ("lib.rs", include_str!("lib.rs")),
    ("provenance.rs", include_str!("provenance.rs")),
    ("real_views.rs", include_str!("real_views.rs")),
    ("tum.rs", include_str!("tum.rs")),
];

#[doc(hidden)]
pub const MANIFEST: &str = include_str!("../Cargo.toml");
