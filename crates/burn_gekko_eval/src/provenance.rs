//! Build provenance for checkpoint identity; no cross-crate filesystem access.

#[doc(hidden)]
pub const SOURCES: &[(&str, &str)] = &[
    ("benchmark.rs", include_str!("benchmark.rs")),
    ("camera.rs", include_str!("camera.rs")),
    ("camera_export.rs", include_str!("camera_export.rs")),
    ("contrasts.rs", include_str!("contrasts.rs")),
    ("efficiency.rs", include_str!("efficiency.rs")),
    ("lib.rs", include_str!("lib.rs")),
    ("main.rs", include_str!("main.rs")),
    ("metrics.rs", include_str!("metrics.rs")),
    ("npy.rs", include_str!("npy.rs")),
    ("process_activity.rs", include_str!("process_activity.rs")),
    ("provenance.rs", include_str!("provenance.rs")),
    ("ranking.rs", include_str!("ranking.rs")),
    ("reference_count.rs", include_str!("reference_count.rs")),
    ("schema.rs", include_str!("schema.rs")),
    ("statistics.rs", include_str!("statistics.rs")),
    ("training.rs", include_str!("training.rs")),
    ("training_export.rs", include_str!("training_export.rs")),
    ("warp.rs", include_str!("warp.rs")),
];

#[doc(hidden)]
pub const MANIFEST: &str = include_str!("../Cargo.toml");
