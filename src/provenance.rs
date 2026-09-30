//! Build provenance for checkpoint identity; no cross-crate filesystem access.

#[doc(hidden)]
pub const SOURCES: &[(&str, &str)] = &[
    ("encoder.rs", include_str!("encoder.rs")),
    ("fusion/attention.rs", include_str!("fusion/attention.rs")),
    ("fusion/decoder.rs", include_str!("fusion/decoder.rs")),
    ("fusion/mod.rs", include_str!("fusion/mod.rs")),
    ("fusion/rotary.rs", include_str!("fusion/rotary.rs")),
    ("fusion/tests.rs", include_str!("fusion/tests.rs")),
    ("heads/appearance.rs", include_str!("heads/appearance.rs")),
    ("heads/matching.rs", include_str!("heads/matching.rs")),
    ("heads/mod.rs", include_str!("heads/mod.rs")),
    ("heads/spatial.rs", include_str!("heads/spatial.rs")),
    ("heads/transport.rs", include_str!("heads/transport.rs")),
    ("lib.rs", include_str!("lib.rs")),
    ("models/latent.rs", include_str!("models/latent.rs")),
    (
        "models/legacy_hybrid.rs",
        include_str!("models/legacy_hybrid.rs"),
    ),
    (
        "models/legacy_released.rs",
        include_str!("models/legacy_released.rs"),
    ),
    ("models/mod.rs", include_str!("models/mod.rs")),
    (
        "models/reconstruction.rs",
        include_str!("models/reconstruction.rs"),
    ),
    (
        "objectives/correspondence.rs",
        include_str!("objectives/correspondence.rs"),
    ),
    ("objectives/fusion.rs", include_str!("objectives/fusion.rs")),
    ("objectives/mod.rs", include_str!("objectives/mod.rs")),
    ("objectives/rgb.rs", include_str!("objectives/rgb.rs")),
    ("provenance.rs", include_str!("provenance.rs")),
    ("sparse/curriculum.rs", include_str!("sparse/curriculum.rs")),
    ("sparse/masking.rs", include_str!("sparse/masking.rs")),
    ("sparse/mod.rs", include_str!("sparse/mod.rs")),
    ("tensor.rs", include_str!("tensor.rs")),
];

#[doc(hidden)]
pub const MANIFEST: &str = include_str!("../Cargo.toml");
