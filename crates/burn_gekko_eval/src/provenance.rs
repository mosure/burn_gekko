//! Build provenance for checkpoint identity; no cross-crate filesystem access.

#[doc(hidden)]
pub const SOURCES: &[(&str, &str)] = &[
    ("heads.rs", include_str!("heads.rs")),
    (
        "adaptation/continuation/completion.rs",
        include_str!("adaptation/continuation/completion.rs"),
    ),
    ("adaptation.rs", include_str!("adaptation.rs")),
    (
        "adaptation/continuation.rs",
        include_str!("adaptation/continuation.rs"),
    ),
    (
        "adaptation/continuation/forecast.rs",
        include_str!("adaptation/continuation/forecast.rs"),
    ),
    (
        "adaptation/view_geometry.rs",
        include_str!("adaptation/view_geometry.rs"),
    ),
    (
        "adaptation/preservation.rs",
        include_str!("adaptation/preservation.rs"),
    ),
    ("refinement.rs", include_str!("refinement.rs")),
    ("benchmark.rs", include_str!("benchmark.rs")),
    ("camera.rs", include_str!("camera.rs")),
    ("camera_export.rs", include_str!("camera_export.rs")),
    ("contrasts.rs", include_str!("contrasts.rs")),
    ("dispatch.rs", include_str!("dispatch.rs")),
    ("dispatch/warm.rs", include_str!("dispatch/warm.rs")),
    ("efficiency.rs", include_str!("efficiency.rs")),
    ("lib.rs", include_str!("lib.rs")),
    ("latent_detail.rs", include_str!("latent_detail.rs")),
    ("latent_replay.rs", include_str!("latent_replay.rs")),
    ("main.rs", include_str!("main.rs")),
    ("metrics.rs", include_str!("metrics.rs")),
    ("metrics/detail.rs", include_str!("metrics/detail.rs")),
    ("npy.rs", include_str!("npy.rs")),
    ("pose/mod.rs", include_str!("pose/mod.rs")),
    ("pose/five_point.rs", include_str!("pose/five_point.rs")),
    ("pose/localization.rs", include_str!("pose/localization.rs")),
    (
        "pose/localization/publication.rs",
        include_str!("pose/localization/publication.rs"),
    ),
    ("pose/replay.rs", include_str!("pose/replay.rs")),
    ("pose/solver.rs", include_str!("pose/solver.rs")),
    ("pose/stability.rs", include_str!("pose/stability.rs")),
    ("pose/synthetic.rs", include_str!("pose/synthetic.rs")),
    (
        "pose/synthetic/publication.rs",
        include_str!("pose/synthetic/publication.rs"),
    ),
    (
        "pose/synthetic_comparison.rs",
        include_str!("pose/synthetic_comparison.rs"),
    ),
    ("pose/benchmark.rs", include_str!("pose/benchmark.rs")),
    ("process_activity.rs", include_str!("process_activity.rs")),
    ("provenance.rs", include_str!("provenance.rs")),
    ("ranking.rs", include_str!("ranking.rs")),
    ("replay.rs", include_str!("replay.rs")),
    ("reference_count.rs", include_str!("reference_count.rs")),
    ("schema.rs", include_str!("schema.rs")),
    ("statistics.rs", include_str!("statistics.rs")),
    ("target_audit.rs", include_str!("target_audit.rs")),
    ("training.rs", include_str!("training.rs")),
    ("training_export.rs", include_str!("training_export.rs")),
    ("warp.rs", include_str!("warp.rs")),
];

#[doc(hidden)]
pub const MANIFEST: &str = include_str!("../Cargo.toml");
