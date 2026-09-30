//! Build provenance for checkpoint identity; no cross-crate filesystem access.

#[doc(hidden)]
pub const SOURCES: &[(&str, &str)] = &[
    ("bin/e2e_pilot.rs", include_str!("bin/e2e_pilot.rs")),
    ("bin/eth3d_export.rs", include_str!("bin/eth3d_export.rs")),
    (
        "bin/hpatches_export.rs",
        include_str!("bin/hpatches_export.rs"),
    ),
    ("bin/hybrid_pilot.rs", include_str!("bin/hybrid_pilot.rs")),
    ("bin/latent_assess.rs", include_str!("bin/latent_assess.rs")),
    ("bin/latent_pilot.rs", include_str!("bin/latent_pilot.rs")),
    (
        "bin/rgb_upload_bench.rs",
        include_str!("bin/rgb_upload_bench.rs"),
    ),
    (
        "bin/transport_pilot.rs",
        include_str!("bin/transport_pilot.rs"),
    ),
    ("data/augmentation.rs", include_str!("data/augmentation.rs")),
    ("data/batch.rs", include_str!("data/batch.rs")),
    ("data/encoding.rs", include_str!("data/encoding.rs")),
    ("data/mod.rs", include_str!("data/mod.rs")),
    (
        "evaluation/assessment.rs",
        include_str!("evaluation/assessment.rs"),
    ),
    (
        "evaluation/correspondence.rs",
        include_str!("evaluation/correspondence.rs"),
    ),
    (
        "evaluation/encoder.rs",
        include_str!("evaluation/encoder.rs"),
    ),
    (
        "evaluation/equivariance.rs",
        include_str!("evaluation/equivariance.rs"),
    ),
    ("evaluation/eth3d.rs", include_str!("evaluation/eth3d.rs")),
    ("evaluation/fusion.rs", include_str!("evaluation/fusion.rs")),
    (
        "evaluation/hpatches.rs",
        include_str!("evaluation/hpatches.rs"),
    ),
    ("evaluation/latent.rs", include_str!("evaluation/latent.rs")),
    ("evaluation/mod.rs", include_str!("evaluation/mod.rs")),
    ("evaluation/rgb.rs", include_str!("evaluation/rgb.rs")),
    ("lib.rs", include_str!("lib.rs")),
    ("main.rs", include_str!("main.rs")),
    ("provenance.rs", include_str!("provenance.rs")),
    ("training/hybrid.rs", include_str!("training/hybrid.rs")),
    (
        "training/latent/config.rs",
        include_str!("training/latent/config.rs"),
    ),
    (
        "training/latent/equivariance.rs",
        include_str!("training/latent/equivariance.rs"),
    ),
    ("training/latent.rs", include_str!("training/latent.rs")),
    ("training/mod.rs", include_str!("training/mod.rs")),
    ("training/pilot.rs", include_str!("training/pilot.rs")),
    (
        "training/preflight.rs",
        include_str!("training/preflight.rs"),
    ),
    (
        "training/reconstruction/config.rs",
        include_str!("training/reconstruction/config.rs"),
    ),
    (
        "training/reconstruction/runner.rs",
        include_str!("training/reconstruction/runner.rs"),
    ),
    (
        "training/reconstruction.rs",
        include_str!("training/reconstruction.rs"),
    ),
    (
        "training/transport.rs",
        include_str!("training/transport.rs"),
    ),
];

#[doc(hidden)]
pub const MANIFEST: &str = include_str!("../Cargo.toml");

const DEPENDENCY_LOCK: &str = include_str!(concat!(env!("OUT_DIR"), "/dependency-lock.txt"));
const BUILD_SCRIPT: &str = include_str!("../build.rs");

/// Exact-resume identity of this trainer and its source-bearing dependencies.
/// Packaged builds intentionally differ from older workspace builds.
pub(crate) fn identity() -> anyhow::Result<String> {
    anyhow::ensure!(
        !DEPENDENCY_LOCK.is_empty(),
        "training requires a Cargo.lock-backed build"
    );
    burn_gekko_data::fingerprint(&(
        "burn-gekko-source-v2",
        (SOURCES, MANIFEST, BUILD_SCRIPT, DEPENDENCY_LOCK),
        (
            burn_gekko::provenance::SOURCES,
            burn_gekko::provenance::MANIFEST,
        ),
        (
            burn_vjepa::provenance::SOURCES,
            burn_vjepa::provenance::MANIFEST,
            burn_vjepa::provenance::UPSTREAM,
        ),
        (
            burn_gekko_data::provenance::SOURCES,
            burn_gekko_data::provenance::MANIFEST,
        ),
        (
            burn_gekko_eval::provenance::SOURCES,
            burn_gekko_eval::provenance::MANIFEST,
        ),
    ))
}
