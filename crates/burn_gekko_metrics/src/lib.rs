//! Portable numerical contracts; reports and interactive demos use the same code.
pub mod calibration;
pub mod camera;
pub mod heads;
pub mod rgb;

#[doc(hidden)]
pub const SOURCES: &[(&str, &str)] = &[
    ("lib.rs", include_str!("lib.rs")),
    ("calibration.rs", include_str!("calibration.rs")),
    ("camera.rs", include_str!("camera.rs")),
    ("heads.rs", include_str!("heads.rs")),
    ("rgb.rs", include_str!("rgb.rs")),
];
#[doc(hidden)]
pub const MANIFEST: &str = include_str!("../Cargo.toml");
