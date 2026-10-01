//! Independent calibration/RGB head fitting on a pinned frozen foundation cache.
mod config;
mod data;
pub mod evaluation;
mod runner;
pub use config::HeadTrainConfig;
pub use runner::run;
