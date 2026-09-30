//! TOML inputs and resolved snapshots. JSON is read only for historical artifacts.
use anyhow::{Context, Result};
use serde::{Serialize, de::DeserializeOwned};
use std::{fs, io::ErrorKind, path::Path};

pub fn read_config<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    toml::from_str(&text).with_context(|| format!("parse TOML config {}", path.display()))
}

pub fn config_toml(config: &impl Serialize) -> Result<String> {
    toml::to_string_pretty(config).context("serialize TOML config")
}

pub fn write_config(path: &Path, config: &impl Serialize) -> Result<()> {
    fs::write(path, config_toml(config)?).with_context(|| format!("write {}", path.display()))
}

/// New runs use TOML; old recorded runs/captures remain readable without rewriting provenance.
/// A malformed or unreadable TOML file must not silently fall back to an older JSON file.
pub fn read_config_snapshot<T: DeserializeOwned>(directory: &Path, stem: &str) -> Result<T> {
    let path = directory.join(format!("{stem}.toml"));
    match fs::read_to_string(&path) {
        Ok(text) => {
            toml::from_str(&text).with_context(|| format!("parse TOML config {}", path.display()))
        }
        Err(error) if error.kind() == ErrorKind::NotFound => {
            let legacy = directory.join(format!("{stem}.json"));
            let bytes = fs::read(&legacy)
                .with_context(|| format!("read historical config {}", legacy.display()))?;
            serde_json::from_slice(&bytes)
                .with_context(|| format!("parse historical config {}", legacy.display()))
        }
        Err(error) => Err(error).with_context(|| format!("read {}", path.display())),
    }
}
