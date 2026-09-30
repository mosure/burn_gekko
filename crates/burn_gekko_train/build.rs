//! Preserve dependency-lock evidence both in the workspace and in registry archives.
use std::{env, fs, path::PathBuf};

fn main() {
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let lock = manifest
        .ancestors()
        .map(|dir| dir.join("Cargo.lock"))
        .find(|p| p.is_file());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap()).join("dependency-lock.txt");
    let bytes = match lock {
        Some(path) => {
            println!("cargo:rerun-if-changed={}", path.display());
            fs::read(path).expect("read dependency lockfile")
        }
        None => Vec::new(),
    };
    fs::write(out, bytes).expect("record dependency lockfile");
    println!("cargo:rerun-if-changed=build.rs");
}
