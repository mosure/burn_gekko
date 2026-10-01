#!/usr/bin/env bash
# Build both Rust WASM modules and assemble the static project/paper/demo site.
set -euo pipefail
demo_repo=$(cd "$(dirname "$0")/../.." && pwd)
cd "$demo_repo"
demo_output=${1:-.data/web}
demo_models=${2:-}
demo_bindgen=${WASM_BINDGEN:-wasm-bindgen}
if [[ $("$demo_bindgen" --version) != "wasm-bindgen 0.2.129" ]]; then
  echo "Install wasm-bindgen-cli 0.2.129 (the version pinned by both Cargo.lock files)." >&2
  exit 1
fi
mkdir -p "$demo_output/demo/viewer" "$demo_output/demo/inference"
cp -a www/. "$demo_output/"
cargo rustc --locked --release -p burn_gekko_inference --lib --crate-type cdylib --target wasm32-unknown-unknown --no-default-features --features web
"$demo_bindgen" --target web --out-dir "$demo_output/demo/inference" "${CARGO_TARGET_DIR:-target}/wasm32-unknown-unknown/release/burn_gekko_inference.wasm"
cargo build --locked --release --manifest-path crates/bevy_gekko/Cargo.toml --target wasm32-unknown-unknown --no-default-features --features web
"$demo_bindgen" --target web --out-dir "$demo_output/demo/viewer" "${CARGO_TARGET_DIR:-crates/bevy_gekko/target}/wasm32-unknown-unknown/release/bevy_gekko.wasm"
if [[ -n "$demo_models" ]]; then
  test -f "$demo_models/manifest.toml"
  mkdir -p "$demo_output/demo/models"
  cp -a "$demo_models/." "$demo_output/demo/models/"
fi
echo "Static site ready in $demo_output. Serve over localhost or HTTPS."
