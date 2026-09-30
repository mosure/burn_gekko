# Releasing the Rust crates

The main workspace contains `burn_vjepa`, `burn_gekko`, `burn_gekko_data`,
`burn_gekko_eval`, `burn_gekko_train` and `burn_gekko_report`. The isolated renderer
is `burn_gekko_capture` in `tools/zeroverse_capture`.

1. Update versions and local dependency version requirements together. Keep
   published Bevy generator pins and both lockfiles explicit.
2. Run `python3 tools/interop/verify_encoder_import.py`, workspace formatting,
   strict Clippy, and `cargo test --workspace --locked`. Check CUDA compilation
   separately; renderer validation includes `--identity` and `--dry-run` only.
3. Inspect `cargo package --workspace --list` and run
   `cargo publish --workspace --locked --dry-run`. Archive allowlists exclude
   datasets, weights, benchmark assets, historical tools and generated pages.
4. Commit reviewed scope, push, and wait for the exact commit's CI checks.
5. Publish the workspace in dependency order (`cargo publish --workspace --locked`
   resolves the order). Publish the isolated renderer after `burn_gekko_data` is
   visible: `cargo publish --manifest-path tools/zeroverse_capture/Cargo.toml --locked`.
6. Verify every registry version, archive checksum and clean `.cargo_vcs_info.json`
   commit against the release commit. Tag only the fully verified release.

CI checks the CPU contracts, CUDA compilation, package archive build, Rust docs
and renderer compilation. It runs no training or large data generation.
Page deployment remains a separate, disabled workflow requiring a reviewed bundle.

## Source identity

Each library exposes an internal compile-time inventory of its own sources.
The trainer fingerprints these inventories, manifests and the captured Cargo.lock
without reaching into sibling package directories. Its build script records the
nearest lockfile (workspace lock in a checkout, archive lock in a packaged build).
An unlocked build cannot train or resume. Use `--locked` for reproducibility.
Changes in package layout or Cargo's normalized manifests deliberately change this
identity; a workspace binary and a registry binary are not interchangeable for
exact optimizer resume. Historical sealed artifacts in `.data/` remain unchanged.

The encoder retains all six original import hashes and an explicit local model
adaptation record. Dual-license texts come from the original author's repository;
the separate Meta V-JEPA weights notice is included without bundling any weights.
