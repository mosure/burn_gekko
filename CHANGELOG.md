# Changelog

## 0.1.0

Initial research release of `burn_gekko`, `burn_vjepa`, `burn_gekko_data`,
`burn_gekko_train`, `burn_gekko_eval`, `burn_gekko_report` and `burn_gekko_capture`.

- Sparse per-view V-JEPA 2.1 encoding, multi-reference fusion and latent completion.
- Fixed-teacher training, staged encoder unfreezing, known-transform descriptor
  supervision, checkpoint ancestry and exact optimizer resume.
- Published Zeroverse room capture with immutable disk caches and TOML recipes.
- Native Rust completion, co-visibility, correspondence, camera and efficiency
  metrics; verified single-run project pages, figures and PDF papers.
- Package-local source provenance and self-contained trainer test fixtures make
  registry archives buildable independently of this checkout.

Crate paths/imports now use `burn_` prefixes; the encoder fork is `burn_vjepa`.
CLI names (`gekko`, `gekko-eval`, `gekko-report`, `gekko_zeroverse_capture`) remain.
This release changes exact-resume source identity. Resume historical optimizer
states with their sealed original binary, or start an audited weights-only phase
with reset optimizers. Existing model parameter layouts are unchanged.

Research qualification remains incomplete: SOTA and sharp RGB reconstruction are
not established; the current latent checkpoint has no trained camera head.
No datasets, model weights, benchmark assets or rendered study bundles are shipped.
