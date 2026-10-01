# Changelog

## Unreleased

- Evaluator and report package manifests advance to 0.1.2; report and trainer
  require the new evaluator API. This prevents packaged builds from resolving
  the previously published evaluator without the shared output-head audit.
- Shared native raw-prediction audits, whole-room output-head uncertainty,
  full-cohort continuation selection and numerically checked runtime comparisons.
- Independent camera calibration and RGB reconstruction heads, with bounded
  cached-feature training, separate clipped optimizers, exact optimizer resume,
  native angular/focal/RGB PSNR evaluation and verified page/PDF visualizations.
- Pilot 12--14 feature preservation, renderer-supervised viewpoint training,
  spatial-detail decomposition, reference-count uncertainty and camera solver
  seed diagnostics. Failed accuracy and numerical qualification gates are retained.
- Fixed local correspondence readout, retaining hard-index controls and adding
  fractional coordinates for HPatches, ETH3D and known-transform evaluation.
- Native paired transfer/precision gates and teacher-power feature SNR in dB.
- Shared page/PDF metric explanations, percentage labels and complete-array
  verification of feature SNR. RGB PSNR remains specific to actual RGB outputs.
- A bounded Pilot 09 continuation using the unspent Pilot 08 command allowance.
- Native TUM RGB preparation and calibrated camera-motion evaluation, with
  deterministic essential RANSAC, explicit failed fits, per-sequence metrics and
  annotated pose-probe panels kept separate from learned camera heads.
- Registered Pilot 11 full-versus-tail encoder adaptation under a new, explicitly
  authorized 12-hour GPU-command ceiling; results remain separate from planned gates.
- Explicit TUM input grids through 512px, with unchanged original-pixel camera
  calibration and a native correspondence-replay qualification contract.
- Native training-prefix checks, matched continuation-parent validation and CUDA
  trace interval analysis; process trace coverage is not presented as SM occupancy.
- Shared selected-phase training/validation curves for the page and vector PDF,
  excluding parent and post-checkpoint updates.

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
