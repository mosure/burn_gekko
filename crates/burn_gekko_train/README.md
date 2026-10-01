# burn_gekko_train

Bounded training and Burn prediction export. `training::latent` is the primary
fixed-teacher pipeline; `training::reconstruction` retains RGB diagnostics. Other
trainers are historical research implementations. Configuration/ancestry and long
training loops have separate files. `data` owns batching and dataset adapters;
`evaluation` owns model loading, readouts and exports. Scalar scoring belongs to
`burn_gekko_eval`, publication to `burn_gekko_report`.

`training::heads` fits the independent calibration and RGB heads from scratch on
an immutable feature cache. `head_cache --config <toml> --output <directory>`
runs the frozen model once; `head_pilot --config <toml> --run <directory>` trains
small heads on CPU. `--resume <checkpoint>` restores both optimizers and sampling;
`--stop-after <step>` exercises bounded interruption. Camera and RGB use separate
AdamW optimizers, warmup and global-norm clipping. Evaluation labels are never
model inputs. `head_cache` uses CUDA only when compiled with the CUDA feature;
`head_pilot` always uses CPU. See the [head study](../../docs/studies/head-stability-15.md).

`evaluation::refinement` shares the fixed local correspondence operator across
HPatches, ETH3D and known-transform audits. A single score readback supplies both
the original hard match and the fractional centroid. Pair, same-image and encoder
controls receive identical operators; renderer geometry is absent from inference.

`evaluation::eth3d::canonical` retains the broad export calculation order and
shares its score arrays between hard and local matches. Set
`canonical_spatial_readouts = true` in both the export and sealed selection TOML
when using this path. It retains complete-population and throughput checks;
it does not certify the optimized exporter's backend parity.

`training::latent::EncoderPreservationConfig` optionally adds a pinned frozen
own encoder as a training-only final-feature target. The objective lives in
`burn_gekko::objectives::preservation`; its online routes and source loading live
in `training/latent/preservation.rs`. It preserves sparse target and full-view
feature coordinates, records anchor ancestry, and retains the original anchor
across optimizer resume. Inference has no additional encoder or parameters.

The `gekko` binary exposes `capture`, `verify-dataset`, `audit-geometry`,
`train-latent` and `assess-latent`, plus historical preflight commands. CUDA is
explicit (`--features cuda`, then `--backend cuda`); CPU is the default. Runtime
outputs stay in `.data/`. Run commands from the workspace root.

Historic binary/API names remain thin compatibility entry points. Exact optimizer
resume verifies source/config/backend and checkpoint hashes. Moving source does
not waive those checks: use the sealed old binary, or explicitly start a weights-only
phase with recorded ancestry and reset optimizers. Noncommercial pretrained
Gekko weights cannot initialize or supervise the primary latent trainer.

## Optional actual-view geometry objective

The `[view_geometry]` TOML block adds bidirectional correspondence NLL between the
dense target and one existing reference view, rotating references each update.
Its `cache`, `weight` and `temperature` fields are explicit; an absent block keeps
the existing training objective. Geometry only supplies detached labels. The
encoder and fusion trunk see RGB, and no camera/depth inputs or new inference
parameters are added. Existing latent, homography, RI and feature-preservation
objectives remain independently configured.

`view_geometry_export --config <toml> --output <directory>` evaluates actual camera
changes on the fixed validation cohort, preserving hard/refined, pair-conditioned,
same-image and encoder readouts. Distances use input pixels; this is a synthetic
development diagnostic, distinct from HPatches, ETH3D and calibrated pose transfer.
