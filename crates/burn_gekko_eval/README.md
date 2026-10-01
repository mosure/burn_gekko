# burn_gekko_eval

Native CPU scoring of immutable Burn prediction exports. No Python or GPU backend.

- `metrics`: masked completion, teacher-power feature SNR, RGB PSNR and correspondence errors.
- `refinement`: fixed local probability centroids in fractional patch coordinates.
- `ranking`: tie-aware AP/AUROC.
- `camera`: SO(3), signed translation direction, normalized intrinsics and pose AUC.
- `pose`: calibrated eight-point RANSAC, failure-inclusive real-image motion scoring
  and same-checkpoint transfer gates; separate from learned camera heads.
- `benchmark`: complete ETH3D/HPatches protocols and deterministic visual examples.
- `statistics`: cluster bootstrap with a specified portable RNG.
- `contrasts`: paired readout differences with declared transfer/precision gates.
- `efficiency`: observed board energy and update timing, with coverage/shared-load limits.
- `training`: exposure coverage, scalar windows and verified encoder gradient stages.
- `training_export`: checkpoint/command-bound training summaries.
- `replay`: matched update-prefix, sample and loss-trajectory verification.
- `adaptation`: matched tail/full encoder screen selection with validation-only gates.
- `adaptation::preservation`: registered three-arm feature-preservation selection,
  including matched source/teacher identities and frozen-anchor probes.
- `process_activity`: shared-GPU process counters with unavailable values preserved.
- `dispatch`: overlapping CUDA trace intervals, launch durations and API synchronization.
- `schema`: extensible checkpoint-bound capability records consumed by publication.

`gekko-eval score --config <TOML>` scores one checkpoint. `gekko-eval camera
--config <TOML>` scores post-inference `CameraRecord` JSONL with pinned record and
provenance hashes. Camera records must declare their coordinate frame and input
contract; they are not an input API for the training model.

`gekko-eval prepare-tum --config <TOML>` caches registered Freiburg 3 pairs with
RGB and camera labels in separate manifests. `gekko-eval pose --config <TOML>`
fits and scores their post-inference matches, preserving every failure and exclusion.
`gekko-eval pose-replay --config <TOML>` checks exact hard matches and declared
fractional-coordinate tolerance when qualifying a modified exporter.

`gekko-eval training --config <TOML>` summarizes a completed training command.
`gekko-eval activity --config <TOML>` scores an immutable NVIDIA pmon log. Both
record their source hashes and keep optimization evidence separate from accuracy.
`gekko-eval dispatch --config <TOML>` analyzes pinned Nsight JSON GPU/API traces
with explicit nanosecond units. It merges overlapping intervals and reports
process trace coverage, not device occupancy or a causal bottleneck diagnosis.

`gekko-eval select-preservation --config <TOML>` verifies the fixed 2,048-update
tail/full/preserved screen before applying its completion and geometry gates.
Its comparisons belong to internal controlled-study records; generated project
pages still present one training run/checkpoint.

See [metric/protocol definitions](../../docs/native-evaluation.md). Add numeric
oracles and leakage/provenance contracts here when introducing new heads.

## Renderer-supervised screen

`prepare-view-targets` runs CPU-only geometry preprocessing from a TOML config.
`audit-view-targets` checks the immutable cache and reports room/pair coverage,
valid-query counts and empty pairs separately for every split.
`select-view-geometry` checks a registered 384-update matched screen, including
recipe, sample/mask/query populations, checkpoint and anchor integrity. The
candidate must retain latent completion within 1%, keep references useful, reduce
actual-view pixel error by at least 5%, and retain within-8-pixel accuracy. Extra
decoder work is measured, so matched updates are not described as matched compute.
External benchmarks are evaluated after this synthetic decision is fixed.
## Completion and camera diagnostics

`gekko-eval synthetic-pose --config <TOML>` scores dense view-audit exports on CPU
over a fixed solver-seed panel. All RGB mutual matches enter the solver; renderer
visibility does not select correspondences. Each view uses its own known focal
length, Bevy camera axes are converted explicitly, and failed fits stay in the
denominator. This calibrated synthetic retention probe is separate from learned
camera-head accuracy and external transfer. See the
[continuation protocol](../../docs/studies/pilot-16-continuation.md).

`compare-synthetic-pose` binds two completed reports to the same room/solver
protocol and computes paired room-bootstrap intervals for angular error and
recall. It averages repeated solver seeds within a room before resampling;
solver repeats do not increase the independent sample count. Comparisons stay
in internal study records and do not replace single-run publication metrics.

`gekko-eval latent-detail --config <TOML>` reads complete, pinned latent exports
on CPU. It separates channel-mean bias from spatial error, measures hidden-token
neighbor differences, and reports teacher-assisted amplitude/variance oracles
as diagnostics only. It writes a checkpoint-bound capability alongside the full
array hashes and per-target measurements.

`gekko-eval latent-replay --config <TOML>` verifies a declared assessment prefix
after exporter changes. It binds model, teacher, dataset, masks and reference
count, compares all target/cross/monocular arrays, and retains failed numerical
checks without changing tolerances.

`gekko-eval pose-stability --config <TOML>` repeats a pinned development pose
export with an explicit solver-seed panel. All points, methods and thresholds
stay fixed; the original seed must replay. Its ranges describe solver randomness,
not training-seed uncertainty or a new generalization result. It also writes a
single-checkpoint capability for publication. Reference-count scoring reports
paired room-bootstrap intervals so correlated target views do not inflate the
uncertainty population.
