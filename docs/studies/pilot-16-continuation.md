# Pilot 16: sustained view supervision and camera retention

Status: **baseline diagnostics complete; continuation training pending**.
No new foundation weights were trained. The baseline used 76.462 of the carried
172.743 GPU-command seconds, leaving **96.281 seconds** of the existing allowance.
Renewal for the matched continuation is pending. CPU preprocessing and solver
analysis do not consume that allowance. Training must not start against a default
or inferred 12-hour ledger.

## Baseline result

The geometry-trained trunk recovers synthetic camera motion better than its
original matched control. All values below use the same 32 reused validation
rooms, eight fixed solver seeds, 2,048 trials and failure-inclusive scoring.
This is a retrospective diagnostic of existing Pilot 13 weights.

| RGB descriptor | Original matched control: mean AUC@10 | Geometry-trained parent: mean AUC@10 |
| --- | ---: | ---: |
| Pair-conditioned fusion | 17.20% | **21.04%** |
| Same-image fusion | 16.22% | 15.79% |
| Encoder only | 14.56% | 13.53% |

The pair-conditioned parent beats the original control in seven of eight solver
seeds. Its AUC@10 ranges from 16.62% to 26.64%; the control ranges from 12.45% to
23.17%. Those ranges describe solver randomness, not confidence intervals or
training-seed variability.

Native paired analysis averages solver repeats within each room and bootstraps
the 32 rooms 10,000 times. Positive differences below favor the geometry parent:

| Comparison | Mean angular-error reduction | Paired 95% room interval |
| --- | ---: | ---: |
| Joint pose error, parent versus original control | **4.29°** | **1.00° to 7.74°** |
| Signed translation error, parent versus original control | **4.57°** | **1.04° to 8.31°** |
| Joint pose error, parent pair versus its same-image control | **9.01°** | **2.79° to 17.81°** |
| Joint pose error, parent pair versus its encoder control | **7.02°** | **2.66° to 11.84°** |

Joint pose error is max(rotation error, signed translation-direction error).
The within-10-degree recall gain versus the original control is 6.64 percentage
points with interval [0.00, 13.67]; that interval touches zero. These are intervals
for angular error and recall, not AUC. All eligible failures stay in the denominator.

The synthetic benefit contrasts with Pilot 13/14's mixed TUM camera transfer.
This supports investigating domain transfer and sustained supervision; it does
not establish an external camera improvement, sharper completion or SotA.
No old checkpoint-selection decision was changed after these measurements.

The full CPU cache audit covers **49,152 training pairs and 768 validation pairs**,
with zero empty pairs. Mean valid patch queries per pair are 119.72 and 124.91;
the minima are 2 and 16, so low-overlap training pairs remain visible. Dataset
hashes, geometry targets and split membership are checked by Rust. The cache is
115,015,680 bytes plus its manifest, and generation uses no GPU.

Both new CUDA exports reproduce every legacy correspondence coordinate/error
and method summary exactly. Commands took 38.755 and 37.707 seconds, respectively.
Shared GPU process observations are retained: GNOME Shell reported a mean 10.63%
SM counter across eight observations; remote-desktop counters were unavailable,
not zero. This short sequential inference audit does not qualify steady-state
training efficiency or SM occupancy.

Artifacts live under `.data/pilot-16/`: `targets-audit.json`, `parent-pose.json`,
`prior-control-pose.json`, `pose-contrast-*.json`, `legacy-replay.json`, command
ledgers/telemetry and pinned binaries. Inputs/configs are TOML under
`configs/data`, `configs/experiments` and `configs/eval`. The registered protocol
snapshot predates the baseline exports; report-only analysis adds no training
or model selection.

Local verification passed: the 168-test workspace suite before the additional
comparison test, all 34 final evaluation unit tests, the real dataset's three
comparison replays, and rejection of missing-room and missing-seed reports.
CPU/workspace and CUDA Clippy pass with warnings denied; formatting and the
original encoder-import audit pass. The pipeline integration test exercises
cache integrity, dense export, CPU pose scoring and a tampered-prediction hash.

The first packaging CI run exposed a registry dependency mismatch: workspace
path builds saw the new evaluator export schema, while the trainer archive
resolved published evaluator 0.1.0. Archive verification also exposed the shared
camera-label type migration's old data-crate minimum. Data/evaluation/training/
reporting now prepare version 0.1.1 with matching dependency minimums; root and
standalone capture lockfiles are updated. All eight workspace archives pass
`cargo publish --workspace --locked --dry-run`. This is release preparation,
not a new crates.io publication or a change to the measured model.

## Motivation

Pilot 13's 384-update geometry phase improved correspondence on synthetic and
reused real-image development cohorts, but did not retain camera recovery in all
TUM sequences. Pilot 14 found deficient spatial structure, rather than a simple
feature-amplitude error. Head Stability 15 trained numerically stable RGB and
camera heads, but the camera head overfit 32 rooms. None establishes SotA.

The next controlled question is whether sustained actual-view supervision over
the full cached training cohort improves the shared spatial representation
without sacrificing completion or camera recovery. Geometry is an explicitly
renderer-supervised auxiliary; RGB remains the sole model input. The initialization
and fixed teacher are MIT V-JEPA 2.1; no noncommercial weights or teachers enter
training.

## Fixed continuation proposal

Both arms start from Pilot 13 geometry checkpoint
`0639afa2e19d0e7ac5c9c1cc8caadc75d0dcf10f105ee2870607856b7a1ed5a7`.
The control continues the latent, reconstruction-improvement, synthetic-warp
and encoder-preservation objectives; the candidate also continues actual-view
geometry NLL at weight 0.1 and temperature 0.07. All other settings match.

- Cached dataset `e54ba670…f66f05b`: 8,192 training rooms, three 256×256 views.
  Historical capture uses Zeroverse 0.25.0 / dataset adapter 0.8.0; the live demo's
  newer renderer does not change this immutable dataset's provenance.
- Seed 853, batch 16, **4,096 updates per arm**, final endpoint only, no best-validation
  checkpoint selection. This is 65,536 target exposures per arm; actual unique
  room/view coverage must come from the native training report.
- Decoder width 384, six layers and heads, 90% random target masking, two references.
- Learning rate 2e-5, encoder ratio 0.02, 128-step warmup then cosine decay;
  full encoder adaptation from step zero, fixed Pilot 11 anchor weight 4.
- First 64 validation rooms for completion; first 32 for correspondence and pose.
  All are reused synthetic **development** rooms, not an independent test set.
- CPU geometry cache covers all 8,192 training and 128 validation rooms, no test
  rooms. The native audit verifies every directed pair and reports empty labels.
- Matched updates are not matched compute: the candidate adds decoder branches.

After explicit budget renewal, a bounded warm preflight must forecast both
complete arms, CPU/GPU scoring and a reserved evaluation margin. Do not silently
shorten the fixed horizon, change weights or select solver seeds to fit the budget.
If the forecast fails, register a different bounded study before training.

## Registered synthetic retention checks

Before candidate outputs exist, register all of these conjunctive checks:

1. Complete fixed horizons, identical sample order/masks/initialization,
   151 encoder gradient tensors at every update, unchanged teacher and anchor,
   finite losses/gradients and active geometry supervision.
2. Candidate hidden-token MSE at most 1% above the better of parent and control;
   actual references must beat the matched monocular branch.
3. Candidate view AEPE at least 5% lower than both parent and control, with
   nondecreasing PCK at 8 input pixels against both.
4. Centered completion MSE at most 1% above control. Centered spatial correlation
   and adjacent-difference correlation each at most 0.005 below control. Increasing
   feature variance alone does not pass a detail check.
5. Mean synthetic pose AUC@10 across **all eight** solver seeds must not decrease
   against parent or control. Report the full range, failed fits and signed
   translation error. Never choose the best solver seed.

Any failure retains the published checkpoint. These gates are screening criteria,
not statistical proof or a SotA claim. Paired room-level uncertainty accompanies
results. Held-out real-data protocols and matched public baselines remain necessary
after a synthetic pass; existing TUM/HPatches/ETH3D cohorts are development data.

## Synthetic pose protocol

`view_geometry_export` additionally writes `pose-predictions.json`: every dense
RGB query, hard-match mutuality and the fixed local centroid for pair, same-image
and encoder controls. The CPU `gekko-eval synthetic-pose` command accepts pinned
predictions and loads camera labels afterward. **Visibility labels never filter
solver matches.** View 0 predicts view 1 in each of the first 32 validation rooms.

Use solver seeds 871–878, normalized eight-point essential RANSAC, 2,048 maximum
and 64 minimum trials, confidence 0.999, at least 12 inliers. Normalize each image
with its known focal length; the 1.5-input-pixel Sampson threshold is divided by
the mean focal length of the pair. Exclude baselines below 0.01 m from translation
and joint pose only; failed fits otherwise contribute 180 degrees. Mean over
rooms within each seed, then over seeds. This measures a calibrated geometric
probe, not learned camera-head or predicted-intrinsic accuracy.

Before using the diagnostic, validate Bevy-to-computer-vision camera conventions,
signed translation, rectangular grids, failed-fit denominators, input hashes and
complete method/room populations. Replay the parent's legacy correspondence
output to ensure the extra export does not change predictions.

The remaining old allowance can also export the original matched Pilot 13 control
(`ac60100b422e80b8a2084b1823fbbc754f9f0d1ce0212f975989b06cbf4aa914`)
under the identical protocol. This retrospective synthetic comparison diagnoses
the existing weights; it is not a new training arm or a post-hoc replacement of
Pilot 13's registered decision. Both runs retain all eight solver seeds and all
three descriptor controls.

## Heads, reporting and promotion

Head capacity or regularization changes must be a separate controlled study.
First expand room coverage substantially beyond 32 rooms and reserve disjoint
camera-head validation. Keep RGB PSNR in dB, rotation/translation in degrees,
focal relative error as a percentage, and latent metrics explicitly in feature
units. A high PSNR does not establish sharp spatial completion.

Only a qualified single selected experiment may replace the project-page/PDF
and live demo weights. Keep failed gates and historical cohort reuse visible.
The current published Head Stability 15 page and demo remain the validated artifact
while this study is pending.
