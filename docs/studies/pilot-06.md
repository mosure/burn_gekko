# Pilot 06: regenerated rooms and extended reconstruction training

**Outcome: blur remains unresolved.** The selected checkpoint improves validation
RGB error and uses the references, but held-out furniture, ceiling lights and
fine texture remain blurred or missing. It is not an accepted reconstruction
model. No noncommercial pretrained weights or teachers enter its ancestry.

Read the [annotated PDF](../../.data/pilot-06/burn_gekko_pilot_06_report.pdf),
[machine-readable report](../../.data/pilot-06/quality-summary.json),
[selection record](../../.data/pilot-06/selection.json), and
[complete command budget](../../.data/pilot-06/budget.json). The PDF includes raw
completions, target-selected detail crops, reference controls, geometry-only
diagnostics, confidence intervals, unsuccessful arms and efficiency measurements.

## Data and provenance

The published pins are `bevy_zeroverse=0.25.0` and
`bevy_zeroverse_burn=0.8.0`, verified against crates.io on September 28 and
rechecked September 29, 2026. The isolated capture tool has an exact lockfile.
Registry publication timestamps, checksums and source/binary receipts are in
`.data/pilot-06/`; earlier caches and experiments are preserved.

The regenerated dataset contains **8,192 training, 128 validation and 128 reserved
test rooms**, each with three simultaneous 256×256 views. Compressed shards occupy
24.08 GB. The training RGB host cache occupies 18 GiB; trainable encoder features
are recomputed, never cached across updates. RGB, depth, world position and camera
metadata remain under the immutable dataset directory:

`.data/datasets/e54ba670d6087c445d1d04c7c3ba1742d0e610d08533690921ac655f1f66f05b`

There are ten layouts, six palettes and three lighting modes; all 180 combinations
occur in training. New seeds are disjoint from previous captures and the separate
32/16/16-room fitting dataset. Main-data seed ranges are 2609300000–2609308191
(train), 2609308192–2609308319 (validation), and 2609308320–2609308447 (test).
Density is 0.35, with portable procedural assets and no humans.

The camera baseline control **0.25 is dimensionless**, not 0.25 metres. This
published policy gives camera-zero reference bounds 0.635–2.7625 m and minimum
pair separation 0.155 m. Actual validation all-pair median separation is 1.336 m,
p95 2.612 m and maximum 3.925 m. The two nonzero cameras can exceed the
camera-zero bound. Proposal overlap is distinct from measured pixel visibility.

Ground-truth reprojection on validation target view zero gives covered-pixel MSE
0.000887 versus 0.011542 at unchanged reference coordinates, with 84.5% average
coverage. This confirms capture coherence on that support. It uses privileged
geometry and is neither a model output nor a matched model-performance bound.

## Architecture and experiment lineage

The shared MIT V-JEPA 2.1 image encoder has 12 blocks, width 768 and 16px patches.
A six-block, width-384, six-head decoder uses 2D rotary positions and joint
attention over two references. Only visible target patches enter the completion
encoder. Fusion, reconstruction and RI heads start randomly at the root of the
lineage. The encoder progressively unfreezes and is fully trainable in later legs.

The direct RGB head removes the earlier forced patch-variance/statistics product.
Both reconstruction branches receive 10×RGB MSE + RGB L1 + aligned derivative
L1 at offsets 1, 2, 4 and 8, restricted to hidden endpoints. Derivative coefficient
is four; there is no unaligned gradient-energy reward. RI error targets are
detached. Dense hidden target features belong only to the separate RI operation.

The optional appearance head predicts reference flow and source-mixture weights.
An image-pyramid photometric auxiliary warps pooled reference images at scales
2/4/8/16/32; pooling an already warped image had a demonstrably flat coarse
matching gradient in a controlled translation test. Native quarter-resolution
edge-aware second derivatives constrain flow curvature without geometry labels.
The selected coefficient is 0.0025 inside the auxiliary weight five. These are
adapted objectives and architectural extensions, not exact paper reproduction.

| Leg | Updates | Validation MSE | Edge cosine | Edge energy |
| --- | ---: | ---: | ---: | ---: |
| Direct main baseline | 12,001 | 0.002435 | 0.404 | 0.180 |
| Pooled-warp diagnostic | 1,000 | 0.002454 | 0.397 | 0.163 |
| Image-pyramid candidate | 3,000 | 0.002373 | 0.409 | 0.180 |
| Curvature 0 control | 400 | 0.002287 | 0.419 | 0.181 |
| Curvature 0.01 diagnostic | 400 | 0.002292 | 0.420 | 0.174 |
| Curvature 0.0025 candidate | 400 | 0.002288 | 0.420 | 0.178 |
| Extended training | 11,336 | 0.002120 | 0.438 | 0.222 |
| Exact continuation, selected | 493 additional | 0.002112 | 0.441 | 0.217 |

Every main-data validation row covers all 128 rooms and three target views.
The curvature comparison holds parent, initialization, batch inputs, masks and
schedule equal; initial outputs are bitwise identical. The weaker penalty meets
pre-recorded continuation tolerances but not the final quality gates. Other
architecture/duration comparisons are adaptive recipe comparisons, not isolated
causal ablations. Rejected diagnostic weights do not enter the selected lineage.

A separate 16-room capacity curriculum reached training MSE 0.000404, aligned
cosine 0.759, energy 0.551 and patch-interior energy 0.510 over 48 targets. This
passes the fitting gate but does not establish generalization. Its 96,000 target
exposures are included in the selected ancestry.

The final model's audit of the first 128 main training rooms (all three views)
has MSE 0.002524 and edge energy 0.202. Blur is therefore also present on training
inputs, not confined to unseen rooms. This prefix is not a random estimate of
the full training set and does not identify a specific capacity or optimizer cause.

The selected ancestry contains **37,230 updates and 531,680 target-view exposures**
over 8,208 distinct training rooms. Main-data ancestry contributes 435,680
exposures, or 17.73 equivalent passes over its 24,576 room/view pairs. Recursive
hash and sampler auditing counts exact-resume prefixes once and finds no
nontraining optimizer inputs. Own-weight transfers reset both optimizers and
record parent hashes; exact continuation restores both optimizer states and the
original cosine horizon.

Selected checkpoint: `.data/runs/pilot-06-refine-continue/final`.
Model SHA-256: `a3b8564e5b34aa14a6b3c593e67a75fa91ed94dc175a9db31ba1df81951c804a`.
Selection was recorded before any reserved-test inference. Neither candidate met
the detail thresholds; the selected continuation trades 1.88% lower validation
interior energy for 0.40% lower MSE and higher edge alignment.

## Reserved-test findings

All 128 rooms and 384 target views are exported. Intervals bootstrap rooms, not
individual views, using 2,000 replicates. Metrics use raw hidden RGB; display
panels retain observed input pixels and clip to [0,1] without enhancement.
Evaluation uses one fixed 75% hidden-patch mask, so mask-seed robustness is unknown.

| Metric | Estimate | 95% room-bootstrap interval |
| --- | ---: | ---: |
| Hidden RGB MSE | 0.002428 | 0.002194–0.002686 |
| Mean per-target PSNR | 27.06 dB | 26.58–27.52 |
| Aligned edge cosine | 0.440 | 0.425–0.456 |
| Gradient energy ratio | 0.222 | 0.207–0.237 |
| Patch-interior energy | 0.216 | 0.202–0.231 |
| Monocular hidden MSE | 0.002718 | 0.002464–0.003004 |
| Monocular minus cross-view MSE | 0.000290 | 0.000240–0.000344 |
| Unrelated minus related MSE | 0.000661 | 0.000577–0.000748 |

Cross-view MSE is **10.68% lower** than the paired monocular branch. Unrelated
reference MSE is 0.003089. Hidden-target replacement changes completion exactly
zero. Independent-versus-batched hidden RGB RMS is 0.0000381. Reference reordering
has RMS 0.0000178 and maximum 0.000407; that maximum **fails the retained 1e-5
check**. The mathematical set symmetry does not imply bitwise native equality.

Co-visibility pooled AUROC is **0.641**, AP **0.888**, with constant-score AP
baseline **0.822**. RI scores represent relative reconstruction utility, not
calibrated visibility probabilities. Geometry is read only for evaluation.

The appearance mixture assigns 85.9% mean weight to generated RGB. Hidden-visible
flow endpoint error is 23.96 px versus 37.02 px for unchanged coordinates; only
7.97% are within 3 px. About 80.85% of hidden pixels have a valid reference
correspondence, and 76.26% have one inside the head's displacement range. The
range limit alone therefore does not explain the remaining blur.

Aligned edge cosine and reference benefit pass their thresholds, but energy,
patch-interior detail, strict reference permutation and manual visual review
fail. In the first reserved room, chair/sofa silhouettes and thin structure remain
blurred despite clear references. The worst-error sample misses ceiling lights
and furniture and distorts window boundaries. These results do not support a
claim that simply scaling the dataset or training duration resolves the issue.

## Efficiency and reproducibility

The single GPU is an RTX PRO 6000 Blackwell Workstation Edition with 97,887 MiB
reported memory. The host has 94.10 GiB RAM; the closeout inventory records driver
610.43.02, Rust 1.98.0 and Burn 0.21.0. Training uses CUDA F32/Fusion.

The shared ceiling is 43,200 seconds of cumulative capture, training and GPU
**command wall time**, including startup, checkpoint writes, evaluation and failed
attempts. Overlapping commands are counted separately. CPU engineering and
reporting are excluded. The final total is **11.747 hours (42,289.2 seconds)**, leaving 15.18 minutes unused under the ceiling. No further training or dataset generation remains active.

Batch 16 was retained after an uncontended RI screen measured 19.47 targets/s
versus batch 32 at 20.22, with approximately twice the memory at batch 32. The
appearance model uses about 35.55 GiB sampled process VRAM. Its extended process
slowed from approximately 0.90 to 2.24 s/update while device utilization fell
from 85% to 32%; clocks stayed near 2.8 GHz and temperature decreased. Exact
restart restored a 1.064 s/update median over 443 post-warmup updates, **2.11×
faster** than the preceding last 100 updates. This is a measured workaround;
the process-lifetime software cause remains unprofiled. Preparation validates
all dataset shards and takes approximately 130 seconds for large-data training.

Version 9 passed **68 workspace tests**, strict CUDA Clippy and native builds.
The final audit confirms all 53 Rust source/dependency files still match that
archive. Five Python report/runner tests, Python syntax checks, Cargo formatting
and whitespace checks pass. Numerical quality failures above remain failures.
The report audits input splits, checkpoint ancestry, model hashes, component
mixture identities, exported target completeness and cumulative timing.

```sh
# CPU-only report regeneration after the completed study; no training is launched.
OPENBLAS_NUM_THREADS=1 .data/analysis-venv/bin/python \
  tools/legacy/long_study_report.py --config configs/archive/pilot-06/pilot06-report.toml
```

Configs and study plans are TOML. Source/binary versions v1–v9, dataset manifests,
raw exports, optimizer checkpoints and telemetry remain under `.data/`. Use the
archived binary matching a checkpoint's identity. The original working notes are
retained as `.data/pilot-06/pilot-06-working-notes.md`.

## Remaining work

Profile the long-process slowdown before scheduling another long run. Isolate
RGB-only correspondence learning on the small-room protocol, comparing the
current direct flow head with explicit coarse-to-fine matching. Require aligned
detail and reference dependence together; sharpening is not evidence of correct
completion. Separate frozen/adapted encoder and entirely random encoder controls.

The reserved test has now been consumed. Any recipe selected after these results
needs a fresh held-out test. Multiple training seeds, wider baselines and real
images remain necessary for paper-level claims. The next experiment is not
launched as part of this bounded study.
