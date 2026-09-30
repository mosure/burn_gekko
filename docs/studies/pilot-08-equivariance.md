# Pilot 08: larger-data known-transform training

Registered 2026-09-30, before the new training or benchmark exports. The user
requested continued experimentation. This study uses a **separate 7,200-second
GPU-command ceiling**, including qualification, capture, training and GPU
evaluation. Pilot 07's ledger remains unchanged. CPU builds, scoring and reporting
are outside that ceiling. Runtime evidence lives under `.data/pilot-08/`.

## Question and fixed intervention

Does the known-transform matching objective retain its local gains after 6,000
updates over all 8,192 training rooms, while improving real-viewpoint matching
and maintaining sparse cross-view latent completion? This is one training phase,
not a hyperparameter sweep. It starts from the audited native-spatial-refinement
parent, resets both optimizers and the unfreeze gate, and creates the bounded
descriptor correction at zero. No noncommercial weights or teachers are used.

The [training TOML](../../configs/experiments/pilot08-equivariance.toml) fixes
seed 739, batch 16, 256-pixel images, two references, 90% random target masking,
6,000-step cosine horizon, 100 warmup steps, learning rate 1e-4, and teacher-fixed
latent prediction plus RI. Both pair-conditioned and same-image-conditioned
descriptors receive the same bidirectional known-transform objective, with weight
0.1 and temperature 0.07. The descriptor residual radius remains 0.25. Camera
labels, renderer geometry and augmentation transforms never enter inference.

Training begins with the encoder frozen. The last two blocks may unfreeze at
encoder LR 1e-6 only after at least 2,000 updates, a validation reduction of 0.2%
from stage entry, and the existing no-large-regression gate. If the gate never
opens, report that result rather than forcing it during the run. Earlier encoder
blocks and the teacher remain frozen. The internal wall ceiling is 5,160 seconds;
the command watchdog is 5,400 seconds. A wall stop before 6,000 updates is an
incomplete schedule and is not described as converged.

Implementation clarification: the imported encoder's existing
`with_last_blocks_require_grad` policy also enables its hierarchical output
normalizations. Stage 1 therefore adapts the last two transformer blocks **and
output norms**, including the block-6 output norm. Early transformer blocks and
patch embedding stay frozen. The encoder matching control is always measured
from the same final checkpoint, so it includes that adaptation. This policy was
already present in the sealed training binary; no mid-run change was made.

Before launch, a separate 64-update CUDA diagnostic explicitly starts at stage 1
from the audited parent. It must show finite losses and gradients, nonzero last
encoder updates, zero first-encoder and teacher updates, and enough measured
throughput/memory headroom for the main phase. This explicit initial stage is an
audited weights-only option; exact resume always restores the checkpoint gate.
The diagnostic is never used as the main phase's weight source. A CPU test checks
both optimizers and deterministic transforms across a real resume boundary.

## Selection and acceptance

Select the final checkpoint at the fixed step/wall stop, before new external
exports. Do not select on HPatches, ETH3D, reference count, or individual examples.
Record its SHA-256 and the readout selection before evaluation. All comparisons
in the publication use this same checkpoint. Controls are centered block-6 encoder
descriptors and trained same-image-conditioned descriptors; the primary readout
is the pair-conditioned spatial descriptor. Temperature is fixed at 0.07.

Run all 580 HPatches pairs (295 viewpoint pairs primary) and all 3,365 ETH3D pairs.
Both remain **development** datasets because earlier experiments already used
them. Use the native Rust scorer, registered macro weighting, PCK3/AEPE and
10,000-resample paired sequence/scene intervals. The combined fusion gate requires
positive AEPE gains with lower paired interval bounds above zero and nonnegative
mean PCK3 gains against both controls on both primary protocols. PCK3 uncertainty
is reported separately; this is not an equivalence test. Do not promote a local-warp gain as
evidence of 3D transfer. Focused ETH3D export must pass its eight-pair exact CUDA
parity check against the broad exporter; partial exports are not scored.

After checkpoint selection, generate a new, disjoint-seed cohort of 128 procedural
rooms with four cameras each using the published capture wrapper (capture seed
2610050000, baseline radius 0.5, density 0.35; registered in
`configs/data/capture-pilot08-heldout.toml`). Evaluate every
test room with two references and 90% random masking. Register the capture seed
before capture. Report cross-view/monocular MSE and cosine, spatial variance,
paired room intervals, co-visibility ranking/calibration, hidden-RGB isolation,
reference permutation error and deterministic annotated examples. If budget
permits, one-/three-reference assessments use the same first 32 test rooms and
are explicitly secondary. Known-transform evaluation uses a new fixed augmentation
seed on 16 validation rooms; these room probes are development evidence.

Camera, depth and RGB heads remain untrained. This study does not establish
state of the art or resolution of RGB blur. Independent benchmarks, broader
competitors and training-seed replication remain necessary even if the internal
fusion gate passes.

## Efficiency and closeout

Record device-wide utilization, power and energy with shared desktop activity
allowed, plus process VRAM/RSS, warm update p50/p95, stage counts, augmentation
time, target throughput, room/view coverage and first/last loss windows. Do not
interpret utilization as occupancy or device energy as process-only energy.

Generate one pinned experiment TOML, an annotated local project page and PDF from
native Rust evaluation. Check assets, hashes, mobile/desktop rendering and PDF
layout. Preserve sealed source/binaries, complete command ledgers and any failed
diagnostics. No staging, commit, push, publishing or deployment is authorized.

## Completed result

Completed 2026-09-30. **All four registered spatial-readout transfer gates pass.**
The pair-conditioned descriptor improves over both the encoder and trained
same-image controls on complete HPatches viewpoint and ETH3D protocols. This is
an accepted result for the bounded development hypothesis, not a SOTA claim or
resolution of RGB blur. The success is specific to the trained spatial readout;
raw decoder features remain weaker than the encoder on HPatches.

- [Local project page](../../.data/publications/pilot08-equivariance/index.html)
- [Annotated 20-page PDF](../../.data/publications/pilot08-equivariance/paper.pdf)
- [Single-run publication TOML](../../configs/publish/pilot08-equivariance.toml)
- Runtime receipts: `.data/pilot-08/`; training: `.data/runs/pilot-08-equivariance/`.

The fixed phase completes all **6,000 updates**, with **96,000 target exposures**
covering **8,192 rooms and all 24,576 distinct room/view pairs**. Encoder stage 0
lasts 2,500 updates; stage 1 lasts 3,500. The gate opens at validation MSE 0.18555.
Final validation MSE is **0.180129**, versus 0.186699 at phase entry (3.52% lower).
It continues to improve late in the run; this is a completed schedule, not proof
of asymptotic convergence. The teacher and first-block QKV probes have exactly
zero change; last-block QKV changes by 0.00012435. Stage 1 logs 28 encoder gradient
tensors on every update, reflecting the two blocks and output norms.

Selected final checkpoint, before fresh capture or benchmark export:

`02b8027b41d951c60afb39fc606b210ecb44e3dec4b9d2a1d53a9ddee5a62244`

### Correspondence from that checkpoint

| Readout | Known-warp AEPE | HPatches viewpoint AEPE | HPatches PCK3 | ETH3D AEPE | ETH3D PCK3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Centered block-6 encoder | 13.9518 px | 25.4678 px | 11.7290% | 36.7393 px | 2.2877% |
| Same-image conditioned | 9.7375 px | 21.3770 px | 12.6977% | 34.8031 px | 2.2940% |
| Pair-conditioned spatial head | **9.4326 px** | **20.3217 px** | **13.1682%** | **33.5004 px** | **2.3194%** |

HPatches includes 580 pairs, with 295 viewpoint pairs / 59 sequences primary.
The paired AEPE reductions are **5.1461 px [4.1983, 6.0477]** versus the encoder
and **1.0552 px [0.8044, 1.2982]** versus same-image conditioning. PCK3 gains are
1.4392 and 0.4705 percentage points, respectively, with positive intervals.

ETH3D includes all 3,365 pairs / ten scenes / seven intervals. Paired AEPE
reductions are **3.2389 px [2.1782, 4.2689]** versus the encoder and
**1.3027 px [0.9129, 1.7980]** versus same-image conditioning. Mean PCK3 gains are
0.03166 and 0.02540 percentage points. The encoder-comparison PCK3 interval is
**[-0.00573, 0.07241] percentage points**, crossing zero. The registered gate uses
positive AEPE intervals and nonnegative mean PCK3, so it passes; strict PCK3
improvement over the encoder is not established. The same-image PCK3 interval is
[0.01289, 0.03900] percentage points.

These are paired 95% scene/sequence intervals with 10,000 resamples; they do not
measure training-seed uncertainty. Both benchmarks remain development evidence.
The pair readout reduces AEPE by 20.21% on HPatches and 8.82% on ETH3D relative to
its own encoder control. No published-model ranking is inferred from these
hard-patch protocols.

Raw HPatches decoder features still give AEPE **28.3549 px**, versus 25.4678 for
the encoder. Reciprocal attention gives 23.1044 px but PCK3 11.1421%, below the
encoder's 11.7290%. The trained head's result must not be generalized to every
decoder/attention readout.

The separate known-transform diagnostic covers 16 development rooms. Pair NLL is
2.40842, versus 2.45117 for self conditioning and 3.16622 for the encoder; pair
PCK8 is 56.58%. Its 32.39% AEPE reduction versus the encoder demonstrates local
augmentation learning, not independent 3D transfer.

### Fresh rooms and remaining representation limits

The new dataset is
`.data/datasets/7ccea4c4c9de4afeb642865043845f55c69f052e0ef846d59511b4c5fdb97ca3`.
It was generated after checkpoint selection using the registered seed and
published bevy_zeroverse 0.25.0 / bevy_zeroverse_burn 0.8.0 wrapper. Assessment
checks seed disjointness through the complete training ancestry. Every one of
the **128 test rooms / 512 target views** is assessed and exported; six evenly
spaced sample identities, including endpoints, appear in the page/PDF.

Cross-view MSE is **0.191326**, room-bootstrap interval [0.188328, 0.194395], versus
**0.201875** without references: **5.23% lower**. Paired room gain is 0.010549,
interval [0.008904, 0.012222]. Latent cosine is 0.89874. Shuffling reference spatial
tokens gives MSE 0.199167; unrelated-room references give 0.244650. These controls
support use of reference content/layout, without proving general 3D reasoning.

Co-visibility RI AP is **0.88746** and AUROC **0.70301**, over 117,699 known hidden
patches, 91,215 positive. Scores come from the separate full-target branch and
are not calibrated visibility probabilities. Spatial feature variance remains
**40.98%** of the teacher's. Hidden-RGB perturbation changes predictions by exactly
zero, but strict reference-order invariance at 1e-5 fails: maximum 0.002890,
RMS 0.0001567. Camera, depth and RGB heads remain untrained.

On the same first 32 rooms / 128 target identities, one, two and three references
give MSE **0.196985 / 0.190683 / 0.186937**. Masks and checkpoint are identical;
monocular isolation passes at 1e-7. These are within-model information controls,
with no checkpoint or example selection by quality.

### Efficiency, verification and budget

Training takes **3,779.09 command seconds / 62.98 minutes**, including 93.80 seconds
of preparation. Warm stage-0/stage-1 update medians are **0.5358 / 0.6211 seconds**,
p95 **0.5657 / 0.6515 seconds**, or approximately **29.86 / 25.76 targets/second**.
Peak process VRAM is **27,380 MiB** and RSS **20,347 MiB**. Observed board energy is
**435.65 Wh**, or 16.34 J per logged target, with 99.97% telemetry coverage.
Mean board power is 415.13 W; median device activity is 94%. These include shared
desktop load and are not occupancy or process-attributed energy measurements.

The process monitor separately records the desktop shell with mean reported
activity 17.43% across 397 observations. Remote-desktop counters are unavailable
in all 397 observations, not zero. These windows/counters cannot be added to
derive occupancy or a causal estimate of training slowdown. Native Rust parses
the log, preserving unavailable values and command names containing spaces.

First/last 32-update means: cross MSE **0.189714 to 0.180218**, pair NLL
**3.25706 to 2.47225**, self NLL **3.25709 to 2.51727**. Augmentation remains about
29 ms per update. These changing minibatches are descriptive optimization evidence.

Focused ETH3D export passes exact CUDA indices/mutual-flag parity on its first
eight pairs and finishes the full population in 157.35 seconds. The main workspace
suite passes 122 tests; the additional activity-parser test and its real-log fix
pass targeted tests. CPU/CUDA Clippy, deterministic tail-stage resume, checkpoint
and telemetry association checks, native report hash/image/link validation and
mobile/desktop browser interactions pass. The PDF is visually checked, with no
overfull boxes or unresolved references in its final build log.

The complete study consumes **4,317.42 / 7,200 GPU-command seconds** (71.96 minutes),
including preflight, capture, training and all GPU assessments. **2,882.58 seconds
remain unused.** Pilot 07's ledger is unchanged. All study GPU jobs and the
read-only process monitor have stopped; no other desktop process was modified.

### Decision and next controlled work

Retain this checkpoint as the current spatial-matching candidate. The immediate
fusion-conditioning weakness is resolved for the registered spatial readout on
these development protocols; broad foundation-model quality is still unproven.
The next study should preserve these controls while testing a finer spatial
readout and an independent real-viewpoint/pose cohort. A camera head needs its
own declared input and supervision contract, calibrated pose/intrinsics metrics,
and a held-out test. A geometry-supervised auxiliary must be identified as such;
it cannot be presented as the current self-supervised recipe. Do not spend the
unused allowance merely to select against the already observed benchmarks.

No staging, commit, push, deployment or publication was performed.
