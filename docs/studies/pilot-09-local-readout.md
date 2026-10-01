# Pilot 09: fixed local correspondence readout

Registered before GPU preflight or new training/export on 2026-09-30. The user
requested continued experimentation and clearer metrics. This study consumes
only Pilot 08's **2,882.584171 seconds of unspent GPU-command allowance**. Its
separate ledger records the immutable parent ledger hash and spent time; this
does not grant a new allowance. Qualification, training, capture and all GPU
evaluation share that ceiling. CPU builds, scoring and publication are outside
it. Preserve partial or negative results; no automatic budget extension.

## Fixed hypothesis and training

Patch-center quantization may limit strict correspondence accuracy. Test a fixed
local probability centroid on the final checkpoint, against the same checkpoint's
hard patch-center readout. The reciprocal conditional score matrix and hard
argmax are unchanged: temperature 0.07, a clipped 3 by 3 neighborhood centered on
the hard match, weighted mean of integer patch-center coordinates. Distant score
modes cannot move the centroid. Use identical refinement for pair-conditioned,
trained same-image-conditioned and centered block-6 encoder descriptors. No camera,
depth, homography, target coordinate or ground-truth visibility enters inference.
This is an independently implemented readout, not a claim to reproduce a published
model's complete matching system.

Before evaluation, continue from the audited Pilot 08 final model for a fixed
**2,800-update** phase over all 8,192 training rooms, 256-pixel images, batch 16,
two references, 90% target masking and seed 751. Both optimizers reset. The trunk
learning rate is 5e-5 with a 100-step warmup and 2,800-step cosine horizon. The
already qualified tail-encoder stage starts immediately: last two blocks and
hierarchical output norms at encoder LR 1e-6. Earlier blocks and the teacher stay
frozen. The latent, RI and known-transform objectives, descriptor radius and
augmentation distribution remain fixed from Pilot 08. No noncommercial weights
are used. This is a continuation, not from-scratch model qualification.

The training TOML is `configs/experiments/pilot09-local-readout.toml`. Internal
wall ceiling is 2,000 seconds; command watchdog is 2,080 seconds. A separate
64-update preflight must show finite losses/gradients, unchanged teacher and first
encoder block, updated final encoder block and prediction head, and plausible
throughput. Never use its weights in the main run. If the fixed schedule does not
finish, retain the endpoint as incomplete rather than claiming convergence.

## Selection, controls and acceptance

Select the fixed final step/wall endpoint and record its hash before new held-out
capture or external matching. The report presents this checkpoint only. Do not
rank or select checkpoints using benchmark results. The hard/local comparison is
a within-checkpoint operator control, not a comparison of model versions.

Run all 580 HPatches pairs (295 viewpoint pairs primary) and all 3,365 ETH3D pairs.
Both remain **development** benchmarks after repeated prior use. Retain equal
sequence / equal scene-and-interval macro weighting and 10,000-resample paired
cluster intervals. The fine readout's precision gate requires a positive lower
95% bound for its paired PCK3 gain over the hard pair readout on both primary
protocols, with no mean AEPE regression. A separate fusion gate requires positive
lower AEPE-gain bounds and nonnegative mean PCK3 gain against both same-operator
controls. Report failures without tuning temperature, window, model or thresholds.
Never infer SOTA or training-seed uncertainty from these gates.

Numeric oracles check quarter-patch transforms and exact preservation of integer
flow. The first eight ETH3D pairs must retain exact hard indices/mutual flags
against the broad exporter. Known-transform scoring uses seed 1000751 on 16
validation rooms, both directions; this measures local geometry only.

After checkpoint selection generate 128 fresh test rooms, four views each, using
published bevy_zeroverse 0.25.0 / adapter 0.8.0. Registered capture seed 2610051000,
256 by 256, baseline 0.5 and density 0.35. Validate disjoint seeds through all
training ancestry. Assess all 512 targets with two references and fixed random
masking. Export every target so native Rust can recheck raw feature metrics and
choose deterministic evenly spaced annotated samples. If time is exhausted,
report missing coverage explicitly; never silently score a partial benchmark.

## Human-readable metrics and report

Latent completion uses feature signal-to-error ratio in dB, computed from actual
teacher squared amplitude and prediction error on hidden tokens. This is **not
RGB PSNR**: no trained RGB decoder or bounded RGB reconstruction exists here.
Retain latent MSE and cosine, show reduction versus references-disabled as a
percentage, and spatial variation retained as a percentage of teacher variation.
For real correspondence show mean pixel error and the percentage within three
pixels, with each benchmark's native/model-input coordinate scale stated.
Camera, depth and RGB heads stay explicitly unevaluated.

Use a single pinned publication TOML to generate an annotated local HTML page and
PDF in Rust. Include metric definitions, formulas, directions and practical limits;
fresh-room and real-view visuals; known-warp diagnostics; complete training/data
coverage; and paired uncertainty. Record power, command time, warm throughput,
VRAM/RSS, optimizer-stage evidence and shared desktop activity. Device utilization
is not occupancy; board energy includes other GPU users.

No commit, push, registry release, deployment or repository-visibility change is
part of this experiment. Independent real-viewpoint/pose evaluation and broader
competitors remain outstanding SOTA qualification work.

## Preflight timing amendment, before main launch

The 64-update CUDA preflight measured a 0.632266-second median (25.31 targets/s).
The initial 2,800-update projection was 2,008.86 seconds after the fixed 5% update
margin and 150-second overhead, just over the 2,000-second internal ceiling.
Reduce the fixed endpoint and cosine horizon to **2,700 updates**, validation every
675 and checkpoint every 1,350, before main training or benchmark export. The
same projection is now **1,942.47 seconds**. All other settings, readout gates and
budget remain fixed. Qualification receipts retain this amendment; no benchmark
result informed it.

## Additional descriptive analysis

While the fixed training phase was running, add a post-hoc completion breakdown
by geometric majority-visibility label. This uses the already planned exported
arrays and CPU scoring only, and changes no primary endpoint or acceptance gate.
Report token-weighted MSE for known hidden patches labeled visible/not visible
in the supplied references, excluding unknown labels. A majority label does not
classify every pixel in its patch. This can locate error concentrations; it does
not establish a causal explanation for smoothing.

## Completed result

Completed 2026-09-30. **Both local-precision gates and all same-operator fusion
transfer gates pass.** Retain the final checkpoint with the fixed local readout
for independent qualification. This is a development result, not a SOTA ranking
or evidence of sharp RGB reconstruction. All comparisons use one checkpoint.

- [Annotated 23-page PDF](../../.data/publications/pilot09-local-readout/paper.pdf)
- [Interactive project page](../../.data/publications/pilot09-local-readout/index.html)
- [Pinned publication TOML](../../configs/publish/pilot09-local-readout.toml)
- Receipts, predictions and native scores: `.data/pilot-09/`.

Selected model SHA-256:
`a02277687d0f10cebd7bfe9e487facc6c24d461ea67a377e29b8f04b495d7185`.
Selection preceded capture and external exports; no readout parameter was tuned
after their results. The precision gains below isolate the inference operator
within this checkpoint, not the training phase's effect relative to another run.

### Real-view matching

All 580 HPatches pairs are scored, with 295 viewpoint pairs / 59 sequences primary.
ETH3D includes all 3,365 pairs, ten scenes and seven intervals. Both remain
development datasets. HPatches uses a 240 by 240 scoring frame; ETH3D uses original
image pixels. Percentages at a fixed pixel threshold are not comparable across them.

| Same-checkpoint readout | HPatches error | HPatches within 3 px | ETH3D error | ETH3D within 3 px |
| --- | ---: | ---: | ---: | ---: |
| Encoder, hard | 25.4494 px | 11.7357% | 36.7222 px | 2.2876% |
| Same-image, hard | 21.2050 px | 12.8115% | 34.4871 px | 2.3073% |
| Pair-conditioned, hard | 20.0946 px | 13.2617% | 33.1232 px | 2.3289% |
| Encoder, local | 24.9503 px | 16.0552% | 34.9851 px | 3.5597% |
| Same-image, local | 20.6082 px | 18.6871% | 32.3730 px | 3.9865% |
| Pair-conditioned, local | **19.4655 px** | **19.6862%** | **30.8876 px** | **4.1922%** |

Local pair versus hard pair precision gains are **6.4245 percentage points
[5.6252, 7.3090]** on HPatches and **1.8634 points [1.5600, 2.2427]** on ETH3D.
Mean-error reductions are **0.6291 px [0.5781, 0.6830]** and
**2.2356 px [1.9779, 2.4864]**. Both registered precision gates pass.

Against equally refined encoder controls, error reductions are **5.4847 px
[4.6185, 6.3456]** and **4.0974 px [3.0177, 5.1254]**; precision gains are
**3.6310 points [3.1522, 4.0995]** and **0.6325 points [0.4673, 0.8031]**.
Against refined same-image controls, error reductions are **1.1426 px
[0.8872, 1.3885]** and **1.4854 px [1.0686, 1.9962]**; precision gains are
**0.9991 points [0.8439, 1.1618]** and **0.2057 points [0.1374, 0.2797]**.

These are paired 95% sequence/scene intervals from 10,000 resamples, not training-seed
uncertainty. Every refined comparison supports both error and precision gains.
The hard ETH3D precision comparison against its encoder still crosses zero.
All ten contrasts and explicit gate definitions remain in the native score files.

### Fresh-room completion and remaining limits

The new cache is
`.data/datasets/03cb8d426bf372dbe11d5b8db7a7462cac5ebfcdee07b98a5ccd14592b5157fa`.
Published Zeroverse 0.25.0 / adapter 0.8.0 generated 128 test rooms with four views
each, plus two required unused train/validation entries. All **512 test targets**
are assessed/exported, with seed disjointness checked through training ancestry.

Feature signal/error is **7.2977 dB**, using actual masked teacher power and
prediction error; it is not RGB PSNR. Hidden MSE is **0.187727 [0.184670, 0.190869]**,
versus **0.199698** without references: **5.9948% lower**. The paired room error
reduction is **0.011972 [0.010471, 0.013442]** over 128 rooms. Cosine is **0.900705**.
Only **42.2729%** of teacher spatial variation is retained.

The descriptive visibility breakdown gives MSE **0.184259** over **92,053**
majority-visible hidden patches and **0.200063** over **25,659** majority-not-visible
patches. These are pooled patch means, unlike the primary equal-view average.
The gap does not explain away visible-patch errors or representation smoothing.
RI AUROC is **0.707326**, AP **0.893792**, with 92,053 positives / 117,712 known
patches. RI uses the separate full-target branch and is not calibrated probability.

Hidden-target RGB perturbation has exactly zero effect. Reference permutation
changes features by **0.004225 maximum / 0.0001800 RMS**, failing the strict 1e-5
criterion; the monocular branch remains unchanged. RGB, camera and depth heads
remain untrained. The held-out result conditions on one fixed random mask.

Known-transform pair error falls from **9.3481 to 7.4508 input pixels** with the
local readout; accuracy within eight pixels rises from **56.7708% to 73.1088%**.
Refined same-image/encoder errors are 7.8292/12.6583 px. Pair NLL is 2.34030.
This 16-room diagnostic does not test 3D parallax or independent viewpoint transfer.

### Training and efficiency

All **2,700 updates**, **43,200 target exposures**, **8,192 rooms** and **24,576
distinct room/view combinations** are covered. Stage 1 logs 28 encoder gradient
tensors throughout. Teacher/first-block QKV probes stay unchanged; last-block QKV
changes by 0.00017155 and the prediction head by 0.00384236. Validation MSE changes
from **0.180050 to 0.178162** (1.05% lower), retaining the early regression.
This is a completed schedule, not evidence of asymptotic convergence.

Training takes **1,904.60 seconds / 31.74 minutes**, including 95.21 seconds of
preparation. Warm median/p95 updates are **0.6387/0.6757 seconds**, about **25.05
targets/second**. Peak process VRAM/RSS are **27,380/20,315 MiB**. Observed board
energy is **207.59 Wh**, **17.30 J/target**, with **99.94%** telemetry coverage;
mean board power is **392.60 W**, median device activity **92%**. Shared desktop
work is included; these are not occupancy or process-attributed energy measures.
The desktop shell reports mean activity 22.51% over 220 samples; remote-desktop
counters are unavailable in all 220. These counter windows cannot be added.

The complete study uses **2,345.76 seconds / 39.10 GPU-command minutes**. Combined
with the preserved Pilot 08 ledger, usage is **111.05 of 120 minutes**, leaving
**8.95 minutes unused**. No additional benchmark-guided training or budget extension
ran. CPU scoring/reporting is outside this command ceiling.

CPU tests cover fractional coordinate oracles, hard-readout preservation, feature
SNR scale/mask semantics, paired gates, report tamper rejection and both optimizer
resume contracts. CUDA preflight and strict Clippy pass. ETH3D preserves exact
indices/mutual flags on its eight-pair native audit and completes in 164.75 seconds.
Rust verifies every exported target array before building the report. Mobile and
desktop galleries, local links, image hashes and the 23-page PDF pass validation.

### Next qualification

Keep the fixed local readout and equally refined controls. Register an independent
real-viewpoint/pose cohort before selecting architecture changes from it. Camera
heads or calibrated pose probes need explicit supervision/input contracts and
angular, focal and coverage metrics. Public-baseline protocol parity and training
replication remain required for SOTA. Completion variance and reference-order
sensitivity are separate open issues. Throughput tuning needs controlled workloads
with both timing and energy measurements.

The page and paper remain local. No commit, push, registry release or deployment
was performed in this study.
