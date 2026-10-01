# Pilot 11: controlled full-encoder adaptation

Registered 2026-09-30 before preflight or screen training. The user explicitly
authorized **up to 12 additional GPU hours**. `.data/pilot-11/budget.json` is a
new, cumulative 43,200-second ledger covering every CUDA preflight, training,
capture and inference command, including failures. CPU builds, preparation,
scoring and publication are outside it. Pilot 10 remains in the older allowance.
No automatic extension, unbounded schedule or new pretrained weights.

## Question and fixed controls

The trained spatial readout uses encoder block 6, but Pilot 09 adapts only the
last two blocks and output norms. Test whether allowing all encoder blocks and
patch embedding to adapt improves geometry without damaging latent completion.
The teacher remains fixed. This changes only the trainable encoder stage;
architecture, losses, readout, data, batch, sample order and learning rates match.
The existing known-transform loss already uses bilinear soft labels; no new
target or nearest-patch label replacement is needed.

Both arms start from audited Pilot 09 final weights
`a02277687d0f10cebd7bfe9e487facc6c24d461ea67a377e29b8f04b495d7185`.
Restart both AdamW optimizers. Use 8,192 cached Zeroverse 0.25.0 rooms, 256px,
batch 16, seed 771, two references, 90% random target masking, spatial block 6,
residual radius 0.25, latent/RI objectives and the unchanged known-transform
objective (weight 0.1, temperature 0.07, equally trained same-image control).
Trunk LR is 5e-5 and encoder LR 1e-6; 100 warmup steps and a fixed 2,048-update
cosine horizon. Validate every 512 updates and save every 1,024.

- `tail`: encoder stage 1 throughout, matching the current recipe.
- `full`: encoder stage 2 throughout, including the spatial input block.

Before these arms, a separate 64-update full-stage CUDA preflight must complete
with finite loss/gradients, unchanged teacher, updated first and last encoder
blocks, and measured memory/throughput. Its weights are never used afterward.
The workstation has 96 GiB VRAM; retain batch 16 if the preflight fits. If it
does not, register a matching batch change in both arms before screen training.
No concurrent GPU studies. Desktop load remains permitted and separately noted.

## Selection and main phase

Evaluate each fixed screen endpoint on the same 32 validation rooms and the
same 32-room known-transform diagnostic (seed 1000771). Export both fixed hard
and local readouts for the pair, same-image and encoder controls. The full arm
is selected only if all of the following hold relative to the matched tail arm:

1. Validation latent MSE is no more than 1% higher, and references reduce MSE.
2. Pair-conditioned local mean pixel error is lower and PCK8 is no lower on
   the fixed transform diagnostic.
3. Teacher probes are unchanged and the intended encoder stage is verified.

Otherwise retain the tail recipe and explicitly reject the full-adaptation
hypothesis for this screen. External datasets do not select the arm.
Screen comparisons are internal controlled-study records; the generated project
page/PDF presents only the subsequently selected training run and checkpoint.

After selection, continue from that arm's fixed endpoint in a weights-only main
phase, again resetting both optimizers. Fix 12,000 updates, batch 16, seed 773,
trunk LR 4e-5, encoder LR 8e-7, 200 warmup steps, 12,000 cosine horizon,
validation every 1,000 and checkpoints every 2,000. Preserve the selected stage.
Before launching, use measured throughput plus a 10% timing margin and preparation
reserve to ensure the complete phase and at least 1,800 evaluation seconds fit
the remaining cumulative budget. If not, register the shorter fixed horizon
before main launch; do not change it in response to validation or test results.
Nonfinite parameters or loss are a stop condition; wall-cut runs remain partial.

## Evaluation and reporting

Select the fixed final endpoint before fresh capture or external exports.
Use all HPatches/ETH3D pairs with the Pilot 09 fixed local readout and equally
refined controls. They remain development benchmarks. Pilot 10's TUM cohort has
now informed the project and any reuse is explicitly development, not another
independent claim. Retain its first frozen-model result as historical evidence.

After selecting the final checkpoint, capture 128 new four-view test rooms with
seed 2610061000, 256px, baseline 0.5, density 0.35, disjoint from every ancestor.
Score/export all 512 target views, including hidden-input isolation, reference
permutation, feature signal/error dB, reference benefit, variance and visibility
strata. Report camera-motion probes separately from untrained camera heads.
No RGB PSNR or RGB sharpness claim is inferred from latent metrics.

Keep complete training coverage, source/weight identity, optimizer resets,
throughput, VRAM/RSS, device energy and shared-load caveats. Record both successful
and negative controls. Independent training-seed replication, broader real-world
pose cohorts and protocol-matched public baselines remain SOTA gates. Generate
one native Rust project page and annotated PDF for the selected main run, with
clear metric units and task-specific limitations. No commit/push/deployment is
part of this research turn.

## Preflight timing amendment

The first full-stage preflight reached 56 updates but did not finish its final
evaluation within the 280-second command cap; the watchdog terminated it at
281.84 seconds. All elapsed time is charged to the study. Its parameters are not
used by either screen. Updates slowed from roughly 1.6 to 4 seconds while a
separate Bevy compilation was active; this is observed contention, not an
isolated causal measurement. Gradient logging showed 151 encoder tensors and
process VRAM around 44 GiB, without an allocation failure.

Repeat the same 64 updates from the original Pilot 09 parent, seed and learning
rate schedule, with a 540-second internal / 600-second command cap. Only these
wall limits and output path change. Retain both attempts, compare overlapping
losses, and record shared CPU/GPU load. Do not start screens until this completes
with the required teacher and encoder-update probes.

## Preflight qualification and screen timing

The repeat completed 64 updates. All 151 encoder gradient tensors were present;
first/last QKV and prediction-head probes changed while the teacher QKV probe
remained exactly unchanged. The native replay check matched all 56 overlapping
updates within the registered 1e-6 absolute plus 1e-5 relative tolerance. Peak
process VRAM was 44,180 MiB, warm median 1.651 s/update, p95 2.738 s/update.
These are shared-workstation measurements, not isolated kernel efficiency.

Before either screen launches, increase both internal wall limits from 4,000 to
6,000 seconds and command caps to 6,200 seconds: 2,048 times measured p95 plus
preparation/evaluation reserve. Keep exactly 2,048 updates and the fixed cosine
horizon; the larger cap is a watchdog, not an adaptive stopping criterion.
Preflight weights are discarded. Run tail then full with the sealed trainer.

## Final evaluation reservation

The registered secondary resolution diagnostic increases the post-training
reservation to 2,700 GPU-command seconds. Primary evaluation command caps total
1,680 seconds: fresh capture 300, complete latent export 300, known-transform
audit 120, HPatches 240, ETH3D 540 and calibrated TUM export 180. Unused time stays
in the cumulative allowance. The final known-transform diagnostic uses 32 rooms
and seed 1000773; it does not select a checkpoint. The original 256px camera
exporter is retained for primary evaluation. The generalized exporter must pass
the separately registered correspondence replay before any 512px inference.

For the fixed 12,000-update main phase, forecast command time as screen p95 update
time times 12,000 times 1.10, plus 600 seconds for preparation, validation and
checkpoint writes. The watchdog may allow a further 25% timing margin plus 120
seconds, constrained by the cumulative budget and evaluation reservation. These
are wall limits; the optimizer horizon and final endpoint stay fixed.

## Completed screen and main launch

Both matched screens completed all 2,048 updates, 32,768 target exposures,
8,192 rooms and 24,576 room/view combinations. Teacher probes remained exactly
unchanged. Tail adaptation logged 28 encoder gradient tensors per update and
unchanged first-block QKV; full adaptation logged 151 tensors and changed both
first and last QKV probes.

| Validation-only selection metric | Tail | Full |
| --- | ---: | ---: |
| Hidden latent MSE | 0.17704871 | 0.18453609 |
| References-disabled MSE | 0.20331964 | 0.20825757 |
| Known-transform local mean error (px) | 7.4941 | 3.4275 |
| Matches within 8px (%) | 72.78 | 96.08 |

The native selector verifies identical recipes except trainable stage, exact
training sample order, validation masks and transformed-query populations.
Full adaptation improves geometry but regresses completion by 4.23%, exceeding
the registered 1% limit. **Select tail; reject full adaptation for this recipe.**
This is evidence of a task tradeoff, not proof of its gradient-level mechanism.
The immutable selection is `.data/pilot-11/screen-selection.json`.

Tail's warm median/p95 update times were 0.6934/1.2635 seconds, versus
0.9455/1.1921 for full; shared-load changes prevent causal speed comparisons.
Peak process VRAM was 27,380/44,180 MiB. These measurements include all periods
of the shared workstation; board energy is not process-attributed.

The main run launched at 2026-09-30 16:18:58 UTC from tail endpoint
`813f84f03e617fb5e27e7841ff4e6247755b743f37fcba96bb2b358a329bab37`.
Its fixed schedule remains 12,000 updates; the 21,719-second command ceiling
includes a conservative timing margin and leaves the evaluation reservation.
The preceding 256px exporter replay reproduced all 558 rows byte-for-byte,
with zero hard-index, mutual-mask or coordinate differences. This qualifies
the generalized exporter for the later registered 512px diagnostic.

## Completed selected main phase

The selected tail continuation completed all **12,000 updates** at 2026-09-30
18:35 UTC. Its final checkpoint is
`bdd1bd204afc5584ce7277af40d8bea80f6987c6611937e79b7fb13074789637`,
sealed before capture or external inference. The phase contains 192,000 target
exposures, all 8,192 rooms and 24,576 room/view combinations. Every update has
the intended 28 encoder gradient tensors; the first encoder QKV and teacher
probes remain unchanged, while the last encoder QKV and prediction head change.
Scheduled validation MSE moves from 0.17671439 to **0.17265890**. This is a
development trajectory, not an independent training-seed comparison.

Fresh capture produced dataset
`c6a43000eb55ab590ff69c87103e4b30c78c71820e53a3f1ab12bcedd1fd0683`
after endpoint selection. All 512 test targets are exported and their metrics
recomputed from arrays by the native publisher.

| Fresh completion metric | Result |
| --- | ---: |
| Hidden latent MSE | 0.18527921 |
| References-disabled MSE | 0.19799510 |
| Error reduction from references | 6.42% |
| Feature signal/error, not RGB PSNR | 7.36 dB |
| Teacher spatial variation retained | 43.30% |
| Learned RI co-visibility AUROC | 0.7170 |

The paired reference-benefit MSE interval is **[0.011131, 0.014351]**, with
128-room bootstrap clusters and 10,000 deterministic replicates. Hidden-RGB
perturbation changes predictions by exactly zero. Strict reference-permutation
invariance fails: maximum 0.00467491, RMS 0.000176116, tolerance 1e-5. Spatial
variation remains strongly suppressed; lower feature error does not establish
sharp RGB reconstruction or correct fine structure.

All 580 HPatches and 3,365 ETH3D pairs are scored. Every registered hard/local
fusion-transfer gate and both local-precision gates pass within this checkpoint.
The local readout results are:

| Development metric | Fusion | Same-image control | Encoder control |
| --- | ---: | ---: | ---: |
| HPatches viewpoint mean error, 240px frame | 18.7348 px | 20.2454 px | 24.8774 px |
| HPatches viewpoint within 3px | 20.78% | 19.34% | 16.11% |
| ETH3D mean error, original image coordinates | 29.7577 px | 31.5032 px | 34.9316 px |
| ETH3D within 3px | 4.37% | 4.09% | 3.56% |

HPatches' primary population is its 295 viewpoint pairs; its 285 illumination
pairs remain separately reported, including local-readout regressions there.
The two benchmarks' pixel thresholds have different relative image scales.
The final 32-room known-transform diagnostic gives 7.4265px mean error and
73.64% within 8px for the local fusion descriptor.

Calibrated TUM motion remains a weakness. At 256px, fusion has 8.61-degree mean
rotation error, 54.99-degree signed translation-direction error, 14.61% pose
recall within 10 degrees and 6.74% AUC@10. Both per-sequence transfer gates fail.
The separately registered 512px diagnostic reaches 6.08 degrees, 38.47 degrees,
25.97% recall and 11.69% AUC@10, respectively, but still fails both gates. These
are fixed readouts of the same checkpoint; known intrinsics enter only the CPU
solver. Resolution results do not select the longer recovery arm.

Main training consumed **8,186.91 command seconds (2.274 hours)**. Warm median
and p95 update times are 0.65688 and 0.70170 seconds; peak process VRAM is
27,380 MiB (26.74 GiB). Observed board energy is 917.52 Wh, mean board power
403.51 W and median device utilization 94%. Desktop activity is included in
board measurements; these are neither process-attributed energy nor occupancy.

The reviewed single-run [page](../../.data/publications/pilot11-main-reviewed/index.html)
and [29-page PDF](../../.data/publications/pilot11-main-reviewed/paper.pdf) retain
learning curves, six fixed completion examples, matching/pose annotations and
all failed qualifications. The original render is preserved separately. Review
corrected only the PDF plot's bounding box; both machine-readable result files
are numerically unchanged. The separate post-screen full recovery subsequently
completes the matched 12,000 updates and fails the unchanged completion-retention
gate; its [full result](pilot-11-recovery.md) retains both synthetic objectives
and the negative decision. The tail main remains selected.

## Study closeout

The [dispatch trace](pilot-11-dispatch-probe.md) and
[batch-capacity probe](pilot-11-batch-probe.md) are completed diagnostics with
discarded weights. Batch 32 passes capacity and correctness checks but misses
its 20% throughput threshold, with only 14.64--16.33% more targets/s at 1.965
times the process VRAM. Batch 16 and the selected quality endpoint remain fixed.

All 21 GPU-command legs have finished. The cumulative ledger records
**26,476.85 / 43,200 seconds (7.355 / 12 hours)**, including the timed-out first
preflight, both screens, both full-length continuations, capture, inference,
profiling and batching. The remaining **16,723.15 seconds (4.645 hours)** are
unused; this ceiling is not a requirement to consume the full allowance.
Pilot 08/09/10 receipts and frozen Pilot 10 artifacts remain unchanged.

Study-owned process monitors are stopped. Native process-counter analysis
retains shared desktop activity and unavailable samples: GNOME Shell has 2,716
numeric samples with a 19.56% mean reported counter, while remote desktop has
only 239 numeric samples (28.46% mean) and 2,478 unavailable samples. These
percentages are not additive, time-weighted occupancy or process energy.
CPU process snapshots are lifetime averages, not instantaneous utilization.

The final `.data/pilot-11/closed.json` binds the current source archive, budget,
native evaluations, replay checks, reviewed bundles and retained negative
decisions. Research changes remain uncommitted. Matching qualification is a
development result; camera transfer, suppressed spatial variation, strict
reference-order invariance, training-seed replication and public baseline
protocols remain open quality gates. No sharp-RGB or SOTA claim is made.
