# Pilot 11: secondary frozen-checkpoint resolution probe

Registered before the main training phase and before higher-resolution inference.
This is an additional development diagnostic within the same authorized 12-hour
ledger, with at most **900 GPU-command seconds** reserved. The original 256px
training screens, selection gates, main schedule and primary evaluations remain
unchanged. Reserve 2,700 seconds after main training: the original 1,800 seconds
for primary evaluation plus this 900-second diagnostic. Skip the secondary probe
if the cumulative allowance cannot support it; do not shorten training based on
external accuracy.

The encoder already constructs its token grid from the input dimensions and
interpolates spatial RoPE positions. Fusion also takes an explicit image grid.
The current external exporters and camera scorer instead assume 256px/16x16.
Generalize the native TUM preparation, pose exporter, scorer and visualizer to
also accept 512px/32x32. Do not modify the imported encoder, fusion position
encoding, descriptor head, temperature, local-centroid operator or weights.

Use the fixed final main checkpoint at **512 by 512** on the same 186 eligible
TUM Freiburg 3 pairs. Decode original 640x480 RGB and resize directly with the
same native triangle filter; do not enlarge the cached 256px images. Preserve
exact pair IDs, ground-truth transforms and original-pixel intrinsics. The cohort
is development data after Pilot 10. Keep its one low-baseline exclusion and all
solver failures. The three descriptors, mutual-mask rule, 3x3 local centroid,
eight-point RANSAC, three-original-pixel threshold, trial limits and seed match
the primary probe. The denser grid increases correspondence count and changes
the physical extent of the local neighborhood; this is an inference resolution
tradeoff, not a new learned refinement method or official benchmark protocol.

Before using the generalized exporter at 512px, replay all 186 original 256px
pairs with the frozen Pilot 09 checkpoint. All hard indices and mutual flags must
match the original Pilot 10 export exactly; local coordinates must agree within
1e-5 grid cells. CPU tests cover grid validation and half-pixel mapping at both
resolutions. If replay or shape/finite-output checks fail, retain the failure and
do not promote the new exporter. All replay time counts against the 900-second
secondary cap and the cumulative study allowance.

Report failure-inclusive angular errors, pose recall and AUC, per-sequence
results, match counts, latency and memory. Preserve the primary 256px report.
If completed, generate a separate annotated 512px diagnostic bundle using the
same single main training run/checkpoint and explicitly reused completion
evidence. Neither resolution result selects a training arm, checkpoint, solver
threshold or subsequent training horizon. Record regressions as well as gains;
no SOTA claim follows from this small reused cohort.

## Completed selected-tail diagnostic

The generalized exporter exactly reproduces all 558 original 256px prediction
rows: no hard-index, mutual-mask or coordinate differences. Its 26.33-second
replay qualifies the export path before higher-resolution inference.

The fixed tail main checkpoint
`bdd1bd204afc5584ce7277af40d8bea80f6987c6611937e79b7fb13074789637`
then completes both resolutions on all 186 development pairs. Results below are
fusion readouts of these same weights; the solver and pair population are fixed.

| Metric | 256px primary | 512px secondary |
| --- | ---: | ---: |
| Mean rotation error | 8.61 degrees | 6.08 degrees |
| Mean signed translation-direction error | 54.99 degrees | 38.47 degrees |
| Pose recall within 10 degrees | 14.61% | 25.97% |
| Pose AUC@10 | 6.74% | 11.69% |
| Solver returned an estimate | 184 / 186 | 186 / 186 |

At 512px, the same-image control's AUC@10 is 11.25% and the encoder's is 12.18%.
Fusion fails the registered positive-gain-in-every-sequence gate against **both**
controls. Higher resolution helps absolute pose accuracy here, but does not
resolve the fusion qualification failure or establish a trained camera head.
All failures, the one translation/pose baseline exclusion and per-sequence
results remain in the native score receipt.

The 512px export consumes 106.86 command seconds. Combined with exporter replay,
133.19 of the shared 900-second secondary reservation has been used; this is a
subset of the same 12-hour study ledger, not a separate allowance. Any later
accepted recovery diagnostic must use its remaining portion.

The reviewed [page](../../.data/publications/pilot11-main-pose512-reviewed/index.html)
and [PDF](../../.data/publications/pilot11-main-pose512-reviewed/paper.pdf) use the
single main training run. Completion and training-efficiency evidence explicitly
remain at 256px. The denser correspondence grid changes the number of queries
and physical neighborhood size; no isolated resolution mechanism is claimed.
