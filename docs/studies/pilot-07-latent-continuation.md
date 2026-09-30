# Pilot 07: controlled latent continuation and correspondence qualification

This continuation targets useful geometric representations, with fixed V-JEPA
latent prediction as the primary task. RGB reconstruction is an optional
historical diagnostic; its blur remains unresolved. The work implements a
stronger experiment and evaluation pipeline, not a state-of-the-art claim.

The [annotated PDF](../../.data/pilot-07/latent-continuation-report.pdf) and its
[structured results](../../.data/pilot-07/latent-continuation-report.json) are generated
from the [TOML report recipe](../../configs/archive/pilot-07/pilot07-latent-report.toml).
The earlier [1,000-update screen](pilot-07-latent.md) remains a separate record.

## Method and selection

Both arms start from the project's own 1,000-update latent checkpoint. Student
and fixed teacher originate from the audited MIT V-JEPA 2.1 Base package; fusion,
latent projection and RI heads originate from random initialization. No
noncommercial checkpoint or teacher enters the lineage. Warm starts verify model
hashes and recursively audit provenance; exact resumes verify both optimizers.

At 256px, each target retains 26 of 256 patches before encoder attention. Two
independently encoded reference views enter a shared six-layer, 384-wide fusion
trunk. Targets are normalized final-layer teacher features; the teacher remains
fixed. This is a squared-error adaptation objective, not a reproduction of
V-JEPA's hierarchical momentum-teacher pretraining. Geometry remains evaluation
only. The shared latent head supports cross-view and matched monocular branches.

The controlled phase uses the same 1,024 training rooms, batch 16, sample order,
learning rates and validation gate in both arms. It compares random visible
patches with a compact connected visible island at 90% masking. The island is
not V-JEPA's multi-block policy. Each arm restarts the two AdamW optimizers and
the unfreezing gate, then runs 1,000 accepted updates. Their adaptive stage
exposures differ: random has 200 frozen, 400 partial and 400 full-encoder updates;
the island has 200 frozen and 800 partial updates. The outcome compares these
masking policies under the same adaptive gate, not equal trainable-block exposure.

The island arm ran out of memory at update 896. Its verified update-800 checkpoint
was resumed with both optimizers and finished update 1000. Failed updates after
800 in the original process are excluded from the accepted trajectory; their
command cost is retained. The random arm was selected by common-mask validation
and a correspondence regression gate. It then continued exactly from 1000 to
2000 through the existing learning-rate horizon, with no optimizer reset.

All four checkpoints are assessed by one archived final-only evaluator, using the
same 16 validation rooms and fixed random 90% mask. Do not compare these losses
to the earlier screen's 75% task. Selection rules and timing are recorded in
[pilot07-latent-selection.toml](../../configs/archive/pilot-07/pilot07-latent-selection.toml),
[pilot07-latent-final-qualification.toml](../../configs/archive/pilot-07/pilot07-latent-final-qualification.toml)
and [.data/pilot-07/final-selection.json](../../.data/pilot-07/final-selection.json).
No fresh-test or HPatches scores were available at final checkpoint selection.

## Common synthetic validation

| Checkpoint | Hidden latent MSE | Matched mono MSE | Spatial variance / teacher | Encoder EPE, px | Decoder EPE, px |
| --- | ---: | ---: | ---: | ---: | ---: |
| Parent | 0.23862 | 0.25167 | 0.313 | 30.34 | 31.16 |
| Random 90%, +1000 | 0.22245 | 0.23704 | 0.310 | 29.52 | 29.70 |
| Compact island 90%, +1000 | 0.23004 | 0.24526 | 0.294 | 30.19 | 30.64 |
| Selected random, +2000 | **0.21747** | 0.23262 | 0.343 | **29.41** | **29.45** |

The selected checkpoint reduces parent latent MSE by 8.86%. The paired room
improvement is 0.02115, 95% CI [0.01921, 0.02306]. Its matched monocular contrast is
0.01515 [0.01163, 0.01855], a 6.51% relative error reduction. Unrelated references
and shuffled reference positions both worsen prediction. The shuffle retains
position information embedded in features, so this is evidence of layout
sensitivity, not proof that all scene-context shortcuts are removed.

Encoder room-mean EPE improves by 0.929px [0.553, 1.288] versus the parent; decoder
EPE improves by 1.806px [1.178, 2.474]. Aggregate EPE above pools visible points,
whereas those intervals weight rooms equally. The unchanged encoder's EPE is
31.82px, same-position is 33.76px and the patch-grid oracle is 6.14px. The sizable
remaining gap and 0.343 feature variance ratio constrain any quality claim.
Only one training seed and one masking seed have been studied.

## Fresh rooms and real-image transfer

A new capture contains 256 test rooms at 256px with three views each, from seed
2609400000. Its one required train and one validation room are unused. The
assessment recursively rejects overlap with every ancestor's training and
selection seeds. All 768 test targets and 1,536 directed camera pairs are scored.
Published generator pins remain `bevy_zeroverse=0.25.0` and
`bevy_zeroverse_burn=0.8.0`. An independent full-pixel audit checks 50.7 million
source pixels: maximum self-reprojection error is 0.00287px and depth error is
1.91e-5m. This tests fresh room seeds under the same camera/generator distribution.

Fresh-test latent MSE is **0.21830**, versus 0.23940 for the parent: an 8.81%
reduction, with paired improvement 0.02110 [0.02068, 0.02153]. The matched
monocular result is 0.23241, giving a 6.07% cross-view reduction and paired
contrast 0.01410 [0.01311, 0.01507]. Variance ratio remains only 0.339. Learned
RI AUROC is 0.666 and AP 0.905 against positive prevalence 0.839; the score is
not a calibrated geometric visibility probability.

| Fresh-test readout | Parent EPE | Selected EPE | Parent-minus-selected room-mean EPE, 95% CI |
| --- | ---: | ---: | --- |
| Encoder cosine | 32.07 | 31.25 | 0.808 [0.724, 0.893] |
| Fused decoder cosine | 32.92 | 31.30 | 1.657 [1.502, 1.816] |
| Reciprocal attention | 33.36 | 32.71 | 0.648 [0.604, 0.693] |

The unchanged encoder is 33.51px, same-position is 35.81px and the grid oracle is
6.11px on these rooms. All matching uses RGB only; renderer visibility filters
scoring queries after prediction, not the candidate correspondence search.

HPatches uses all 116 sequences / 580 pairs from its checksum-pinned author-hosted
archive. The primary viewpoint subset matches ZeroCo's official 59-sequence /
295-pair list; an independent homography audit agrees within 4e-11. Models receive
only RGB resized to 256px. A separate CPU scorer applies homographies at 240px
and bilinearly upsamples hard patch displacements. This local readout differs
from published refinement recipes, so published-number parity is not claimed.

| HPatches readout | Primary viewpoint AEPE | Viewpoint PCK3 | Supplementary illumination AEPE |
| --- | ---: | ---: | ---: |
| Frozen V-JEPA encoder | 34.08 | 7.11% | 10.81 |
| Adapted encoder | **32.54** | **7.69%** | 10.07 |
| Centered frozen encoder | 32.78 | 7.71% | 10.05 |
| Centered adapted encoder | 31.01 | 8.27% | 9.39 |
| Fused decoder | 36.65 | 5.36% | 5.39 |
| Fused latent projection | 37.92 | 4.96% | 5.96 |
| Reciprocal attention | 43.32 | 1.04% | 1.03 |
| Same position | 44.63 | 0.69% | 0.15 |

Encoder adaptation improves primary viewpoint AEPE by 1.534px [1.256, 1.810],
or 4.50%, against the matched frozen readout. Centering is a separate declared
control: centered adaptation gains 1.772px [1.447, 2.095] over centered frozen
features. These intervals resample the 59 sequences, preserving each sequence's
five pairs. They do not capture independent-training-seed uncertainty.

**Fusion transfer is the current failure.** Relative to the frozen encoder,
decoder AEPE worsens by 2.572px and attention by 9.243px, with intervals excluding
zero. On viewpoint pairs, attention predicts the identical patch index 60.1% of
the time and stays within one patch 94.7% of the time. The encoder does so only
4.5% and 15.9%. This is consistent with a strong position prior in the attention
readout; it does not yet isolate data, objective, positional encoding or head
averaging as the cause. Illumination images largely preserve geometry, so the
same-position baseline is already excellent there. Combining both subsets can
hide the viewpoint failure. All readouts and failures are retained in the report.

The PDF shows the first two fresh room seeds and fixed first/median HPatches
viewpoint sequences, with green/red prediction errors and cyan geometric truth.
Latent maps share a teacher-fitted PCA basis and color limits. They are feature
maps, not reconstructed RGB. No further quality training, checkpoint selection
or readout tuning used the external results.

## Native memory diagnosis and engineering verification

Unused hierarchical output norms caused sustained memory growth while only the
last encoder blocks were trainable. The latent objective consumes final tokens,
but the old forward created normalized intermediate branches that disconnected
from the backward traversal. The vendored encoder now provides
`forward_image_capture_layers`, and the latent path requests no intermediate
captures. Default hierarchical behavior remains available for losses that use it.
Loss logging also uses the non-autodiff inner backend.

An isolated CUDA audit used identical RGB, initialization, deterministic synthetic
feature targets, batch 48 and the last two trainable blocks for 200 updates. It
performed no manual allocator cleanup. Legacy capture grew 113.2 MiB/update and
peaked at 36.70 GiB; final-only stayed at 14.92 GiB. Median updates were 136.9ms
and 136.0ms. Maximum difference between loss trajectories was 1.19e-6. This is a
memory diagnostic, not an additional quality checkpoint. CPU tests verify exact
dense/sparse final-output and active-gradient parity.

The full suite passed 84 Rust tests, 8 Python tests and strict all-target CUDA
Clippy. Native CUDA binaries and source archives are sealed with SHA-256 records.
The quality continuation retains its original binary for exact resume; new
training uses the repaired code. A float64 Fusion preflight failed before any
updates, so the supported native policy remains float32. Measured reference-order
rounding residuals are retained; hidden-RGB perturbations produce zero change.

## Compute and reproducibility

The selected lineage has 3,000 updates, 48,000 target exposures and 1,024 distinct
rooms. The complete capture contains 8,192 available training rooms; unused rooms
are not counted as training exposure. Full-encoder continuation took 17.9 minutes
including preparation/evaluation, with median 0.823s/update, 19.43 targets/s and
29.92 GiB peak process VRAM. Device-wide median utilization was 97%, including
preparation, evaluation and any desktop activity. These are workstation pilot
measurements, not a multi-GPU scaling claim.

All generated data, checkpoints, immutable source/binary copies, predictions and
reports are under `.data/`. TOML recipes are under `configs/`. Every capture,
training, native audit and GPU evaluation command debits the existing cumulative
43,200-second Pilot 07 ledger, including failed attempts. CPU analysis/reporting
and the external archive download are excluded under its recorded scope. The
final PDF reads the closed ledger rather than resetting the ceiling.

This continuation closes with **7,128.1 seconds / 1.98 hours** of cumulative
Pilot 07 command time, below the unchanged 12-hour ceiling. This includes earlier
Pilot 07 legs, not just the experiments described here. All GPU commands finished.
The selected model SHA-256 is
`c83bc1fc4d6fdd7b57f63eee24d85cdd1e41e93ca377c489e122b0e65685b64b`.

Rebuild the derived report with:

```sh
OPENBLAS_NUM_THREADS=1 .data/analysis-venv/bin/python tools/legacy/latent_study_report.py \
  --config configs/archive/pilot-07/pilot07-latent-report.toml
```

Raw prediction scoring is reproducible with `tools/legacy/hpatches_score.py` and
`configs/archive/pilot-07/pilot07-hpatches-score.toml`, using a new output path because scored
receipts are immutable. `tools/legacy/hpatches_displacement.py` reproduces the positional
diagnostic. The native plans, actual run configs, binary/source hashes and failed
command logs are retained alongside the report.

## Next qualified experiments

Prioritize the measured viewpoint-transfer failure before a longer same-objective
run. Separate hypotheses are broader camera-baseline distributions, frozen versus
adapted encoders, hierarchical teacher targets, a self-supervised augmentation
correspondence objective, and local refinement. Use controlled data, compute,
initialization and validation for each; renderer geometry must stay outside the
self-supervised loss unless the variant is explicitly supervised. Preserve
HPatches as held out, or explicitly declare development reuse before another
optimization cycle informed by its scores. Exact ETH3D protocol parity, current
matched baselines, independent seeds and visibility calibration remain claim gates.
