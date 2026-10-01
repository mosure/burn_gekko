# Pilot 13: actual-view geometry auxiliary screen

Registered before new GPU inference, 2026-10-01. Residual GPU-command allowance is 2982.4822693681344 seconds from the same 12-hour authorization. No renewed allowance is inferred. Prior Pilot 12 closeout is pinned in the budget ledger.

Hypothesis: RGB homography supervision does not span depth-dependent viewpoint changes. Add a renderer-supervised bidirectional correspondence NLL to the pair-conditioned descriptor, with weight 0.1 and temperature 0.07. This is an explicitly geometry-supervised auxiliary, not self-supervision. Geometry supplies detached labels only; cameras, positions and depth are never inference inputs. Retain all previous latent, RI, homography and final-feature-preservation losses.

Two fixed matched arms start from Pilot 12 main b8e5d38c3a62eefd2b2d7ebc89e5b2d47b6ee91ac844d7e3d6eb257ae487aad8. Both reset optimizers and train all encoder tensors. Batch 16, 2048 training rooms, 32 validation rooms, 384 updates (6144 target exposures; one complete room-view epoch), seed 829, LR 2e-5, encoder ratio 0.02, 32 warmup updates, cosine 384. Mask 90%, two references, preservation 4 against the unchanged Pilot 11 anchor. Control has no geometry objective. Extra decoder work is measured, not called matched compute. Only the geometry coefficient/objective differs. Endpoint selection is fixed, not best-validation checkpoint.

Qualification: analytic camera/translation/visibility tests; CPU exact optimizer-resume and immutable label-cache checks; full CUDA compile and strict Clippy; then 8-update CUDA preflight. Require finite active geometry loss, positive valid fraction, all 151 encoder gradient tensors, unchanged teacher and anchor probes, and no memory failure. Forecast screen runtime from preflight plus measured control timings; reserve >=1200 seconds for evaluation/closeout. If the fixed plan cannot fit, do not silently shorten it or consume the reserve.

Synthetic selection: measure actual view 0 versus view 1 in both directions in the same 32 validation rooms, with identical known visible queries. Local 3x3 refined readout is primary; hard, same-image and encoder controls are retained. Accept candidate only if validation completion MSE <=1.01 control, reference completion improves on monocular, mean view pixel error <=0.95 control, and PCK within 8 pixels is nondecreasing. Cache and query identities, source, teacher, sample order, recipe and anchor integrity must match. Results from HPatches, ETH3D and TUM cannot change selection. Both fixed endpoints will receive the complete development protocols if the runtime forecast fits. Do not omit a rejected candidate from internal analysis.

External evaluation uses existing complete protocols: HPatches580 pairs, canonical ETH3D3365 pairs, calibrated TUM186 pairs at256px with one predefined tiny-baseline exclusion. Both hard/refined matching and every transfer gate remain visible; camera pose is a calibrated solver probe, not a learned camera head. One seed and reused development cohorts cannot establish SOTA. A single-run paper/page, if generated, uses only the selected checkpoint and its own controls. Completion samples remain latent visualizations, never RGB reconstructions or RGB PSNR.

Stop rules: no long continuation this study; no post-result coefficient sweep; retain failed commands and incomplete populations without scoring them as complete. All GPU legs are charged by the shared budget runner. Shared desktop activity stays running. Board telemetry and process VRAM are recorded; the separate process-activity observer failed at startup, as detailed below. Preserve all Pilot 12 artifacts unchanged.

Status: **completed; geometry selected by the registered synthetic gates**. Both arms completed 384 updates and 6,144 distinct room-view exposures, followed by every external evaluation. A conditionally registered independent completion cohort also completed. No longer continuation or coefficient sweep was launched. GPU jobs are finished. This study establishes a useful spatial-descriptor improvement; camera transfer remains mixed and SOTA is not established.

The native target-cache audit covers 12,288 directed training pairs and 192 validation pairs, with no empty pairs. Mean valid queries are 119.77 and 129.22 of 256 respectively. Synthetic scoring uses 8,433 known visible queries across 32 validation rooms and two directions. These renderer labels remain separate from RGB model inputs.

## Implementation and qualification

`burn_gekko_data::view_targets` constructs a compact, content-bound cache from
the published renderer's positions, depth and camera metadata. Each query stores
a visibility state and projected pixel coordinates. The source is the central
2 by 2 pixels of a patch: all depths must be positive, the world-position spread
must not exceed max(2 cm, 1% of depth), and source reprojection must be within
0.1 pixel. Reference depth agreement uses max(2 cm, 1% of projected depth).
Unknown, occluded, out-of-view and visible states remain distinct. Bilinear
labels must lie inside the descriptor-center hull; border coordinates are
excluded instead of clamped. Cache identity includes room splits, source-shard
hashes, labeling policy, labeling source and target-byte hash.

`burn_gekko_train::training::latent::view_geometry` reuses dense target/reference
encoder outputs, rotates the reference by update, and trains both directions
through the existing pair-conditioned spatial decoder. Detached renderer labels
add two decoder branches, no encoder forward or inference parameter. The
training cache and policy are pinned in exact-resume identity. Native evaluation,
matched-arm selection, provenance checks, metrics and figures live in their
respective `burn_gekko_eval` / `burn_gekko_report` crates. User configuration is
TOML; caches, weights and generated artifacts remain under `.data/`.

Analytic identity, translated-camera, visibility and boundary tests pass. The CPU
pipeline verifies exact optimizer resume, immutable cache identity, split guards,
teacher/anchor immutability and the active gradient path. The CUDA preflight
passes all registered checks. Both training arms show all 151 encoder gradient
tensors and unchanged teacher/anchor probes. Geometry NLL averages 2.7680 over
the first 32 updates and 2.4189 over the last 32; usable label fraction averages
46.54% and 47.10%. This is a short optimization screen, not a convergence result.

## Synthetic selection and external matching

| Fixed endpoint metric | Control | Geometry auxiliary |
|---|---:|---:|
| Validation completion MSE | 0.171102 | 0.171700 |
| Monocular MSE | 0.198573 | 0.198813 |
| Actual-view mean error, 256px input frame | 16.6651 px | 12.1806 px |
| Actual-view matches within 8px | 49.01% | 58.72% |
| HPatches viewpoint mean error, 240px scoring frame | 16.9341 px | 14.8376 px |
| HPatches viewpoint matches within 3px | 24.02% | 26.23% |
| ETH3D mean error, original image frame | 28.4394 px | 23.8085 px |
| ETH3D matches within 3px | 5.17% | 5.68% |

The candidate reduces synthetic view error by 26.9%, with a 9.72 percentage-point
PCK8 gain and a 0.35% completion-MSE increase, within the registered 1% limit.
All four selection gates pass. Selection is frozen in
`.data/pilot-13/screen-selection.json` before external inference. The selected
checkpoint is `.data/runs/pilot-13-geometry/final`, SHA-256
`0639afa2e19d0e7ac5c9c1cc8caadc75d0dcf10f105ee2870607856b7a1ed5a7`.

Both endpoints score all 580 HPatches pairs (295 viewpoint pairs primary) and
all 3,365 canonical ETH3D pairs. The local spatial readout reduces external mean
error by **12.4% / 16.3%** versus the matched control. These are one-seed arm
effects on reused development data; no training-seed significance is claimed.
Within the selected checkpoint, all five declared contrasts on each benchmark
pass: coarse and refined pair-conditioned descriptors versus equally processed
encoder and same-image controls, plus local precision versus the coarse readout.
The previously failing coarse ETH3D encoder-control gate now passes. Its mean
PCK3 gain is only 0.0375 percentage points and its 95% interval still includes
zero; the registered gate requires nonnegative mean precision, not a positive
precision interval. The refined ETH3D gains have positive error and precision
intervals against both controls. These results qualify the fixed spatial readout
family, not every raw decoder feature or attention layer.

ETH3D uses canonical broad calculations. Hard/local readouts share score arrays
and pass exact hard-index and mutual-flag consistency on every pair. The earlier
optimized-export parity failure and encoder startup numerical discrepancy are
retained as unresolved historical limitations; no cause is asserted here.

## Camera transfer remains incomplete

The calibrated TUM probe scores all 186 input pairs, with 185 pose-eligible pairs
after the predefined tiny-baseline exclusion. Solver failures stay in denominators.
The candidate has 7.78-degree mean rotation error, 52.80-degree signed translation
direction error, 21.10% pose recall within 10 degrees, and 7.65% AUC@10. These are
sequence-macro metrics. The control has 8.18 degrees, 49.84 degrees, 18.40% recall
and 8.35% AUC@10 respectively: higher recall at one threshold does not mean a
uniform camera improvement.

Candidate transfer passes against the encoder in every sequence, but fails
against its same-image control on `structure_texture_near` (AUC@10 gain -1.704
percentage points). The other two gains are +3.054 and +1.972 points. Both
control-arm camera contrasts fail their all-sequence gates. The parent Pilot 12
checkpoint passed both gates and had 8.50% AUC@10; the new objective must not be
presented as an overall pose improvement. The fixed synthetic selection remains
unchanged by these external findings. Known intrinsics and a native essential
matrix solver produce these estimates; a learned camera head remains untrained.

## Independent completion and co-visibility

The original 128-room Pilot 12 cohort is reused as development evidence only:
512 targets, MSE 0.187221, monocular MSE 0.202061. A fresh cohort was conditionally
registered after synthetic selection and before generation or fresh outcomes,
independent of external accuracy. Its >=360-second reserve condition passed.
Seed range 2610091000--2610091129 is disjoint from every prior completed dataset
manifest. The first two rooms fill train/validation placeholders; the remaining
128 rooms form the test set. The pinned published capture binary generates four
views at 256px; native dataset verification passes. No fitting, checkpoint
selection or setting change uses these results.

All 512 fresh targets are evaluated and exported with two references and 90%
random masking, seed 829. The primary publication uses only this fresh cohort:

| Completion / co-visibility measure | Result |
|---|---:|
| Hidden-token feature MSE | 0.184311 |
| References-disabled MSE | 0.198570 |
| Reference error reduction | 7.18% |
| Paired MSE reduction, room-bootstrap 95% interval | 0.014259 [0.012480, 0.016027] |
| Feature cosine | 0.90256 |
| Feature signal/error | 7.38 dB |
| Teacher spatial variation retained | 43.56% |
| Reference positions shuffled MSE | 0.194029 |
| Unrelated reference room MSE | 0.241977 |
| Training position-mean baseline MSE | 0.308393 |
| Learned RI co-visibility AUROC / AP | 0.7382 / 0.9074 |

Co-visibility scores cover 117,715 known hidden patches, 91,472 majority-visible.
RI uses a separate full-target branch; it is neither sparse completion nor a
calibrated visibility probability. The majority-visibility evaluation policy
differs from the patch-center training supervision. Held-out labels never enter
training or model inputs. Feature signal/error dB is **not RGB PSNR**.

Hidden RGB isolation and monocular reference permutation are exact. Strict
cross-view reference-order invariance fails: maximum difference 0.00375938,
RMS 0.000171087, tolerance 0.00001. Latent maps still lose substantial spatial
variation. RGB, camera and depth heads remain untrained; changing the target
space has not established an RGB blur fix.

## Efficiency, failures and budget

| Training measurement | Control | Geometry auxiliary |
|---|---:|---:|
| GPU-command duration including startup/evaluation | 531.31 s | 605.15 s |
| Median update, excluding initial 10 updates | 1.0611 s | 1.2554 s |
| Observed board joules per target | 29.75 | 33.50 |
| Peak process VRAM | 44,916 MiB | 49,110 MiB |
| Observed mean board power | 344.69 W | 340.68 W |
| Telemetry coverage | 99.80% | 99.83% |

Extra decoder work costs 18.3% longer updates and 12.6% more gross board energy
per target. CPU builds/tests did not overlap either training measurement window.
Final CPU checks did overlap external export, so their time is not a clean
inference-throughput comparison. Shared desktop processes stayed running. The
supplementary `nvidia-smi pmon -d 30` observer failed immediately because the
maximum supported interval is 10 seconds. No historical per-process SM samples
exist. The raw usage error and failure receipt are retained. Board telemetry and
process memory remain valid; desktop power contribution is unresolved. This is
not an occupancy measurement or evidence of a dispatch optimization.

The initial workspace test attempt failed in a capture fixture after accepting
an unintended setup error. A tighter assertion exposed a concurrent executable
fixture race (`ETXTBSY`, repetition 68). A test-only mutex now serializes the
three capture fixture launches. All 100 post-fix stress repetitions pass, as do
**152 workspace tests**, CUDA strict Clippy over owned crates/all targets and
owned-crate formatting. Production capture behavior was not changed. The initial
failures and successful reruns remain retained. The imported audited encoder's
preexisting formatting difference is outside the owned-crate formatting claim.
Report-only wording refinements subsequently pass scoped tests, Clippy and format.

This study spends **2,621.266630 seconds** from the residual allowance. Combined
Pilot 11--13 usage is **42,838.784361 / 43,200 seconds (11.900 / 12 hours)**,
leaving **361.215639 seconds (6.02 minutes)**. Every training, inference and capture
command is charged; no new allowance is inferred. There are no running model jobs.

## Review artifacts and next controlled question

The [30-page reviewed PDF](../../.data/publications/pilot13-geometry-reviewed/paper.pdf)
and [project page](../../.data/publications/pilot13-geometry-reviewed/index.html)
show this one selected checkpoint and its own controls. They include six evenly
spaced completion examples, known-view geometry, real matching, camera diagnostics,
uncertainty, energy and failed gates. Internal arm comparisons stay in this study.
Native validation checks 114 hashes, 97 images and 42 local links; the browser
passes six sample choices at 390/768/1440px without overflow or JavaScript errors.
Cover, completion and correspondence PDF pages and desktop/mobile page views
were visually reviewed. The original bundle remains available separately.

Rebuild locally from the pinned experiment (into a fresh output directory):

```sh
cargo build --profile pilot --locked -p burn_gekko_report
target/pilot/gekko-report build \
  --experiment configs/publish/pilot13-geometry-reviewed.toml \
  --output .data/publications/pilot13-reproduction --pdf
target/pilot/gekko-report validate --bundle .data/publications/pilot13-reproduction
```

The next controlled question is whether sustained actual-view supervision
preserves this matching gain while retaining camera AUC and completion detail.
A future study should preregister a matched continuation and training-seed
replication, with completion retention and calibrated camera retention measured
on separate synthetic validation rooms before new real-data qualification.
Longer training is not justified by this short screen alone. Pose-consistency
supervision or a learned camera head would be separate architectural experiments,
not a post-hoc adjustment selected on these three TUM sequences. Independent
real cohorts and protocol-matched public baselines remain prerequisites for SOTA.
No additional training, commit, push or deployment is included in this closeout.
