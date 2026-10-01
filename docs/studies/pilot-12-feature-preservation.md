# Pilot 12: preserve final encoder features while adapting geometry

Registered before GPU preflight or any new training. Pilot 11's matched full
adaptation improves synthetic transform matching but leaves completion MSE
4.99% above the tail-only control after 12,000 updates. This study tests whether
an explicit final-feature anchor can reduce that tradeoff. It does not assume
that the earlier result identifies a gradient mechanism.

## Budget and immutable inputs

Use only **16,723.149945212062 GPU-command seconds (4.645 hours)** remaining from
the user's existing 12-hour authorization. Pilot 11 consumed 26,476.850054787938
seconds. Keep its completed ledger, source snapshots, weights and publications
unchanged. Pilot 12 has a separate residual ledger linked to that closeout;
combined usage cannot exceed 43,200 seconds. CPU implementation, scoring and
reporting are excluded. No commit, push, deployment or crate release is planned.

Use the same published Zeroverse 0.25.0 / adapter 0.8.0 cached 256px rooms:
`.data/datasets/e54ba670d6087c445d1d04c7c3ba1742d0e610d08533690921ac655f1f66f05b`.
The model and frozen feature anchor both begin from Pilot 11's selected endpoint:
`bdd1bd204afc5584ce7277af40d8bea80f6987c6611937e79b7fb13074789637`.
Its audited MIT V-JEPA ancestry remains mandatory. No released noncommercial
Gekko weights or targets are used.

## Objective and implementation qualification

The new optional TOML `encoder_preservation` objective uses a frozen copy of an
explicitly pinned own encoder. Compute raw final-feature squared error divided
by the anchor's mean squared activation, independently for each image, with an
energy floor of 1e-6. Average equally over four routes: sparse target, full
target, and both full reference views. Weight this relative feature MSE by
**4.0**. Retain offsets and magnitudes. The intermediate block-6 features are
not directly anchored. The frozen encoder supplies loss targets only and is
absent from inference and model records. Its weight dependency is recorded in
checkpoint ancestry and is reloaded from the same source on exact resume.

Test per-image normalization, stop-gradient behavior, zero/offset cases,
checkpoint identity and exact optimizer resume. Perform two discarded CUDA
preflights, each 64 updates with a 600-second command cap:

1. Disabled objective: replay the first 64 updates of Pilot 11's batch-16
   A-before control using its original parent, seed 797 and 12,000-step schedule.
   Require native sample/stage/scalar replay within 1e-6 absolute + 1e-5 relative.
2. Enabled objective: the new full preserved recipe, but stop at 64 updates.
   Require finite losses, all 151 encoder gradient tensors, changed first/last
   QKV and prediction-head probes, unchanged teacher and anchor QKV probes,
   initial preservation error below 1e-6 and nonzero error after adaptation.

Seal the new trainer and source before GPU use; retain the old executable.
No allocation or nonfinite failure is retried automatically. Preflight timings
determine whether the complete fixed study fits; do not shorten horizons after
seeing quality results. Record shared desktop load without stopping it.

## Matched screen

Run tail control, unrestricted-full control, then preserved-full candidate.
All start from the same pinned parent and reset both AdamW optimizers. Keep
8,192 rooms, 2,048 updates, batch 16, seed 823, two references, 90% random
masking, 32 validation rooms, and the existing spatial head/latent/RI/warp losses.
Use trunk LR 4e-5, encoder LR 8e-7, 100 warmup updates and a 2,048-update cosine
horizon with minimum ratio 0.1. Validate every 512 and checkpoint every 1,024.
Use identical 4,080-second internal limits and 4,200-second command caps. The
global watchdog always takes precedence. Tail fixes stage 1; both full arms fix
stage 2. Only the preserved arm adds the pinned feature objective.

Each arm receives the same 32,768 ordered target exposures. After all screens,
score their complete validation populations and the same 32-room RGB-only
known-transform diagnostic, seed 1000823, with hard and fixed local readouts.
Native selection verifies exact recipes, ordered samples, masks, transform
queries, checkpoint sources, intended trainable stages and frozen probes.

The preserved arm qualifies only if all conditions hold:

- Completion MSE is no more than 1% above tail and lower than unrestricted full.
- Reference views reduce completion MSE.
- Local transform error is at least 10% below tail, with no decrease in PCK8.
- All source, coverage, gradient, teacher and anchor checks pass.

Otherwise retain the tail control as the screen outcome and reject this
preservation recipe. Do not change the coefficient or thresholds afterward.

## Real-view transfer and optional continuation

After sealing the synthetic decision, export **all three** screen endpoints on
all 580 HPatches pairs, 3,365 ETH3D pairs and the existing 186-pair 256px TUM
cohort. These are development diagnostics and never select the arm. Evaluate
rejected arms too: synthetic geometry and completion alone cannot establish
real-view transfer. Retain equally processed same-image and encoder controls,
complete populations, paired intervals and camera failures. TUM uses known
intrinsics and a CPU solver; it is not a learned camera head.

If the preserved arm qualifies and measured timing permits, run a fixed
**4,096-update** weights-only continuation: seed 827, batch 16, full stage 2,
trunk LR 2e-5, encoder LR 4e-7, 100 warmup updates, 4,096 cosine horizon,
validation every 512 and checkpoints every 2,048. Keep the feature anchor at
the original Pilot 11 endpoint, not the new screen parent. Reset both AdamW
optimizers. Require a forecast of 4,096 times preserved-screen p95 times 1.10,
plus 600 seconds preparation/evaluation/checkpoint reserve and a further 1,800
seconds for final inference/capture. Skip this phase if it does not fit;
never adapt its horizon to observed validation or real-view metrics.

The final endpoint is fixed before fresh capture. Capture 128 new four-view
rooms, seed 2610081000, with the same rendering settings and mask seed 827;
verify disjoint room seeds and score all 512 targets. Export complete real-view
benchmarks and the fixed transform diagnostic for that final endpoint. If no
continuation runs, the preserved screen receives the single-run diagnostic
report, explicitly retaining any rejected status. Use that same fresh cohort
and mask seed for its completion evaluation; it does not retrospectively select
a model. Reserve final reporting inference even after a rejected screen.

The report generator presents one run/checkpoint with native metrics, learning
curves, annotated completion/matching/camera examples and explicit units. Private
arm comparisons remain in this internal study. Feature signal/error dB is not
RGB PSNR, and lower latent error does not establish sharp RGB reconstruction.
Independent seeds, broader pose data and protocol-matched public baselines
remain necessary for SOTA claims.

## Preflight numerical investigation

The disabled-objective replay passes all 64 updates: cross-view and monocular
losses agree exactly; other scalar differences remain within the registered
tolerances. Its warm median is 0.65576 s/update with 27,380 MiB process VRAM.

The enabled 64-update preflight completes with all 151 encoder gradient tensors,
unchanged teacher and anchor probes, and changed student/head probes. However,
its initial relative feature MSE is **2.00147315e-6**, exceeding the registered
1e-6 startup limit. This is approximately 0.14% RMS drift at identical weights.
The screen does not launch, and this preflight remains a failed numerical gate.

Before changing any threshold, run one native CUDA diagnostic (at most 180
command seconds, charged to the same residual allowance) on that exact first
minibatch. Compare each sparse/full/reference route using the student's batch
partitions versus the joint anchor batch, with and without intermediate output
capture. No optimizer step occurs. If a matched forward layout meets the
existing threshold, register that engineering correction and requalify before
the screen. Retain the original binary, sources and failed preflight. Quality
objectives, coefficient, samples, acceptance gates and update horizons stay fixed.

The partition audit does **not** support a batch-layout correction: matched and
joint full-view batches have identical errors, and all diagnostic routes are
below 2.54e-7. Intermediate output capture makes negligible differences. This
forward-only audit does not reproduce the complete trainer.

A second discarded 64-update preflight, with the unchanged numerical gate, adds
a native common-parent check of **every encoder floating parameter** before
any optimizer update, and logs the four route errors from the actual loss.
Neither the recipe nor the loss equation changes. Retain the original failed
preflight and both sealed implementations. This check is skipped on resume or
when the anchor differs from the student starting checkpoint.

The instrumented full trainer repeats the exact original mean discrepancy,
2.001473148993682e-6. All 158 floating parameter tensors (86,833,152 values)
match exactly. Its four relative route errors are 2.53849976e-7,
7.39620100e-6, 2.27226906e-7 and 1.28614658e-7. Only the full-target
route differs from the standalone audit. Both discarded preflights therefore
fail the unchanged original feature-error threshold. No training-screen
quality metric has yet been measured.

A further forward-only diagnostic (180-command-second cap) retains student
autodiff graphs and inserts the decoder prediction before full-target encoding,
matching that ordering in the trainer. This tests an execution-path difference;
it is not evidence that the CUDA kernel choice or fusion optimizer is the cause.

## Numerical qualification amendment before quality measurements

The decoder/autodiff diagnostic still produces the original standalone values.
It does not reproduce the trainer's full-target discrepancy; the specific kernel
or execution mechanism remains unresolved. Do not label batching, decoder order
or parameter drift as its demonstrated cause.

The two full-trainer preflights independently reproduce mean relative error
2.001473148993682e-6 (0.1415% RMS) before any optimizer update. The second compares
all 86,833,152 encoder parameter values exactly and finds zero differences.
This is an empirical numerical floor in this executable/workload, not a measured
learning change. The original strict 1e-6 forward-equivalence gate remains
**failed**; its records and sources remain immutable.

Before launching any screen, amend only numerical initialization qualification:
require exact equality of every floating encoder parameter, mean relative
feature MSE **below 3e-6** (approximately 1.5 times the repeated observed floor),
and **each of the four routes below 1e-5**. These are engineering bounds on
the observed starting discrepancy, not a claim of strict CUDA parity. The
source of that discrepancy remains an open issue. Require all previous
finite-loss, gradient, frozen-anchor and teacher checks, and repeat the
disabled-objective 64-update replay with the instrumented binary. The candidate
and both controls use that same binary. Do not increase these bounds again if
the matched screen fails them.

The original coefficient 4.0, raw objective, first minibatch, sample order,
update horizons, 1% completion-retention limit, 10% geometry-improvement limit,
nondecreasing precision requirement and external evaluation populations are
unchanged. No quality score from the new study has been used to set this
amendment. The single-run publication must disclose the numerical limitation.

## Completed matched screen

All three arms complete 2,048 updates and 32,768 ordered target exposures,
covering all 8,192 rooms and 24,576 room-view combinations. The native selector
checks common recipes, weights, samples, source identity, teacher, masks and
transform populations. It selects **preserved** before external inference.

| Same-budget arm | Validation latent MSE | Monocular MSE | Transform error (px) | Matches within 8 px |
| --- | ---: | ---: | ---: | ---: |
| Tail only | 0.17546027 | 0.20231869 | 7.4364 | 74.97% |
| Full, unrestricted | 0.18139246 | 0.20740306 | 3.5863 | 94.91% |
| Full, preservation 4.0 | **0.17545510** | 0.20275786 | **4.7757** | **89.05%** |

Preservation retains completion error at the tail-control level while improving
known-transform error by about 35.8% and eight-pixel accuracy by 14.1 percentage
points. It gives up some geometry improvement compared with unrestricted full
adaptation. All five quality gates pass. These are one-seed synthetic results,
not proof of real-view transfer, RGB sharpness or SOTA.

The selected screen endpoint is
`6d163190fd85c033693221265dea6dc429ca013d75c6bd240394542400341259`.
All 151 intended encoder tensors receive gradients on every full-stage update;
teacher and preservation-anchor probes remain exactly unchanged. Every initial
encoder floating parameter matches the fixed anchor. The startup forward error
repeats the preflight value and passes only the documented amended numerical
bounds; the original strict 1e-6 criterion remains failed.

Observed median update times are 0.652 / 0.922 / 1.081 seconds for tail / full /
preserved. Whole-command board energy is 17.51 / 24.67 / 28.75 joules per trained
target, including shared desktop load and initialization/evaluation. Peak
process GPU memory is 27,380 / 44,180 / 44,916 MiB. These are observations on
this shared workstation, not isolated energy or SM-occupancy measurements.

Receipts are `.data/pilot-12/screen-selection.json`, `screens-done.json`, the
three `*-screen-summary.json` files and their pinned native inputs. The
[conditional stronger-coefficient protocol](pilot-12-preservation-strength.md)
is not triggered because the primary candidate passes completion retention.
All original screen endpoints still receive complete development-benchmark
evaluation. Any continuation follows the predeclared time forecast and fixed
4,096-update endpoint, independently of those external scores.

## Canonical ETH3D execution amendment

The tail-screen optimized ETH3D exporter fails exact hard-index parity for the
spatial pair-conditioned readout on its sixth pair. Its failed command consumed
25.1865 seconds. Keep the partial five-pair export and failed receipt unchanged
and unscored. The canonical and optimized paths call different decoder entry
points; the specific CUDA numerical mechanism behind the discrepancy has not
been demonstrated. The optimized implementation has not qualified.

Before resuming ETH3D, register a canonical export for every screen endpoint and
the optional final continuation. It retains the original broad calculation
order, including teacher and layer controls. Hard and local matches are derived
from the same CPU score arrays, and their exact hard indices and mutual flags
are checked against the canonical hard readouts for every pair. The three
methods, 3 by 3 refinement, temperature, 3,365-pair population, learned weights
and accuracy gates are unchanged. This is a calculation-path repair; it is not
a relaxed parity threshold or a new model. CPU reference tests and strict
Clippy pass. The prior optimized failure is disclosed in the publication.

The first 64 pairs qualify throughput using p95 latency, a 15% margin and the
remaining population. Each canonical command has a 600-second command cap and
a 570-second internal cap. If the forecast does not fit, stop without scoring
partial results. No automatic retry or population reduction is allowed. The
global residual GPU allowance remains binding, including the failed command.
The fixed 4,096-update continuation still requires its original forecast plus
1,800 seconds reserved for final evaluation; it is skipped if they do not fit.

Reuse the completed, hash-checked tail HPatches score. All other HPatches and
TUM exports retain their sealed binaries and protocols. Synthetic screen
selection remains sealed and does not depend on real-view outcomes. Preserve
all earlier source archives, scripts, binaries and receipts. The revised
controller scripts only schedule these registered operations.

## Complete screen development evaluation

All three fixed endpoints finish 580 HPatches pairs, 3,365 canonical ETH3D
pairs and 186 TUM pairs. Every refined-readout matching transfer and local-precision
gate passes within each checkpoint. The coarse ETH3D readout's encoder-control
gate fails for full and preserved, because PCK3 decreases despite improved mean
pixel error. Both camera-motion transfer gates fail in every
arm; complete failure-inclusive per-sequence results are retained.

| Arm | HPatches viewpoint error (px) | Within 3 px | ETH3D error (px) | Within 3 px | TUM pose recall at 10 degrees |
| --- | ---: | ---: | ---: | ---: | ---: |
| Tail | 18.7520 | 20.85% | 29.7567 | 4.38% | 17.33% |
| Full | 16.5203 | 25.36% | 27.6103 | 5.51% | 21.12% |
| Preserved | 16.9403 | 23.66% | 27.9765 | 4.95% | 17.83% |

These are development outcomes, not additional selection criteria. HPatches
uses a 240 by 240 scoring frame and ETH3D original image pixels; their thresholds
are not interchangeable. TUM uses known intrinsics and a geometric solver, not
a trained camera head. Cross-arm point estimates do not measure training-seed
uncertainty. All three canonical ETH3D exports pass their complete-population
and shared-score checks; their earlier optimized failure remains unscored.

The registered optional continuation fits: 8,316.98 GPU-command seconds remain,
versus a 5,670-second training forecast and 1,800-second evaluation reserve.
Its immutable launch receipt is `.data/pilot-12/main-registration.json`. The
4,096-update horizon, seed 827, lower learning rates, original fixed anchor and
final-endpoint selection rule remain unchanged. External outcomes do not enter
the launch decision. The stronger-coefficient follow-up is not triggered.

## Completed fixed continuation

The main run completes all 4,096 registered updates, with 65,536 target exposures,
all 8,192 rooms and all 24,576 room-view combinations. The selected final model
is `b8e5d38c3a62eefd2b2d7ebc89e5b2d47b6ee91ac844d7e3d6eb257ae487aad8`.
Validation MSE decreases from 0.17528713 to 0.17434781 within this phase's fixed
mask protocol, a modest 0.54% change. This is not a matched continuation-arm
comparison. Final monocular MSE is 0.20145994; teacher-relative spatial variation
is 45.84%. Intermediate checkpoints are not used for endpoint selection.

All 151 intended encoder tensors receive gradients on every update. Teacher and
anchor parameter probes remain exactly unchanged. The phase consumes 4,664.68
GPU-command seconds (77.74 minutes). Median/p95 update times are 1.0793/1.1214
seconds, peak process VRAM is 44,916 MiB, and observed whole-command board energy
is 517.44 Wh or 28.42 J per target. Telemetry coverage is 99.98%. Shared desktop
load remains present; workspace CPU tests also overlap initialization and early
updates. These are shared-workstation measurements, not process-attributed
energy or an isolated hardware benchmark. All 149 workspace tests pass.

The first fresh-capture invocation fails argument parsing because its wrapper
uses `--helper` instead of the actual `--binary` flag. No rendering occurs.
Retain and charge the 1.05-second failed command. The v4 recovery only corrects
that flag, keeps the exact seed/config and final checkpoint, and uses a new
command/output identity. Its receipt is
`.data/pilot-12/capture-wrapper-recovery-sealed.json`. The corrected capture
finishes and passes native dataset verification before held-out assessment.

## Final checkpoint evaluation and review

The fixed main endpoint completes fresh-room assessment, all 580 HPatches pairs,
all 3,365 canonical ETH3D pairs and all 186 TUM input pairs. One predefined
low-baseline TUM pair is excluded from translation/pose, leaving 185 eligible
pairs; solver failures remain in the appropriate denominators.

| Final endpoint measurement | Result |
| --- | ---: |
| Fresh hidden latent MSE | 0.18663135 |
| Fresh reference benefit | 7.47% |
| Feature signal/error, not RGB PSNR | 7.329 dB |
| Teacher spatial variation retained | 42.52% |
| Learned RI co-visibility AUROC / AP | 0.7315 / 0.8993 |
| HPatches viewpoint error / within 3 px | 16.9561 px / 24.00% |
| ETH3D error / within 3 px | 28.3917 px / 5.15% |
| TUM pose recall within 10 degrees | 19.48% |
| TUM pose AUC at 10 degrees | 8.50% |
| TUM rotation / signed translation-direction error | 8.24 / 49.54 degrees |
| Known-transform error / within 8 px | 4.0568 px / 92.54% |

The fresh-room paired absolute latent-MSE reduction interval is
[0.0132542, 0.0168980], resampling 128 rooms 10,000 times. It measures sampling
uncertainty within this one checkpoint, not training-seed uncertainty. Geometric
labels identify 91,242 visible hidden patches among 117,727 known patches;
co-visibility AP must be interpreted with this positive prevalence.

The refined readout passes its registered matching transfer and local-precision
gates. The coarse ETH3D readout still fails its encoder-control transfer gate:
mean pixel error improves, but PCK3 drops by 0.0606 percentage points (95% scene
interval [-0.0977, -0.0197] percentage points). The same-image coarse control
passes. Both
calibrated-motion transfer gates now pass: pair conditioning's AUC@10 gain is
positive in every TUM sequence, and overall solver success does not regress.
Macro gains are 2.0644 and 2.3674 percentage points over the same-image and
encoder controls. These are controls of this final checkpoint, not private
training-arm comparisons. The longer phase does not uniformly improve every
external point estimate; for example, its ETH3D mean error is higher than the
screen endpoint's while within-three-pixel accuracy is higher. The registered
final-endpoint rule is retained.

Hidden RGB isolation is exact. Strict reference permutation still fails:
maximum feature difference 0.00347710, RMS difference 0.00017017, against the
unchanged 1e-5 strict threshold. Latent maps retain limited spatial variation.
They are neither RGB reconstructions nor evidence that the original RGB blur
has been solved. RGB, camera and depth heads remain untrained; TUM uses known
intrinsics and a CPU geometric solver. Absolute fine-match and pose accuracy
remain low. Reused development data, one seed and unmatched public protocols
prevent a SOTA claim.

The single-run artifact is `.data/publications/pilot12-main-reviewed/`, generated
from `configs/publish/pilot12-main.toml`. Native validation checks 114 hashed
files, 97 decoded images, all source metric/geometry contracts and six deterministic
completion examples. Software-browser checks cover 390, 768 and 1,440-pixel
layouts with no overflow or JavaScript errors. Manual review checks the paper,
completion maps, matching annotations, pose examples and limitations. Crowded
PDF training-axis labels are repaired in the Rust plot generator; the original
bundle is retained. Every metric, checkpoint, capability, efficiency record and
learning-curve data point remains identical across the layout repair.

All 149 workspace tests and the subsequent report-specific checks pass. Scoped
format checks pass for the changed owned crates; whole-workspace formatting
still flags an existing import-order difference in the unchanged audited
`burn_vjepa/tests/checkpoint.rs`. The encoder import audit verifies six files:
five verbatim and one recorded adaptation. No imported code is reformatted.

The stronger-weight experiment is explicitly skipped: its registered trigger
is false. Pilot 12 consumes 13,740.67 GPU-command seconds across 30 command legs,
including both failed commands. Combined with the immutable Pilot 11 spend,
total use is **40,217.52 / 43,200 seconds (11.172 / 12 hours)**; **49.71 minutes
remain unused**. Only study-owned monitors are stopped. Shared GPU process
observations, including unavailable counters, are summarized by native Rust.
No commit, push, deployment or crate publication occurs in this research study.

## Next controlled work, not launched

The preserved encoder supplies a qualified starting recipe, not an established
foundation-model ranking. A useful next quality study would compare the present
homography objective with an additional true-view correspondence objective from
cached renderer geometry. Visibility must separate occlusion, out-of-frame and
unknown depth, and any geometric labels must remain outside RGB inference.
This would be a **geometry-supervised auxiliary objective**, distinct from the
current image-transform self-supervision. Keep the completion-retention gate,
match compute and data exposure, and freeze selection before external scoring.

Camera qualification next needs an independent real-view cohort and independent
training seeds. Match public baseline protocols before any SOTA table; public
Gekko weights may remain an isolated evaluation baseline and must not become a
teacher or initialization dependency. A learned camera head would require its
own intrinsics/extrinsics targets, metrics and capability record, independently
of the current calibrated solver.

For numerical qualification, inspect actual matmul kernels and score margins
before changing precision or parity criteria. The locked `cubek-matmul` 0.2.0
source permits TF32 staging for accelerated f32 matmuls, a concrete hypothesis
rather than a demonstrated cause of these discrepancies. The evidence and
proposed diagnostic are in `.data/pilot-12/numerics-followup.md`.
For performance, caching invariant full-view anchor features is a candidate;
sparse-mask anchors still require online encoding. It needs a fixed-prefix
numerical replay, measured cache-build cost and a warmed throughput/energy
comparison before use. Neither follow-up has been implemented or claimed as an
optimization in this study.
