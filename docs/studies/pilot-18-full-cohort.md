# Pilot 18: full-cohort geometry continuation

Status: both full runs and required evaluations completed; **candidate rejected**
by the registered adjacent-detail gate. This executes the scientific
proposal in [Pilot 16](pilot-16-continuation.md), under the user's new ceiling of
**12 additional GPU-command hours**. The unused 96.281 seconds of the previous
allowance remain separate. Capture, training and GPU evaluation command time,
including failed commands, count toward the new 43,200-second ledger. CPU builds,
scoring and report generation do not.

The original preregistration remains immutable at
`.data/pilot-18/registered-protocol.md`. Native forecast and selection receipts,
source pins and the cumulative ledger live alongside it.

## Dataset and execution checks

The full-pixel native geometry audit passed on all 8,448 rooms (including the
reserved test rooms for coordinate integrity only; no model scores were computed
on test data). Across 1,660,944,380 valid source pixels, the largest self-reprojection
error was **0.008215 input pixels** and the largest self-depth error was
**0.000180 m**. This supports camera/world/depth alignment in the frozen dataset;
it does not establish learned correspondence or camera accuracy. The CPU-only
audit took 217.32 seconds. Its receipt is
`.data/pilot-18/full-pixel-geometry-audit.json`.

Each full run's first 64 updates reproduced its preflight prefix, including losses,
sample order, masks, encoder gradients and the geometry auxiliary where enabled.
The comparison uses absolute tolerance 1e-6 plus relative tolerance 1e-5. The
immutable prefix snapshots and native replay receipts live in `.data/pilot-18/`.

The control completed all 4,096 updates and covered every one of the 8,192 training
rooms and 24,576 room/view targets: 65,536 target exposures, or eight per room on
average. Every update carried all 151 encoder gradient tensors; the teacher and
preservation-anchor parameter probes stayed unchanged. Its training-time validation
MSE moved from 0.176511 to 0.174985, a 0.865% reduction. Common standalone assessment
is required for model selection; a reduction in feature MSE alone does not show
that RGB reconstruction is sharp.

The control took 71.79 GPU-command minutes, with a warm median update of 0.960 s
and p95 of 1.052 s. Peak process GPU memory was 44,916 MiB and host RSS was
20,853 MiB. Shared-board telemetry covered 99.98% of command time: median device
utilization 89%, observed mean power 427.2 W, energy 511.0 Wh and 28.07 J per target
exposure. These are whole-command board observations including the authorized
desktop load, startup and validation; they are not process-attributed power or SM
occupancy. See `.data/pilot-18/control-summary.json`.

The separate [warm-update dispatch replay](pilot-18-warm-dispatch.md) completed
after quality evaluation and passed numerical replay. It cannot change model
selection. Its optional profiling build leaves the frozen full-run trainer
untouched.

## Final quality decision

All source, initialization, sampled-target, fixed-horizon and gradient checks
passed. Both runs covered 8,192 rooms, 24,576 unique room/view targets and 65,536
target exposures, with 151 encoder gradient tensors on every update. The common
assessment covers 64 rooms / 192 targets; matching and camera scoring cover 32
rooms, with all eight solver seeds retained.

| Synthetic development measure | Published parent | Matched control | Geometry candidate |
| --- | ---: | ---: | ---: |
| Hidden latent MSE, lower better | 0.176511 | 0.174985 | 0.176544 |
| Centered structure MSE, lower better | 0.170802 | 0.169393 | 0.170803 |
| Centered spatial correlation | 0.673794 | 0.677119 | 0.673772 |
| Neighbor-difference correlation | 0.433037 | 0.438091 | 0.433042 |
| Neighbor-difference power / teacher | 19.26% | 19.85% | 19.31% |
| Actual-view mean pixel error | 12.181 | 15.766 | 9.294 |
| Matches within 8 input pixels | 58.72% | 50.89% | 67.64% |
| Mean calibrated pose AUC at 10 degrees | 21.04% | 17.16% | 24.84% |

The candidate passes seven of eight quality checks. The failed check is small
but explicit: neighbor-difference correlation falls **0.005048509** below the
control, exceeding the registered 0.005 tolerance by **0.000048509**. Its paired
95% room interval is [-0.005939, -0.004174]. The criterion is not rounded or
waived after observing the result. The selected checkpoint remains Pilot 13;
the control is not promoted automatically either.

The candidate's own [17-page PDF](../../.data/publications/pilot18-geometry/paper.pdf)
and [interactive report](../../.data/publications/pilot18-geometry/index.html)
contain six deterministic annotated samples and complete camera readouts. Native
validation passed for 99 hashed files, 82 images and 27 local links. Browser checks
passed at 390, 768 and 1,440 pixels, including every sample selection. This private
diagnostic bundle does not replace the accepted public project page or demo.

Relative to the published parent, viewpoint error falls 2.886 pixels (paired
95% room interval 2.312–3.556), while joint camera angular error falls 4.572
degrees (1.121–8.938). Pose AUC improves in seven of eight solver seeds against
the parent and every seed against the control. Candidate pose AUC spans
21.60–29.43%. These are synthetic development improvements. Completion detail
is essentially unchanged from the parent, and the control's modest detail gain
is lost; RGB blur and real-view transfer are not resolved by this result.

The geometry run takes **4,890.283 GPU-command seconds / 81.50 minutes**. Its
warm median/p95 update is 1.100 / 1.253 seconds, peak process GPU memory is
49,110 MiB, and peak host RSS is 21,134 MiB. Observed shared-board energy is
565.92 Wh, or 31.09 J per target, with 99.98% telemetry coverage. Board mean
power is 416.7 W and median device utilization is 88%; these retain the shared
desktop scope above. The geometry auxiliary adds computation; no energy saving
is claimed.

The native selection receipt is `.data/pilot-18/selection.json`. Its source
closure, per-room uncertainty, complete solver panel and failed gate remain
preserved. The next controlled study halves geometry-loss weight, testing
whether some matching benefit can coexist with the control's detail improvement.
It uses a new preregistration and the same cumulative GPU allowance.

## Preflight outcome

Both 64-update preflights completed with all 151 encoder gradient tensors,
finite update records and unchanged teacher/preservation-anchor probes. Command
time was 274.01 seconds for control and 285.29 seconds for geometry. The full
host cohort uses about 21 GB RSS; control process GPU memory peaks at 44,916 MiB.
Warm p95 updates are 1.0993 and 1.2675 seconds respectively. Native forecasting
projects 6,414 and 7,274 seconds, including registered margins: **3.80 hours**
for both full runs, plus the separate two-hour evaluation reserve. The budget
after preflights is 42,640.697 seconds. Startup-heavy preflight board averages
are not sustained-training efficiency estimates.

## Fixed experiment

Both arms initialize from Pilot 13 geometry checkpoint
`0639afa2e19d0e7ac5c9c1cc8caadc75d0dcf10f105ee2870607856b7a1ed5a7`.
They use the same 8,192 cached training rooms, batch 16, seed 853, 4,096 updates,
128-update warmup and 4,096-update cosine horizon. Two reference views accompany
a 90%-masked target. The entire encoder is trainable from the first update.
The fixed MIT V-JEPA 2.1 teacher and Pilot 11 preservation anchor stay frozen.
No noncommercial teacher or checkpoint enters training.

The control retains latent completion, reconstruction improvement, synthetic-warp
equivariance and encoder preservation. The candidate additionally uses actual-view
renderer geometry NLL, weight 0.1, temperature 0.07. All other scientific settings
match the prepared Pilot 16 configs. Geometry supervises training only; the model's
inputs remain RGB. This compares equal updates, not equal compute.

Data are the immutable historical Zeroverse 0.25.0 / dataset-adapter 0.8.0 cache
`e54ba670d6087c445d1d04c7c3ba1742d0e610d08533690921ac655f1f66f05b`:
three 256×256 views per room. The newer live-demo renderer does not change this
dataset's provenance. The CPU geometry cache covers all 8,192 train and 128
validation rooms, with no test labels. The first 64 validation rooms score
completion and the first 32 score matching and pose. These reused rooms are
development data, not independent qualification data.

## Execution and budget

Use the prepared CUDA trainer built from commit
`d61b84be4a50e6629edc55e24d01241181db09f4`, binary SHA-256
`b411843bf402e02fc04bbf68d4a2b04dc8e0556f00234cb55990f18fb534843e`.
Pin the binary, original lockfile, configs, protocol and input identities before
launch. CPU evaluator engineering may continue independently; do not rebuild or
replace the trainer between arms.

First run each arm for 64 updates on the full resident cohort. Keep its 128-update
warmup and 4,096-update decay horizon, and disable periodic checkpoints. These are
throughput and numerical preflights, not candidate checkpoints. Each has a
900-second trainer cap and a 950-second command cap. Do not tune the scientific
protocol using their validation outcomes.

Forecast each complete arm from warm update p95 (excluding the first ten updates)
times 4,096 times 1.25, plus measured preparation/finalization overhead and a
600-second allowance for periodic evaluation/checkpointing. Require each forecast
to fit its wall cap and both arms plus a **7,200-second evaluation reserve** to fit
the remaining ledger. If this fails, do not silently reduce the fixed horizon.

The only change from Pilot 16's proposed execution limits is a **10,000-second
trainer cap per full arm**, with a 10,100-second command cap, allowing the fixed
4,096-update horizon to finish despite shared-workstation overhead. These limits
are registered before any new outcome. Final endpoints alone enter selection;
periodic checkpoints are recovery artifacts, not a best-validation sweep.

Retain one-second board telemetry and separate process memory, plus ten-second
process activity observations. Shared desktop load is authorized and remains
running. Board watts/utilization are shared measurements, not trainer-only power
or SM occupancy. Report energy per target including startup and evaluation.

## Conjunctive retention gates

1. Both complete all 4,096 updates with identical sampled targets, masks, source,
   initialization and recipe except for the geometry auxiliary. Every update has
   151 encoder gradient tensors. Losses and gradient norms are finite; teacher and
   preservation anchor probes remain unchanged; geometry supervision is active.
2. Candidate hidden-token MSE is at most 1% above the better of parent and control,
   using common standalone assessment data/masks. References beat the matched
   monocular branch.
3. Candidate actual-view AEPE is at least 5% lower than both parent and control;
   PCK at eight input pixels is nondecreasing against both.
4. Candidate centered completion MSE is at most 1% above control. Centered spatial
   correlation and adjacent-difference correlation are each at least control
   minus 0.005. Greater feature variance alone cannot satisfy this gate.
5. Mean synthetic pose AUC@10 across all eight registered solver seeds is at least
   both parent and control. Report the seed range, failed fits and signed
   translation error. Use seeds 871–878, normalized eight-point RANSAC with
   2,048 maximum / 64 minimum trials, confidence 0.999, at least 12 inliers,
   a 1.5-input-pixel Sampson threshold normalized by mean focal length, and a
   0.01-m minimum baseline. Renderer visibility never filters solver inputs.

All gates are necessary. A failure retains the current published checkpoint.
Paired room-level uncertainty accompanies the comparisons. Preserve complete
source closures, failures and every solver seed; neither reused TUM/HPatches/
ETH3D results nor Pilot 17's five-point diagnostic may choose the arm.

## Reporting and subsequent work

Report completion in feature units, actual RGB reconstruction as PSNR in dB,
camera angles in degrees and focal error in percent. Do not relabel a geometric
camera probe as learned camera-head prediction. This study alone cannot establish
sharp reconstruction, real-world generalization or SotA.

Only after a synthetic pass may a separate preregistered study expand the RGB and
camera-head training cohort beyond Head Stability 15's 32 rooms. A replacement
project page, paper or demo must describe one selected foundation run and its
audited heads. Failed studies remain documented without replacing accepted
weights. Do not automatically consume unused budget.
