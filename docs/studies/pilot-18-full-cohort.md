# Pilot 18: full-cohort geometry continuation

Status: both throughput preflights passed; matched full runs active. This executes the scientific
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

A separate [warm-update dispatch replay](pilot-18-warm-dispatch.md) is registered
for after the quality evaluation. It cannot change model selection. Its optional
profiling build leaves the frozen full-run trainer untouched.

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
