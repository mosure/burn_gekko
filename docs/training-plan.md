# Training stages and single-workstation resource plan

Status: original proposed experiment protocol. The user selected **one workstation
GPU, then scale only if justified**. Subsequent implementation and measured runs
are documented in [pilot 05](studies/pilot-05.md) and the [end-to-end guide](e2e-pipeline.md).
The settings below are research proposals; they do not describe every current
pilot setting. The current project excludes NC pretrained weights and supports
both entirely random training and progressive adaptation of MIT V-JEPA weights.

## 1. Hardware and initial operating policy

A read-only inventory reports one NVIDIA RTX PRO 6000 Blackwell Workstation
Edition with 97,887 MiB total memory and driver 610.43.02. Total memory is not
available memory; backend support, precision stability and throughput still need
qualification. Record actual free/peak memory and all concurrent processes in
benchmark metadata.

Use CUDA as the initial proposed training backend, CPU for small reference
fixtures, and WGPU for a separately qualified inference lane. Start with float32
correctness; enable BF16 only after forward/backward comparisons. If a Burn
operation lacks reliable BF16/autodiff support, keep it in float32 or use a
measured supported path and record the change.

Run capture, conversion and timed training sequentially on the workstation.
Use one renderer worker and small chunks initially. Keep preprocessing/readers
bounded and prefetch enough to hide disk latency without allowing unbounded RAM
residency. Training must not depend on a live generator keeping up with the GPU.

## 2. Initial training specification

Proposed settings are starting points for bounded pilots, not validated optima.

| Setting | Initial choice |
| --- | --- |
| Encoder | V-JEPA 2.1 Base, native image modality, pretrained weights frozen |
| Resolution | 256 square diagnostic; 384 square primary, with controlled rectangular extension |
| Decoder | 512 width / 8 blocks / 16 heads primary; 256 / 4 / 8 diagnostic |
| Views | Target + one reference, then target + 1–3 references |
| Mask | 90% hidden target patches, shared by both reconstructions |
| Reference retention | Full initially; sparse budgets are a later independent factor |
| Loss | `released_linear_eps` initial reference; all three coefficients 1 |
| Optimizer | AdamW, decoder LR `2e-4`, betas `(0.9,0.95)`, weight decay `0.05` |
| Weight-decay exclusions | Bias, normalization scale/bias, learned mask/role tokens |
| Schedule | Linear warmup for first 5% of updates, cosine decay to `1e-6` |
| Gradient clipping | Global norm 1.0, log pre-clip norm and clipped-update fraction |
| Effective batch | Initially 64 target/reference groups per optimizer update |
| Microbatch | Start at 4 groups, increase after memory profiling; use accumulation |
| Precision | BF16 compute only after qualification; float32 loss/statistics and optimizer state where supported |
| Checkpoints | Every 1,000 updates and before a controlled stop; atomic writes |
| Validation | Fixed small RGB validation protocol every 1,000 updates; richer geometry evaluations at declared milestones |
| Confirmatory seeds | Three independent training seeds for each selected comparison |

Do not treat a change in view count as a change in effective group batch size.
Log images, selected tokens, decoder queries, pair edges and groups separately.
Normalize accumulated loss so unequal microbatches have the intended sample or
pixel weighting. The primary convention is equal weight per target example,
with masked-valid-pixel means inside each example.

The original paper's training scale is context, not a workstation recipe. The
initial experiment is a smaller adaptation with pretrained per-view encoders.
Model/data/compute differences remain explicit in every comparison to released
models or published numbers.

## 3. Staged training protocol

### T0: correctness and tiny overfit

Use analytic images and the separate 64-scene diagnostic subset. First exercise
a tiny random encoder/decoder on CPU for gradient checks, then the real frozen
encoder plus diagnostic decoder on GPU. Verify masked reconstruction support,
RI gradients, reference-invariant MAE, native-image loading, and exact sampler
resume. A tiny model does not establish the real encoder's quality.

Require decreasing masked reconstruction loss on a deliberately learnable
fixture, finite gradients, and invariance tests before considering a metric gain.
For copied target/reference images, test reconstruction dependence explicitly;
do not count these degenerate pairs as normal training data.

Recommended ceiling: 200 optimizer updates per diagnostic and 30 minutes per
GPU diagnostic run. Nonconvergence or a timeout is a failed/incomplete diagnostic,
not a reason to launch a longer full-scale run.

### T1: objective screen

Use the pilot split, one fixed seed, and the same encoder/decoder initialization,
pair list, augmentation sequence and target masks across:

1. Cross-view reconstruction only.
2. Cross-view plus monocular reconstruction.
3. Both reconstructions plus RI prediction.

Run each for 2,000 updates, with a proposed two-GPU-hour wall ceiling. Keep the
encoder frozen. Verify that reference swapping degrades informative examples
and that low-overlap/textureless results are visible in reports. This screen
selects a valid implementation and broad operating regime; it is not a final
multi-seed result.

All compared rows must complete the same intended exposure budget for a
fixed-step comparison. If the wall ceiling truncates one, compare only a common
completed prefix as diagnostic evidence or replan the entire matched comparison.
Never rank a partial row as a full-budget result.

### T2: two-view primary study

Move to the initial-study scene split. Keep S0 RGB-only and S1 geometry-curated
studies separate. Start with S0; use S1 as a controlled sampling ablation if the
pair-distribution diagnostic warrants it.

Proposed primary budget: 20,000 optimizer updates, effective batch 64, at most
12 GPU-hours per run after throughput confirms feasibility. This is 1.28 million
target-group exposures per completed run, not 1.28 million unique scenes. For
the three core objective rows and three seeds, the training ceiling is 108
GPU-hours. Use the final scheduled checkpoint for the confirmatory primary
comparison; preserve intermediate checkpoints for learning-curve analysis.

The screen may choose among a small predeclared mask-rate grid `{0.75,0.9}` and
decoder LR grid `{1e-4,2e-4}` if needed. Limit this to at most four short trials
for the affected model family, share the selected schedule with controls, and
charge all trials to the development budget. Additional searches are separately
registered follow-ups.

### T3: variable-view fusion

Initialize from the accepted pair model and introduce reference sets. Sample one
target per group, one to three references, and one auxiliary pair edge. Preserve
a pair fraction so low-view behavior remains represented. Use independent
reference dropout; keep target reconstruction masks shared across branches.

For a causal comparison, give the pairwise control the same additional data and
updates. Compare joint sets with uniform pair-feature pooling and score-weighted
pair pooling. Conduct both fixed total reference-token and equal per-view
resolution analyses. The latter buys more information and compute as views grow.

Proposed extension: one chosen joint-set model and its matched continued-pair
control, three seeds each, up to 10,000 additional updates and 6 GPU-hours per
run. The six continuations therefore have a 36-GPU-hour ceiling. Uniform and
RI-weighted pooling share the continued-pair checkpoint. Training costs include
the shared pair initialization and extra continuation. A later from-scratch set
run tests whether the curriculum itself mattered.

### T4: sparse reference training

First evaluate the dense model at reference keep ratios
`{1,.75,.5,.25,.125}` to expose sensitivity. Then train one selected mixed-budget
model with the same sparse reference subsets in cross-view and RI paths.
Use uniform/stratified spatial masks before learned policies.

Compare sparse encoding to dense encoding followed by gather. Keep the same
visible input patches when checking numerical implementations. Train/test at
multiple budgets and evaluate at held-out budgets; do not select a different
best checkpoint for each test budget without disclosure.

Proposed extension: one selected sparse model and a dense continued-training
control, three seeds each, up to 10,000 additional updates and 6 GPU-hours per
run. The six continuations have a 36-GPU-hour ceiling. If no end-to-end savings
survive packing, full-target decoding and selection overhead, stop the efficiency
branch and report the result. Learned routing is optional follow-up work.

### T5: encoder adaptation, only if justified

Use frozen probes and diagnostics to decide whether the encoder bottlenecks
geometry. Suggested sequence: train fusion only; then encoder norms; then the
last two blocks; finally broader unfreezing if the prior stage helps.

Start encoder learning rate at one-tenth the decoder rate, with a separate
optimizer group and explicit schedule. Measure pretrained-feature drift on an
independent image set. A detached original-encoder consistency term is an
optional regularizer with its own ablation and cost accounting.

Custom sparse-patchify/attention kernels require verified backward support in
this stage. A frozen-kernel speed result cannot justify unfreezing through an
untested operation. Full-backbone training increases memory and changes the
scientific claim; keep frozen and adapted results in separate tables.

### T6: downstream probes and domain transfer

Freeze the selected fusion backbone and train equally sized correspondence,
pose, depth/pointmap probes under equal labeled-data and update budgets. Then,
optionally, run end-to-end supervised fine-tuning as a separate track.

Evaluate the synthetic-trained backbone on real scenes before any real-domain
adaptation. If adaptation is needed, distinguish RGB-only real adaptation,
labeled probe training, and full supervised fine-tuning. Report the synthetic-
only result even when adapted performance improves.

## 4. Loss scheduling and stable pseudo-targets

The primary objective activates all terms from the start after correctness
qualification. Do not add an undisclosed RI warmup merely because early scores
look noisy. A bounded warmup is a registered ablation: for example, ramp
`lambda_ri` from zero to one over the first 5% of updates, with identical
reconstruction schedules in controls.

Generate RI targets from the same model state and the same target mask as the
current update. Detach both error tensors. Log their distribution and the
fraction below the chosen coefficient epsilon. An EMA reconstruction teacher,
smoothed targets, clipping or different branch capacities changes the estimator
and requires its own named configuration.

Photometric augmentation is initially shared in policy and modest across views.
Independent strong color transforms can make a geometrically visible reference
unhelpful for RGB reconstruction. Test that later as a robustness factor; keep
the target observation and its supervised reconstruction target consistent.

Do not add GT co-visibility, reprojection, flow, pointmap or semantic losses to a
run still labeled S0/S1. Latent-feature prediction, cross-view contrastive terms,
cycle constraints and supervised auxiliaries are secondary ablations after the
pixel objective, each with explicit target provenance.

## 5. Single-GPU memory and performance engineering

For frozen encoding, use non-autodiff/no-gradient encoder execution and attach
its outputs as constants to the trainable projection/decoder. Reuse reference
and masked-target features within a step when their full identity matches.
Caching is governed by [the leakage rules](architecture.md#5-information-leakage-and-cache-rules).

Benchmark increasing microbatch sizes at fixed total token count. Prefer a
registered token budget and gradient accumulation to OOM-driven sample dropping.
Padding-aware buckets reduce wasted work for sparse or rectangular inputs.

Start with a straightforward combined-loss backward. If activation memory is a
limitation, consider recomputation/checkpointing or sequential branch backward
only after gradient-equivalence tests. Do not assume the selected Burn backend
supports every checkpointing or distributed feature.

Benchmark 20 warmup iterations and at least 100 measured iterations for a pilot,
with GPU synchronization around timings and compilation reported separately.
Retain median/p90 and repeated-run variation. Measure encoder, projection,
decoder, loss/backward, optimizer, input decode and transfer; also report the
unpartitioned full-step time. Timer scopes must not overlap in summed breakdowns.

Budget equations:

```text
run GPU-hours = optimizer_updates * measured_seconds_per_update / 3600
study GPU-hours = sum(all completed, failed, screen and confirmation run hours)
capture hours = scene_count / measured_scenes_per_hour
storage bytes = measured_bytes_per_scene * scenes + indexes + checkpoints
```

Include all accumulation microsteps in an optimizer-update timing. Encoder FLOP
estimates and selected-token counts explain costs; they do not replace timings.
Account for weight conversion, dataset generation and probe fitting separately
from pretraining in the paper's total-cost table.

## 6. Proposed resource envelope and scaling trigger

Before starting actual experiments, replace these proposed ceilings with a
registered budget informed by the throughput pilot. Stop at the first reached
limit: updates, wall time, disk budget, or an explicit failure condition.

| Work | Candidate GPU-hour ceiling |
| --- | ---: |
| Encoder/loss correctness and extra development trials | 16 |
| Three objective pilot runs | 6 |
| Two-view three-objective, three-seed confirmation | 108 |
| Joint-set plus continued-pair control, three seeds each | 36 |
| Sparse plus dense continuation control, three seeds each | 36 |
| Capture, evaluation, and selected small probes | 32 |
| Unallocated recovery margin | 6 |
| **Full staged initial envelope** | **240** |

This is at most ten device-days if every stage and ceiling is used, not a predicted
runtime or a commitment to spend it. Most optional ablations and encoder
fine-tuning do not fit automatically inside that envelope. Prioritize H1–H3;
defer the rest rather than silently increasing the budget. Stop earlier when a
stage fails its scientific/correctness gate. Data volume has a separate measured
disk ceiling to avoid confusing cheap model steps with expensive rendering.

Scale beyond one GPU only after a useful model is established and the measured
cost of the next justified study exceeds workstation limits. First improve data
loading or batching if it is the bottleneck. Multi-GPU work then requires its
own gradient-accumulation equivalence, sampler sharding, equal global-batch,
checkpoint and failure-recovery qualification; no distributed capability is
assumed from the current plan.

## 7. Monitoring, stop rules, and status labels

Log losses separately, gradient norms per module, effective LR, parameter update
norms, score quantiles, MAE/cross-view error distributions, retained token counts,
view counts, sample IDs, skipped/invalid counts, throughput and peak memory.
Record how much reconstruction improves over MAE on held-out informative regions.

Stop immediately on nonfinite loss/gradients, required checkpoint mismatch,
data leakage, invalid masking, wrong geometry convention, or silent sample loss.
Treat persistent collapse of reference dependence as a failed screen after its
fixed diagnostic budget. Do not endlessly retry seeds or extend training until
one run wins.

Statuses: `planned`, `running`, `completed`, `completed_negative`,
`failed_correctness`, `failed_numerics`, `incomplete_budget`, `interrupted`.
An incomplete run can contribute clearly labeled diagnostic curves, never a
completed-study headline. Resume preserves status history and consumed budget.

S0's confirmatory checkpoint is the fixed final step. If geometry-labeled
validation chooses architecture/hyperparameters or checkpoints, disclose that
as label-assisted model selection even though the gradient objective is
self-supervised. Calibrated probabilities and supervised probes also have
explicit fitting data; they are not zero-shot outputs.
