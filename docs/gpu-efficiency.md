# GPU efficiency: measured transfer bottleneck and repair

Measured 2026-09-29 on the RTX PRO 6000 Blackwell workstation. GPU activity
alone had hidden substantial transfer and dispatch overhead. The change below
is qualified for numerical preservation and a bounded training workload.

## Diagnosis

Nsight Systems 2026.1.3 profiled a replay of full-encoder updates 3501-3532,
restoring the original model, both optimizers and the gate from update 3500.
The sealed v13 binary and original 4000-step learning-rate schedule were used.
All 32 total losses match the unprofiled run; gradient norms differ by at most
4.65e-8. Profiler timings are retained as diagnostic timings, not performance
claims: median time in the warmed window is 1.0602 seconds versus 1.0367 seconds
in the original execution.

In the 24 complete updates 3505-3528, bounded by their distinctive RGB uploads:

| Measurement | Result |
| --- | ---: |
| Elapsed window | 27.393 s |
| GPU kernels | 523,272 |
| Union of kernel intervals | 11.051 s / 40.34% |
| RGB host-to-device transfers | 7.668 s / 0.3195 s per update |
| All traced kernels and copies | 72.70% of elapsed time |
| Gaps without a traced kernel/copy | 7.479 s |

These are intervals for the profiled process, not SM occupancy. Gaps may include
CPU work, dispatch, synchronization and competing untraced desktop activity.
The device's 97% median activity counter therefore cannot establish efficient
useful computation. The trace also contains many small metadata transfers and
kernel launches; this repair does not claim to eliminate all dispatch overhead.

The pinned Burn 0.21 `Autodiff::attention` unconditionally uses its differentiable
attention fallback. The V-JEPA module calls Burn's attention API, but that alone
does not establish use of a fused attention kernel during training. The decoder
also explicitly computes attention scores because its auxiliary objective and
matching readouts consume them. A subsequent attention optimization must preserve
those scores and gradients and pass a separate native throughput/energy check;
no such optimization is claimed here.

The pinned CubeCL 0.10 CUDA upload implementation uses a pitched 2D transfer for
rank >= 2. Passing `[B,H,W,3]` RGB directly creates 12-byte rows. The new
`encoder::upload_rgb` uploads `[B*H*W*3]` contiguously, then reshapes and permutes
to NCHW. It does not resize, reorder pixels incorrectly, quantize or alter color
normalization. Training host batches, single-view encoding and both external
benchmark exporters use the helper.

## Qualification

The upload microbenchmark alternates legacy NHWC, flat NHWC and CPU-packed NCHW
at 256 pixels, with three warmups and 32 measured repetitions per layout and
batch size. Normalized GPU RGB arrays are bitwise identical.

| Batch | Legacy upload + normalize | Flat upload + normalize | CPU planar layout |
| --- | ---: | ---: | ---: |
| 1 | 3.771 ms | **0.957 ms** | 1.201 ms |
| 16 | 61.119 ms | **7.186 ms** | 8.982 ms |

Two subsequent 64-update runs use the same own checkpoint, fresh optimizers,
64 training rooms, two validation rooms, batch 16, frozen encoder, examples,
masks and schedule. They use sealed v13 and v16 binaries respectively. The
new optional hierarchy objective is disabled in both. The first eight updates
are excluded from the steady update-time comparison.

| Matched training measurement | Before | After |
| --- | ---: | ---: |
| Median warmed update | 0.7911 s | **0.4860 s** |
| Complete command | 152.54 s | **128.05 s** |
| Observed board energy | 7.311 Wh | **6.665 Wh** |
| Gross board energy per trained target | 25.70 J | **23.43 J** |
| Validation latent MSE | 0.194953735 | 0.194953735 |

Every recorded component loss is equal across all 64 updates. Maximum
gradient-norm difference is 1.73e-8. This passes the predeclared 1e-6 absolute
plus 1e-5 relative training tolerance and 1e-6 validation tolerance. Teacher and
frozen-encoder parameters remain unchanged.

The observed result is 1.63x steady update throughput and 8.84% less gross board
energy for this short complete workload. Startup/validation dilute the training
gain. Power measurements are device-wide, include desktop activity, and are
sampled near 1 Hz. They are integrated with short endpoint holds and explicit
gap exclusion; both comparisons have complete supported coverage. These data
are not whole-system electricity measurements or a multi-seed energy study.
Full-encoder training continues to be monitored separately.

## Reproduction

Configs and generated receipts are under `configs/pilot07-*` and
`.data/pilot-07/fusion-audit/`. GPU commands always use the existing Pilot07
budget ledger; the failed first profiling invocation is retained and charged.

```sh
# Use fresh output paths when repeating GPU commands.
.data/analysis-venv/bin/python tools/study/run_study.py \
  --plan configs/archive/pilot-07/pilot07-rgb-upload-bench-plan.toml \
  --output .data/my-upload-check --budget-ledger .data/pilot-07/budget.json

# These analysis configs name immutable outputs; copy and adjust before repeating.
.data/analysis-venv/bin/python tools/legacy/cuda_trace_review.py \
  --config configs/archive/pilot-07/pilot07-cuda-trace-review.toml
.data/analysis-venv/bin/python tools/legacy/compare_training_execution.py \
  --config configs/archive/pilot-07/pilot07-upload-training-compare.toml
.data/analysis-venv/bin/python tools/study/gpu_efficiency.py \
  --config configs/archive/pilot-07/pilot07-upload-training-energy.toml
```

The GPU plan also contains its native output filename; changing only the runner
directory does not make an existing native output safe to overwrite. Retain
the original artifacts and update both paths in a copied TOML plan.

## Shared-device replay after the repair

A later repeat slowed even with the previous sealed v16 binary. The first 16
updates use identical samples, component losses and schedule; warmed median
time rises from 0.4806 to 0.9813 seconds. Maximum gradient-norm drift is 1.55e-8.
This rejects attributing that slowdown solely to the new spatial-input code.

A concurrent read-only process monitor records remote desktop and the desktop
shell active on the GPU. Over its 140-sample window, their respective recorded
mean SM-activity counters are 24.0% and 22.8%; the counters are not additive
occupancy measurements. A separate idle-training sample also observed substantial
remote-desktop activity. These observations identify competition on a shared
device, but do not isolate how much each process caused the slowdown. No other
application was stopped or modified. Subsequent experiments record whole-board
energy and flag shared load; they do not compare gross energy across these
conditions as if the GPU were dedicated.

Receipts: `contention-replay-review.json`, `contention-replay-pmon.log`, and
`contention-replay-run/ledger.json` under `.data/pilot-07/fusion-audit/`.

## Final execution qualification

The exact v18 continuation completes all 5,000 additional updates in 4,185.34
command seconds. Its frozen/last-two/full-encoder stages execute 250/1,000/3,750
updates at warmed median 0.490/0.549/0.761 seconds per update. Complete-command
board energy is 460.12 Wh, or 20.71 gross joules per trained target, with full
supported telemetry coverage. Changing desktop conditions prevent interpreting
this as a controlled comparison with the slower earlier matched-input arms.

A final 32-update exact replay from step 5000 has identical total loss to the
unprofiled trajectory and maximum gradient-norm difference 4.82e-8. The middle
24 updates contain 524,976 kernel launches in a 19.318-second window. Kernel
union covers 60.37%; kernels plus copies cover 69.25%. RGB transfer costs
0.01954 seconds per update; unattributed gaps cost 0.24754 seconds per update.
The profile records 5,895 event synchronizations. Synchronization API duration
overlaps device execution, so it cannot be added to kernel time as extra waste.
These measurements support further work on launches and synchronization; they
do not establish that all dispatch overhead is solved or quantify SM occupancy.
This model/objective differs from the original trace, so their kernel coverage
fractions are not a controlled before/after speedup.

The long process log covers local 19:29:42 through 22:11:55 on September 29.
The shell has 7,496 numeric activity reports averaging 21.75%, out of 9,472
observed rows. Remote-desktop rows in this later window report `-`, which is
unavailable, not zero. Earlier numeric remote-desktop measurements remain
separate. The monitor only reads process counters and is stopped after the
final profile; no desktop process or device power setting is changed.

Final receipts: `route-upload-trace-review.json`,
`route-balanced-continuation-energy.json`, `final-phase-energy.json`, and
`shared-gpu-activity-summary.json` under `.data/pilot-07/fusion-audit/`.
The full study charges 37,240.67 seconds (10.345 hours) to the original
12-hour ceiling. Accuracy selection closes before independent evaluation;
remaining allowance is not used to tune against those outcomes.

## Selective cache verification

The spatial-descriptor preflight spent its entire 60-second command ceiling
auditing the 8,192-room cache before inference. It produced no model result, and
the failed command remains charged to the shared ledger. A retry on the existing
130-room development cache completed the neutral-head probe in 27.2 seconds and
the separate 16-update preflight in 57.6 seconds. Dataset size changed only for
these diagnostics.

The current Rust loader separates manifest validation from shard verification.
Training and assessment verify membership, filename, checksum, decoded shape and
camera metadata for every selected room as it is read, exactly once. The full
`open_dataset` audit still reads all shards when an exhaustive audit is requested.
Tests reject corrupted selected shards, forged membership and corrupted unused
shards during a full audit; an unused corrupt shard does not block a valid
selected read. Exact optimizer/sample-order resume also passes with the loader.

The 3,000-update spatial-descriptor run and its GPU evaluators retain the sealed
v23 source and binaries from before this loader change. Their timings therefore
do not measure a cache-startup improvement. No GPU kernel, objective or power
setting changed, and no throughput or energy speedup is claimed for this repair
without a future matched measurement.
