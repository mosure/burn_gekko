# Pilot 18: warm-update dispatch diagnostic

Registered while the fixed full-cohort geometry arm is still running, before
collecting any new profile. This is a discarded performance replay and cannot
select a model or change the registered quality experiment.

After both full runs and their required quality evaluations finish, permit one
additional GPU command capped at **900 seconds**, charged to the same new
43,200-second allowance. Require at least 910 seconds remaining. Retain any
failure and stop this diagnostic without retrying. Other GPU jobs must be idle;
the authorized desktop processes remain running and separately monitored.

Replay the first **32 updates** of the geometry preflight: same Pilot 13 parent,
all 8,192 rooms, batch 16, seed 853, full encoder adaptation, actual-view NLL
weight 0.1, preservation weight 4, 128-update warmup and 4,096-update cosine
horizon. Keep its recipe unchanged except for the stop step and a 600-second
trainer wall limit. There is no periodic evaluation within these 32 updates and
periodic checkpoints are disabled. Final validation and checkpoint writing remain
in command accounting, but outside the collected warm trace. Discard the weights.

Build `burn_gekko_train` with optional `cuda,profiling` features. The annotations
add no tensor operations or synchronization. Pin the committed source, lockfile,
executable and recipe before launch. Use Nsight Systems 2026.1.3 with CUDA and
NVTX tracing, CPU sampling/context-switch collection disabled, capture range
`warm-training@burn_gekko`, and capture-end behavior `stop`. That behavior lets
the application complete normally after tracing ends. The range starts after ten
updates and closes before final evaluation: **22 warm updates** are required.

The nested host ranges are `data`, `frozen-targets`,
`forward-and-loss-readback`, `backward-and-clip`, and `optimizer-and-sync`.
Each update ends after the trainer's existing device synchronization. Native Rust
analysis verifies the complete parent/child range panel, process/thread identity,
integer nanosecond bounds and nonoverlapping update/phase ranges. It intersects
GPU/API events with those ranges and merges overlaps before measuring duration.

Phase measurements describe temporal overlap with host scopes, not kernel
ownership: asynchronous work can cross phase boundaries. Counts may overlap
between phases and durations of nested ranges must not be added. An empty phase
is a valid zero-GPU observation, not a missing value. Uncovered intervals are not
whole-device idle time or SM occupancy. Profiled durations are not unprofiled
throughput or evidence of an energy saving.

Before interpreting the trace, require the native update-prefix replay check to
match all 32 sample/stage/gradient identities and all component losses (including
geometry and preservation) at absolute tolerance 1e-6 plus relative tolerance
1e-5. Compare against the immutable original geometry preflight. Do not change
the full-arm CUDA executable or any learned model in response to this diagnostic.
Any resulting optimization needs its own numerical and unprofiled performance
qualification.

Raw traces, range exports, numerical replay and native dispatch reports belong in
`.data/pilot-18/dispatch/`. The public single-run model page retains its own
unprofiled training measurements.
