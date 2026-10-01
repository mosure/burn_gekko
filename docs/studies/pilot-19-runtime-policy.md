# Pilot 19: staged-upload lifetime window

Registered while the geometry candidate is running, before any new performance
command. This is a discarded runtime experiment, independent from checkpoint
selection and from the optional output-head study.

The Pilot 18 warm trace contains 503,052 host-to-device transfers and 7,926 CUDA
event synchronizations over 22 updates. The pinned CubeCL 0.10.0 CUDA backend
uses a double-buffered deferred-drop queue. Its default flush threshold is 64
allocations or 64 MiB, whichever arrives first; a flush waits for the previous
batch's fence before releasing its host buffers. These observations suggest a
testable synchronization cost, not a proven bottleneck.

Sources: [published queue policy](https://github.com/tracel-ai/cubecl/blob/v0.10.0/crates/cubecl-runtime/src/memory_management/drop_queue/policy.rs)
and [lifetime implementation](https://github.com/tracel-ai/cubecl/blob/v0.10.0/crates/cubecl-runtime/src/memory_management/drop_queue/queue.rs).
Local registry source hashes are pinned with the diagnostic.

## Fixed comparison

Build two executables from burn_gekko commit
`3dfc6c0124151343a1b71ad54375be6b6d1b5613`, using the same CUDA features and
compiler. Baseline uses unmodified published dependencies. Candidate changes
only CUDA stream construction to use **512 allocations**, retaining the **64 MiB
byte threshold**, two-batch lifetime protection and every device synchronization
required by readback or update completion. No fence, synchronization, check,
gradient, tensor operation or numerical precision is removed.

The candidate dependency copy, patch, separate lockfiles and binaries remain in
`.data/pilot-19/runtime-policy/`; the registry cache and main workspace's runtime
dependencies remain unchanged. A detached checkout isolates both builds. Build
failures and shell orchestration errors are retained but consume no GPU time.

After Pilot 19 training and required quality evaluations finish, run **A-B-B-A**,
with **128 updates per command**. Both use the original Pilot 18 geometry
preflight recipe, extended only to 128 updates with a 900-second trainer cap.
Keep the 4,096-update decay horizon, 128-step warmup, seed 853, all 8,192 rooms,
same parent, batch 16, geometry weight 0.1 and all other losses. Disable periodic
evaluation/checkpoints within the prefix; retain final evaluation/saving and its
time/energy. Every produced checkpoint is discarded from model selection.

Each command is capped at **950 seconds**, with all four charged to
`.data/pilot-18/budget.json`. Before starting, require the entire 3,800-second
cap plus a **3,600-second remaining reserve**. Never overlap another experiment
GPU command. Preserve the shared desktop load and process monitoring. Do not
retry a failed leg or selectively retain a favorable repetition.
Stop before the next leg if native trajectory replay fails.

## Decision and limits

Require both baseline runs and both candidate runs to reproduce the complete
128-update loss/sample/stage/gradient trajectory of A1 at absolute tolerance
1e-6 plus relative tolerance 1e-5. Also verify A1's first 64 updates against the
original immutable preflight. Require unchanged teacher and preservation-anchor
probes and finite active geometry supervision.

Native training summaries report warm medians and p95 values (first ten updates
excluded), command energy per target, memory and telemetry coverage. A useful
performance screen requires **each** B run's median update to be at least 5%
faster than the faster A run, its p95 no worse than either A, and each B's board
joules per target no greater than the smaller A observation. All four command
telemetry coverages must exceed 99%, and B peak RSS/VRAM may rise by at most 5%
against the larger A peak. These are shared-workstation observations, not
process-attributed power or a training-quality improvement.

Only if every numerical and performance criterion passes, permit one additional
32-update candidate NVTX capture capped at **900 seconds**, requiring at least
910 seconds plus the 3,600-second reserve. Use the completed Pilot 18 warm-trace
protocol and all 32 replay checks; integer bytes/nanoseconds are mandatory. That
trace may diagnose mechanism but cannot supply the unprofiled speedup estimate.
No runtime change is adopted into published crates merely because this screen
passes; any production integration needs explicit dependency provenance and
native/browser compatibility qualification. No other threshold sweep is run.
