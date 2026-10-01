# Pilot 11: bounded dispatch diagnostic

Registered before profiling. The user's shared-workstation efficiency request
calls for direct traces rather than interpreting device utilization as kernel
occupancy. After the fixed main run and its primary/secondary evaluations finish,
allow one additional command of at most **600 seconds**, charged to the same
43,200-second ledger. Skip if that allowance is unavailable. This is neither a
new accuracy candidate nor an extension of the main training horizon.

Use installed NVIDIA Nsight Systems 2026.1.3 to trace CUDA APIs, kernels and
memory operations for a separate 64-update replay of the selected main run's
prefix. Start from the same screen parent, seed 773, reset optimizers, batch 16,
learning rates, warmup and 12,000-update cosine horizon. Change only the stop
step, command wall limit and output path. Discard its trained weights. Do not
profile other processes or enable system-wide sampling. Capture the whole short
command, including preparation and final evaluation, with CPU sampling/context
switch collection disabled. Profiling overhead makes its timing unsuitable as
the main run's throughput measurement.

Finish the primary native paper/page build and automated browser checks before
starting this trace, so our own report generation does not compete with the
profiling replay. The later recovery arm starts after profiling closes out.
It also waits for manual primary-report review and any CPU-only layout repair,
so our own report rendering does not overlap the recovery launch.

Require the native update-prefix replay check to preserve all 64 sample/stage
identities and finite component losses within 1e-6 absolute plus 1e-5 relative
tolerance. Retain failures. The main checkpoint, schedule and accuracy results
cannot change in response to this diagnostic.

Export the vendor's CUDA GPU/API traces with nanosecond units. Native Rust
analysis must merge overlapping GPU intervals instead of adding them as though
they were serial. Report launch counts, durations, small kernels (at most 10 us),
API synchronization and gaps in the traced process's GPU work. Gaps are not
device-idle or SM-occupancy measurements: desktop work may use the GPU, CPU
preparation/JIT is included, and this trace does not isolate steady-state training.
Keep raw profiler artifacts and hashes. Record these performance diagnostics in
the internal study; the primary page/PDF retains its own unprofiled training
telemetry and singular run/checkpoint.

## Completed diagnostic

The 64-update replay completes in **261.09 GPU-command seconds**, including
profiler collection/finalization. Native comparison matches every sample, stage,
gradient count and scalar within the registered tolerances. Cross-view and
monocular losses agree exactly; the largest recorded component difference is
7.15e-7 in same-image transform NLL. Its weights remain discarded.

Nsight's GPU/API JSON is analyzed in Rust with integer nanosecond timestamps.
The first-to-last GPU-event window is 119.787 seconds. Overlap-aware totals are:

| Trace observation | Result |
| --- | ---: |
| GPU operation interval union | 44.500 s / 37.15% of window |
| Kernel interval union | 31.924 s |
| Intervals without this process's traced GPU work | 75.286 s |
| Largest uncovered interval | 8.380 s |
| Kernel launches | 3,937,628 |
| Kernels at most 10 microseconds | 3,477,938 / 88.33% |
| Host-to-device copies | 2,606,899 |
| CUDA API interval union | 47.726 s |
| Synchronization API interval union | 26.701 s |

Per-name API duration sums include 6.983 seconds in kernel launches and
26.701 seconds in event synchronization. These intervals can overlap productive
GPU work; they cannot be added as exclusive elapsed-time components. Copy counts
alone do not establish their byte volume or cause. Large preparation/JIT gaps
and the first update's 19.67-second warmup are present, as is final evaluation.
The trace confirms many small operations but **does not isolate a steady-state
dispatch bottleneck, whole-device idle fraction or SM occupancy**. Normal main
training's unprofiled warm median remains 0.65688 seconds per update.

The appropriate next performance measurement is explicit training/forward/
backward/optimizer ranges after warmup, followed by a same-prefix numerical
replay for any proposed caching or dispatch change. No optimization or energy
saving is claimed from this trace. Raw `.nsys-rep`, SQLite, JSON exports, native
summary and replay receipts remain in `.data/pilot-11/`; the main checkpoint is
unchanged. CPU trace export and analysis are outside the GPU-command ledger.
