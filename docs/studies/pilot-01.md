# Pilot 01: published Zeroverse upgrade and bounded GPU training

Completed on 2026-09-27 America/Chicago (2026-09-28 UTC). The pipeline trains,
reuses captured data, batches samples, caches frozen features, resumes optimizer
state, and evaluates held-out rooms. Reconstruction losses decrease. **Useful
co-visibility and a consistent benefit from reference views are not established.**

All generated artifacts remain in `../.data/`. This report records a diagnostic
study, not a reproduced Gekko result or a qualified V-JEPA checkpoint conversion.

Configuration syntax was subsequently migrated to TOML. Examples below use the
current `.toml` filenames; the recorded pilot's JSON configs and command ledgers
retain their original contents. New study plans and resolved configs use TOML,
while historical snapshots can still be read for evaluation.

## Package and dataset upgrade

The registry was checked before running the study. The latest unyanked releases
were [bevy_zeroverse 0.22.0](https://crates.io/crates/bevy_zeroverse/0.22.0) and
[bevy_zeroverse_burn 0.5.0](https://crates.io/crates/bevy_zeroverse_burn/0.5.0),
published at 02:36 UTC on September 28. Both are pinned exactly in
`tools/zeroverse_capture/Cargo.toml` and its independent lockfile. Registry
responses, timestamps, and checksums are in `.data/pilot-01/*-registry.json`.
The main training workspace stays on Burn 0.21.0.

The new release adds the connected multi-view camera rig. `capture-pilot.toml`
enables it explicitly. It also exports **unclamped affine AABB positions** under
the `position=2` capture-engine contract. Values outside [0,1] are legitimate
visible surfaces outside the reconstruction crop. The reader requires the new
metadata before allowing them and applies the inverse AABB transform. Legacy
captures retain their original identity and stricter range checks.

The capture rendered all 16 rooms before the old reader rejected this contract.
After fixing the adapter, `recover-capture` verified the original binary/config
fingerprint, acquired a lock, checked every shard and the geometry, and published
the existing capture. No second render was needed. Recovery is explicit; normal
cache opens never promote partial data.

| Dataset setting | Recorded value |
| --- | --- |
| Cache ID | `c2565d5eea331f4f8533d408d172239ed6a984b5efb5b59fc0e715c5faa99851` |
| Independent rooms | 12 train / 2 validation / 2 test |
| Seeds | 2026092800–2026092815, disjoint room splits |
| Views | Three static cameras, one instant, 128×128 |
| Storage | Approximately 13 MiB; zstd Safetensors with F32 RGB/depth/position and metadata |
| Rendering | Procedural indoor, density 0.35, portable lighting, no humans, one worker |
| Geometry audit | 786,432 source pixels; max self-reprojection 0.002695 px; max depth error 0.00001335 m |
| All-pair audit | 96 directed view pairs |

Rendered visible fractions across directed pairs range from 18.9–94.7% on train,
33.0–86.2% on validation, and 48.5–92.1% on test. The camera sampler's proxy
overlap setting is not a guaranteed pixel overlap floor. Geometry is used only
for audit/evaluation; training inputs and objectives remain RGB-only.

The new cache was reopened without running the generator. The old four-room
0.21.0/0.4.0 cache also verifies without relabeling its provenance. Capture data is
persisted on disk; small training/validation RGB sets and optional frozen full-view
features are resident during each run. There is no persistent feature cache yet.

[Capture examples](../../.data/pilot-01/capture-samples.png) ·
[Geometry audit](../../.data/pilot-01/geometry-audit.json)

## Protocol and implementation

The protocol and command plan were written before training in
`.data/pilot-01/protocol.json`. The ceiling was 30 minutes of cumulative GPU-job
wall time, 16 generated rooms, and 220 optimizer steps across all runs. The study
completed **220 steps in 382.9 seconds of job wall time**, including initial and
final evaluations and an evaluation-only extension for reconstruction metrics.
No hyperparameters or checkpoints were selected from test results. No longer
training or larger generation followed.

Common settings: one RTX PRO 6000 Blackwell Workstation Edition (97,887 MiB),
driver 610.43.02, Rust 1.98.0, optimized `pilot` build, Burn CUDA/fusion, F32
execution, frozen local V-JEPA 2.1 Base encoder, 75% target masking, two reference
views, decoder width 64/depth 2/four heads, per-patch normalized RGB targets,
AdamW with weight decay 0.05 and true global gradient clipping at 1.0. Encoder
weights have verified package hashes; independent official-checkpoint image
parity remains open.

| Run | Steps | Batch | Training rooms | Full features | Learning rate |
| --- | ---: | ---: | ---: | --- | ---: |
| Online timing control | 8 | 1 | 12 | Recomputed | 0.0001 |
| Batched timing control | 8 | 4 | 12 | Recomputed | 0.0001 |
| Cached timing control | 8 | 4 | 12 | Resident | 0.0001 |
| Fixed-example fit | 64 | 1 | 1 | Resident | 0.001 |
| Small training set | 128 | 4 | 12 | Resident | 0.0003 |
| CUDA resume check | 4 additional, from step 4 to 8 | 4 | 12 | Resident | 0.0001 |

The fixed-example check uses one target/reference set and one fixed mask. Its
predeclared pass criterion was finite training and at least 30% loss reduction.
The 12-room run cycles all 36 room/target pairs, visiting 512 examples (about
14.2 passes), with masks determined by seed and step. A batch shares its mask.
Fixed train/validation panels each contain four examples for this run; they are
monitoring panels, not whole-split estimates. Final evaluation covers all six
target views in each held-out split with one fixed evaluation mask.

New implementation paths are `src/pilot.rs`, `src/batch.rs`, and
`configs/pilot-*.toml`. They separate preparation from step timings, save step-zero
and periodic checkpoints, and log all room/target identities. Masked targets are
always re-encoded before attention. Only unmasked frozen features can be reused.
The sampler now visits every target in every room without room/view modulo
aliasing. Global gradient norm aggregates on the GPU and makes one scalar
readback, replacing per-parameter readbacks.

`tools/study/run_study.py` executes an explicit command list without shell expansion,
records logs and 1 Hz GPU telemetry, and stops on errors or cumulative/per-command
wall limits. `tools/legacy/analyze_pilot.py` regenerates JSON and standalone SVG/PNG
figures. Analysis dependency versions are recorded alongside the results.

## Efficiency observations

Warm timing excludes preparation and the first two optimizer steps. Each timing
screen therefore has **only six measured steps**. Phase screens synchronize
between phases; these are approximate diagnostic estimates, not saturated GPU
benchmarks. Runs occurred sequentially with different cold compilation/cache
histories. Millisecond step timers are rounded down to integers.

| Screen | Warm median / p90 step | Examples/s | Preparation | Total run | Peak process VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Batch 1, online | 47 / 61 ms | 20.5 | 68.7 s | 147.0 s | 4,962 MiB |
| Batch 4, online | 54 / 61 ms | 73.4 | 19.5 s | 29.3 s | 5,042 MiB |
| Batch 4, cached | 24 / 29 ms | 166.7 | 15.2 s | 22.7 s | 4,976 MiB |

Batching improved this screen's throughput about **3.6×**. Adding frozen
full-view caching improved batch-four throughput another **2.3×**, or about
**8.1×** versus batch-one online encoding. CPU tests verify cached/online feature
agreement; the CUDA controls' final validation losses differ by about 0.00060,
consistent with their different batched floating-point execution paths.

Mean encoder phase time fell from **32.90 ms to 6.24 ms** for batch four.
In the cached screen the other means were 4.18 ms forward/loss, 12.24 ms
backward/global clipping, and 1.86 ms optimizer. Backward and gradient handling
are now the largest measured phase. There is no operator-level profile separating
autodiff, norm reduction, kernel launch, and host synchronization within it.

The 128-step run measured about 163.6 warm examples/s, 22 ms median and 30 ms
p90 steps, 15.2 s preparation and 25.3 s total time. Its process peaked at
5,074 MiB VRAM. The bounded resident RGB/normalized arrays use 15.75 MiB and
full-view tokens 7.875 MiB; CUDA allocations/runtime dominate observed VRAM.
Preparation and first-step work dominate these short runs, so warm throughput
must not be substituted for end-to-end throughput.

The long-enough-to-sample portion of the small-set run has just four 1 Hz device
samples, averaging 39.5% utilization and 122.9 W. Desktop activity contributes;
the eight-step screens have **no reliable steady telemetry samples**. These
measurements establish neither occupancy nor peak hardware efficiency. Longer
repeated timing windows, batch sizes up to eight, and a real operator profile
are the next efficiency checks before changing hardware.

## Convergence and held-out evidence

The fixed-example loss decreased **4.4534 → 1.0735 (75.9%)**, passing its bounded
criterion. Cross reconstruction decreased 1.3100 → 0.5325, MAE 1.2908 → 0.5251,
and RI loss 1.8526 → 0.0160. It has not perfectly memorized the image. Validation
reconstruction on its single monitored example became slightly worse late in
training, while its RI term improved; total loss alone can obscure overfitting.

The 12-room fixed-panel total losses decreased **5.3808 → 1.7326 (67.8%)** on
train and **5.6463 → 1.7210 (69.5%)** on validation. All recorded losses and
gradients were finite and weights changed. Most initial total loss was RI error;
its decline is not evidence of geometric co-visibility.
The small-set raw gradient norm ranged from 0.149 to 23.50 (median 0.522);
global clipping applied on 47 of 128 steps.

Whole-held-out results use the same preset initial/final checkpoints and mask:

| Split / step | Total loss | Cross MSE | MAE MSE | RI loss | Co-visibility AP | AUROC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation / 0 | 5.7908 | 1.2908 | 1.2630 | 3.2370 | 0.8220 | 0.5002 |
| Validation / 128 | 1.6652 | 0.8145 | 0.8123 | 0.0383 | 0.8171 | 0.5002 |
| Test / 0 | 4.3175 | 1.3060 | 1.2815 | 1.7300 | 0.8743 | 0.4953 |
| Test / 128 | 1.1379 | 0.5586 | 0.5630 | 0.0164 | 0.8718 | 0.4891 |

Visibility is directional **to any supplied reference**, not pair-specific.
Unknown pixels are excluded: validation ranks 93,827 pixels, test 97,640. The
positive fractions, and approximate random-ranking AP baselines, are 0.8216 and
0.8753. High raw AP is therefore not evidence of skill. Pixel observations are
strongly correlated, and there are only two rooms per split; no significance or
generalization claim is justified.

At step 128 cross-view MSE is 0.27% worse than MAE on validation and 0.79% better
on test. The branches have no consistent reference benefit. RI ranking remains
near chance. This is a successful pipeline/optimization pilot and an unestablished
fusion/co-visibility result. Scaling the study or hardware is not justified yet.

## Checkpoint and verification evidence

The CUDA cached control was resumed from step 4 with both decoder and AdamW
state. Its next four samples, all four logged loss terms, and final fixed probes
matched the uninterrupted run exactly. Global norm differed by at most
`7.52e-7`, consistent with reduction ordering. This checks the actual CUDA flow,
not bit-identical checkpoint files or every possible backend/configuration.

All **39 CPU tests** passed. Strict CUDA workspace Clippy, strict isolated capture
Clippy, formatting, and six imported encoder-file hashes passed. Added regression
coverage includes batched cache/online agreement without masked-token reuse,
complete room/target cycles, pilot resume, legacy cache identities, unclamped
position round trips, capture lock/identity recovery, and corruption rejection.
The two seeded CPU training tests share a mutex because NdArray's RNG is global.
Existing model/loss/visibility tests continue to pass. CI was not run remotely.

## Next bounded study

1. Qualify the local encoder against official native-image outputs. The current
   package has strict tensor/hash loading, but not this independent parity proof.
2. Establish reference utility with matched MAE, true-reference, shuffled-reference,
   and unrelated-reference controls on the same masks and examples. Measure
   per-room reconstruction deltas and their relation to geometric visibility.
3. Compare pairwise and three-view objectives with a slightly longer fixed budget,
   still using validation for decisions. Audit textures, lighting, and depth
   boundaries; the capture sample sheet includes dark scenes and very small images.
4. Repeat warm timing for sufficient duration to measure utilization and variance;
   profile backward/clipping/host synchronization and test batch eight. Retain
   the frozen full-feature cache for static, unaugmented inputs.
5. Advance to more rooms or decoder capacity only after useful reference signal
   appears. AMP, sparse reference selection, real-data transfer, multiple seeds,
   and paper-level claims remain outside this pilot.

## Artifacts and reproduction

- [Machine-readable summary](../../.data/pilot-01/summary.json)
- [Summary figure, SVG](../../.data/pilot-01/pilot-summary.svg) and
  [PNG](../../.data/pilot-01/pilot-summary.png)
- [Protocol](../../.data/pilot-01/protocol.json), [initial command ledger](../../.data/pilot-01/execution/ledger.json),
  [evaluation extension](../../.data/pilot-01/evaluation/ledger.json)
- `.data/runs/pilot-01-*/`: configs, step logs, probes, model/optimizer checkpoints,
  and evaluation JSON with per-target metrics.
- `.data/pilot-01/execution/*-gpu.jsonl`: raw process/device telemetry.

Follow the build/capture/train commands in the [README](../../README.md), using a
fresh run name. Run each `configs/pilot-*.toml` for the corresponding control;
`steps` is the desired total when resuming. `eval-preflight --step 0` evaluates
the initial checkpoint and `--step 128 --split validation|test` evaluates the
small-set endpoint. The recorded command plan contains every exact invocation.
It requires new run names and a fresh ledger directory when repeated.

To regenerate the report artifacts without training:

```sh
.data/analysis-venv/bin/python tools/legacy/analyze_pilot.py --study .data/pilot-01
```
