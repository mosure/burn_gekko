# Pilot 11: bounded batch-capacity and throughput probe

Registered after the completed dispatch trace and during the fixed full recovery,
before any batch-32 training. The trace contains many small operations, and the
batch-16 runs leave substantial process VRAM headroom. Test whether doubling the
batch increases observed target throughput without changing model architecture,
objectives, precision or the sealed training executable.

This is a **performance diagnostic, not a new accuracy arm**. Wait until the
registered quality decision, its required evaluations and manual report review
are complete. Use that decision's fixed checkpoint and encoder stage; if full
recovery is rejected, use the already completed tail main endpoint. Throughput
and external accuracy do not choose the parent. Discard every probe checkpoint.
Do not change any accepted training recipe or claim batch-size accuracy parity.

Reserve at most **1,800 additional GPU-command seconds within the same 43,200
seconds**, never outside it. Require at least 1,810 seconds remaining before
launch. Run serially:

| Leg | Batch | Updates | Target exposures | Command cap |
| --- | ---: | ---: | ---: | ---: |
| A before | 16 | 256 | 4,096 | 600 s |
| B larger | 32 | 128 | 4,096 | 600 s |
| A after | 16 | 256 | 4,096 | 600 s |

All arms start from the same own checkpoint with both AdamW optimizers reset.
Keep all 8,192 rooms, 256px inputs, two references, 90% masking, objectives,
learning rates, 200-update warmup and 12,000-update cosine horizon. Use seed 797
and the selected fixed encoder stage. Each internal wall limit is 560 seconds.
Change only batch size, update count, wall limit and output path. Initial/final
validation and checkpoint writing remain in command energy/time accounting.
On an allocation failure, nonfinite result or incomplete leg, retain the failed
receipt and stop the probe; do not retry or attempt a larger batch.

The sampler addresses target identities as `step * batch_size + offset`, so the
4,096 ordered target identities must match across legs. Verify the serialized
sample stream hashes, complete exposure counts, fixed stages, finite gradients,
unchanged teacher and updated encoder/head probes. Masks, transform grouping
and optimizer update counts differ across batch sizes: losses and trained
weights are not numerically equivalent across A and B. The two A legs must pass
the native full-prefix scalar/sample replay check at 1e-6 absolute plus 1e-5
relative tolerance. A failed replay prevents a batching qualification.

Use native training reports and `gekko-eval training` for warm target throughput,
median/p95 update time, process VRAM/RSS, command duration and board energy per
target. The trainer's warm target throughput uses its upper-median update time
after the first ten updates; the summary's median uses linear interpolation.
Label that distinction when quoting either. Board energy includes shared
desktop activity and initialization/evaluation; it is not process energy or a
steady-state power measurement.

Retain both bracketing A measurements. If their warm target throughputs differ
by more than 10% (larger divided by smaller minus one), label the throughput
comparison inconclusive under changing shared load. Otherwise a useful observed
throughput result requires B to exceed **both** A values by at least 20%, with
peak process VRAM at most **81,920 MiB (80 GiB)**. These are performance screening
criteria only. No inference about convergence, learning-rate scaling, public
benchmark accuracy or SOTA follows from a pass. A later quality study must
separately validate any larger training batch.

Keep this comparison in the internal study. The primary single-run project
pages/PDFs retain their actual batch-16 training telemetry and accuracy evidence.
Native Rust performs all numerical metric computation and replay verification;
temporary orchestration only schedules commands and checks artifact identities.

## Completed result: capacity passes, throughput threshold fails

All three legs complete at 2026-09-30 22:28 UTC from the retained tail main
checkpoint `bdd1bd204afc5584ce7277af40d8bea80f6987c6611937e79b7fb13074789637`.
Each covers the same 4,096 ordered room/view targets, spanning 3,478 rooms.
The ordered target-stream SHA256 is identical across all three partitions:
`3525e62ad1a9dab9c8e9e284a517960cfacf84baf4814fba354e07377e02b24c`.
Every update has the required 28 encoder gradient tensors, unchanged teacher
and first-block QKV probes, and updated last-block/head probes.

| Native measurement | A before, batch 16 | B larger, batch 32 | A after, batch 16 |
| --- | ---: | ---: | ---: |
| Updates | 256 | 128 | 256 |
| Warm targets/s, trainer upper median | 24.2349 | 28.1925 | 24.5928 |
| Warm median update, summary interpolation (s) | 0.66014 | 1.13496 | 0.65059 |
| Warm p95 update (s) | 0.69224 | 1.17936 | 0.68312 |
| Peak process VRAM (MiB) | 27,380 | 53,812 | 27,380 |
| Peak process RSS (MiB) | 20,317.63 | 20,385.08 | 20,489.40 |
| Command duration (s) | 310.03 | 287.46 | 305.04 |
| Observed board energy (Wh) | 23.3342 | 21.6024 | 23.6440 |
| Observed board joules/target | 20.5086 | 18.9865 | 20.7809 |

The bracketing A rates differ by **1.48%**, below the registered 10% drift
limit. Native replay passes all 256 A updates at the prescribed tolerances;
cross-view and monocular losses match exactly. The largest component difference
is 9.54e-7 in same-image transform NLL. This qualifies the A repeat, not A/B
accuracy equivalence.

Batch 32 fits the memory ceiling but is only **14.64--16.33% faster** than the
two controls, below the required 20% gain against both. Its peak process VRAM
is **1.965 times** the batch-16 value. Retain batch 16; do not promote a larger
batch or any discarded probe weights. Lower observed whole-command board
energy accompanies this small throughput gain, but includes preparation,
evaluation and shared desktop activity. It is not a measurement of process
energy savings or steady-state efficiency.

The probe uses **902.53 of its 1,800 reserved GPU-command seconds**, entirely
inside the unchanged study allowance. Native summaries, complete A replay,
per-command telemetry, sample streams and the completion receipt are under
`.data/pilot-11/batch-probe/`. All three generated TOML experiment recipes and
native evaluation configs are retained. This negative performance screen does
not change the selected quality checkpoint, its batch-16 training telemetry,
or either reviewed single-run publication bundle.
