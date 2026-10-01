# Pilot 11: conditional longer full-encoder recovery

Registered after the completed 2,048-update adaptation screen, before the main
run's first scheduled validation and before its fresh/external evaluations.
This is a **post-screen follow-up**, not part of the original confirmatory
screen. That screen rejected full adaptation: geometry improved substantially,
but completion MSE exceeded its 1% allowance. Do not rewrite that verdict.

The follow-up tests a specific alternative explanation: 2,048 updates may be
too short for the fusion trunk to accommodate a changing early encoder. Continue
the full-screen endpoint for the **same 12,000 updates** as the selected tail
main phase. The full endpoint is
`1e11fbcab5fd31458328854c79f3f1c39e8726a07c686dd3b0d238c32e767dc3`.
Reset both AdamW optimizers. Match seed 773, all 8,192 rooms, batch 16, architecture,
masking, objectives, trunk LR 4e-5, encoder LR 8e-7, 200 warmup steps, fixed 12,000
cosine horizon, validation every 1,000, and checkpoints every 2,000. Full stage 2
remains fixed. The only intended differences across the full two-phase paths are
trainable encoder stage and the corresponding matched screen endpoint.

Run after the tail main's complete evaluation and bounded profiling diagnostic,
never concurrently with another GPU study. Charge the **same** 43,200-second
allowance. Before launch, require remaining time to exceed 12,000 times the full
screen p95 times 1.10 plus 600 preparation/evaluation seconds and 2,700 seconds
reserved for endpoint evaluation. Keep the same optimizer horizon; skip the
arm rather than shorten it if there is insufficient headroom. A watchdog may
use the remaining allowance less the evaluation reserve. Nonfinite/time-cut
runs cannot be accepted as completed arms.

Compare only the two fixed main endpoints on the shared synthetic validation
population and 32-room known-transform diagnostic (seed 1000773). Require:

1. Full completion MSE no more than 1% above tail and positive reference benefit.
2. Lower full pair-conditioned local mean pixel error and no reduction in PCK8.
3. Complete matched sample order, validation masks, transform/query populations,
   verified stages, unchanged teacher and correctly linked screen parents.

Reuse the native selection rule with an explicit 12,000-update contract and
pinned parent summaries; do not alter thresholds in response to these outcomes.
HPatches, ETH3D, TUM and the tail's new held-out rooms cannot select this arm.

If accepted and budget permits, seal the full final endpoint before generating
a **new** 128-room/four-view test cohort, seed 2610071000, otherwise identical to
the main fresh capture. Export all 512 target views with mask seed 773; run the
same fixed HPatches, ETH3D and 256px TUM development protocols. A 512px TUM
diagnostic may reuse the already qualified exporter within that 2,700-second
reservation. If rejected, retain the negative internal result and skip new
external exports/capture. The existing tail report stays unchanged.

An accepted full endpoint receives its own single-run page/PDF with controls
from that checkpoint. Comparisons between these private training paths belong
only in this internal study record. One seed and unmatched public benchmark
protocols still cannot establish SOTA, regardless of the outcome.

## Launch record

The full recovery launches at **2026-09-30 18:57:38 UTC**, after main evaluation,
profiling and manual review of the corrected primary PDF/page. The layout repair
leaves every numerical result unchanged. Available study time is 29,264.85
seconds; the preregistered p95-based training forecast is 16,337 seconds, plus
the evaluation reserve. The command watchdog is 21,719 seconds and the fixed
12,000-update horizon remains unchanged. This phase uses the original sealed
trainer, not a newly compiled training executable.

The comparison endpoint is the completed tail main checkpoint
`bdd1bd204afc5584ce7277af40d8bea80f6987c6611937e79b7fb13074789637`.
Its validation MSE is 0.17265890, and its matched known-transform local error is
7.42653px with 73.64% PCK8. The fixed 1% completion-retention rule still applies;
the observed real-image camera failures do not modify that rule or this recipe.

## Completed result: reject full recovery

The full continuation completes all **12,000 updates** and 192,000 target
exposures at 2026-09-30 22:11 UTC. Final checkpoint:
`19183b77b9edffdc93f289b573a53c1d78c7750188776ddb19f39c10d982138f`.
Every update has 151 encoder gradient tensors. First/last encoder QKV and the
prediction-head probes change; the teacher QKV probe remains exactly unchanged.
All 8,192 rooms and 24,576 room/view combinations are covered. The native
selector verifies both matched screen parents, complete sample order, fixed
12,000-update horizons and identical validation/transform populations.

| Fixed endpoint metric | Tail main | Full recovery |
| --- | ---: | ---: |
| Validation hidden latent MSE | 0.17265890 | 0.18127593 |
| References-disabled MSE | 0.19795573 | 0.19930535 |
| Known-transform local mean error | 7.42653 px | 2.27205 px |
| Known-transform within 8px | 73.64% | 98.99% |

Full adaptation substantially improves the synthetic transform task but leaves
completion MSE **4.99% higher**, outside the registered 1% retention allowance.
Reference use still helps. The combined gate fails, so retain the tail main
checkpoint `bdd1bd204afc5584ce7277af40d8bea80f6987c6611937e79b7fb13074789637`.
This longer negative result is additional evidence of a task tradeoff for this
recipe; it does not identify a gradient mechanism or prove that no full-encoder
recipe can work. It leaves the original short-screen rejection unchanged.

As registered, skip fresh capture and external inference for the rejected arm.
Its checkpoint, logs, validation arrays and transform diagnostic remain intact.
The published-style page/PDF continue to present the single selected tail main
run, without combining private-arm accuracy results. No SOTA claim follows.

Full training consumes **11,618.20 command seconds (3.227 hours)**. Warm median
and p95 are 0.92297 and 1.00392 seconds/update. Observed peak process VRAM is
44,180 MiB (43.14 GiB), board energy 1,300.68 Wh and mean board power 403.06 W.
The recorded shared desktop load briefly accompanies 1.6--1.7-second updates;
causal stage-performance comparisons are not established by these observations.

The completed [discarded batch-capacity probe](pilot-11-batch-probe.md) uses the
retained tail checkpoint and fails its 20% throughput threshold. It does not
change this quality verdict. For
future quality work, test whether a separate geometric adaptation route or an
explicit feature-preservation constraint can retain completion while learning
geometry; neither mechanism is tested here. Camera qualification also requires
a stronger, protocol-matched real-view study and public baselines.
