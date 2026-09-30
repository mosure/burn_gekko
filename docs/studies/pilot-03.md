# Pilot 03: reconstruction diagnosis and controlled comparison

Completed on 2026-09-28 UTC. **Partial reconstruction improvement; coherent
cross-view detail and co-visibility remain inadequate.**

[PDF report with annotated samples](../../.data/pilot-03/burn_gekko_pilot_03_report.pdf)
contains the full screen, convergence curves, resource measurements, confidence
intervals and predetermined held-out room examples. [Test contact sheet](../../.data/pilot-03/main-candidate-test.png).

## Main result

Both models used 512 training / 32 validation / 32 test rooms, three views per
room at 256×256, a frozen official V-JEPA 2.1 Base encoder, a 384-wide six-layer
six-head decoder, batch 16, 75% masking, two reference views, seed 29 and 9,000
updates. Each arm saw 144,000 target/reference tuples (93.75 shuffled passes).
The terminal checkpoint was fixed before test evaluation.

| Test endpoint | Semantic-only baseline | Calibrated candidate |
| --- | ---: | ---: |
| Hidden RGB MSE, all 96 views | 0.005452 | 0.003941 |
| Pooled co-visibility AUROC | 0.4949 | 0.5290 |
| Average precision | 0.8255 | 0.8449 |

The candidate reduces test RGB MSE by **27.7%**. The paired 32-room bootstrap
for baseline-minus-candidate MSE is 0.001511, with 95% interval
[0.001233, 0.001802]. Test-export edge cosine rises from 0.0131 to
0.1053 on 16 predetermined target-zero views. Predicted edge energy is only
6.3% of target energy on those candidate exports; fine structures remain blurred.
The baseline's larger edge energy includes poorly aligned patch discontinuities.

Correct references reduce candidate RGB MSE by 24.3% versus unrelated rooms.
The monocular branch is unchanged under that intervention (maximum loss change
zero). This establishes reference sensitivity, not coherent geometric matching.
Cross-view content MSE is only 0.48% below the monocular content MSE. On the 16
exported test views, standalone monocular RGB MSE is 0.003697 versus 0.003614
cross-view; the paired difference interval includes zero. These subset RGB
metrics and all-view normalized-content metrics have different scope and units.

Co-visibility remains weak: pooled AUROC 0.5290, room/view
macro AUROC 0.5409 (room-bootstrap interval
[0.5279, 0.5547]). Positive prevalence is
0.8286, explaining much
of the apparently high AP. RI is an unconstrained reconstruction-utility score.

## What changed and what was checked

- Published capture dependencies are now `bevy_zeroverse=0.23.0` and
  `bevy_zeroverse_burn=0.6.0`, with verified archive checksums and a new cache
  identity. Registry versions were rechecked during closeout.
- Corrected the MAE cross-attention context to the pre-block target used by the
  audited Gekko source. The compatibility flag preserves historical behavior.
  Both arms of the main comparison enable the correction; its isolated effect
  was not measured.
- Added optional observed-RGB features, 2D rotary attention, and learned patch
  mean/log-standard-deviation prediction alongside normalized reconstruction.
  The candidate changes these together, so this is a recipe comparison.
- The old normalized-head displays use hidden target patch statistics. They are
  explicitly marked as oracle displays. The candidate predicts its own
  statistics. Display composites copy only visible input patches; metrics use
  unclipped predictions, and no sharpening is applied.
- All 158 encoder tensors match the official EMA checkpoint after F16
  conversion. Dense/sparse CPU outputs agree with official PyTorch using the
  same quantized weights to relative RMS below 0.000004. CUDA encoder error up
  to 0.0029 fails the original 0.001 gate and remains separately documented.
- A fixed-image/mask overfit reaches normalized cross MSE 0.000369 and edge
  cosine about 0.999. Ground-truth geometry warping gives roughly 24× lower RGB
  error than unwarped references on covered pixels. These diagnostics argue
  against gross patch-layout or RGB/geometry misalignment, without proving
  generalization or learned matching.
- The corrected CPU/CUDA decoder prediction/input-gradient audit passes its 1%
  relative RMS gate; worst error is 0.002183. All three exported head weight
  arrays match exactly. Its scope is a two-block, width-384, batch-two decoder
  on a 4×4 token grid, not full published-Gekko parity.

The first backend audit was **invalid** because a lazy CPU parameter clone was
serialized and a separately initialized original was evaluated. Both backends
now reload one identical checkpoint. The pilot trainer already materializes its
record before cloning, so the training runs were unaffected. The invalid result
and its 147.5 seconds remain in the evidence and cumulative budget.

## Selection and evidence limits

The five 5,000-update screens improved standalone RGB error, but the calibrated
screen's RI AUROC 0.5153 failed the amended 0.53 gate. Advancing it into a fresh,
explicitly exploratory reconstruction study was a recorded protocol deviation,
not a qualification pass. Main validation/test AUROC also remains below 0.53.

Main seeds are disjoint from all pilot-02 rooms. Some screening seeds overlap
historical pilot-02 training seeds, regenerated with the new packages; screen
decoders start fresh. Main train/validation/test seeds are mutually disjoint.
The test split covers nine of ten training layout families. One optimization
seed, one evaluation mask and synthetic rooms do not establish transfer.

Measured study-command time is **90.21 minutes**, below
the 120-minute ceiling, including failed/invalid audit commands, capture,
training and evaluation. Setup, builds, CPU plotting and report writing are
separate. Main warm throughput is about 120 examples/s. The PDF separates warm
throughput from total command throughput and records process VRAM. A cold
384-wide/eight-head benchmark timed out before a step; no warm throughput is
claimed for that configuration. Intermittent slow training steps remain included.

## Reproduce and inspect

All user-facing inputs are TOML. Dataset, logs, models, floats and reports are
under `./.data/`. The main immutable cache is
`4287ae3d171d81e05febe5ad057d2ca448a03d006f0da30e018c5350e90ec149` (1.65 GB).
The exact training runtime and source snapshot are retained in
`.data/pilot-03/bin/gekko` and `.data/pilot-03/final-source.tar.gz`.

```sh
OPENBLAS_NUM_THREADS=1 .data/analysis-venv/bin/python tools/legacy/render_quality_report.py \
  --config configs/train/report-pilot03.toml
```

Training inputs: `configs/pilot03-main-{baseline,candidate}.toml`.
Command plans and telemetry: `.data/pilot-03/{main,final-evaluation}/`.
Selection receipts: `protocol.toml`, `main-decision.json` and
`test-checkpoint-freeze.json` in the study directory. The machine-readable
`quality-summary.json` contains measurements and room-level uncertainty.
See [source provenance](../source-audit.md) for registry and checkpoint hashes.

Local verification: **48 Rust tests**, strict default/CUDA Clippy, capture
Clippy, formatting, encoder import hashes, two report-data tests, independent
exported RGB checks, the corrected decoder numerical audit, and one historical
checkpoint evaluation with exactly matching losses. No remote CI result is
claimed. Original pilot-02 artifacts are preserved.

## Next experiments justified by this result

1. Establish useful reference-dependent reconstruction on paired known warps
   and very small camera baselines, requiring sharp held-out edges and a clear
   advantage over the monocular branch. An exact-image memorization test alone
   is insufficient.
2. Test pairwise conditioning and a reference/overlap curriculum before joint
   multi-reference fusion; instrument correspondence or attention behavior.
3. Separately ablate the audited decoder differences, masking schedule and
   explicit reference-RGB transport or richer appearance features. Scale data
   and compute only after those controls demonstrate learned correspondence.
