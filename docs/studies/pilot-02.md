# Pilot 02: bounded 256-pixel training study

Status: **completed bounded pilot**. All 65,000 scheduled updates and five held-out
evaluation passes completed in **92.22 cumulative command minutes**, below the
120-minute ceiling. Correct references improve reconstruction, and the RI channel
has modest above-chance co-visibility ranking on this synthetic cohort.

The [17-page PDF report](../../.data/pilot-02/burn_gekko_pilot_02_report.pdf) includes
architecture, data validation, performance/convergence plots, reference controls,
room-level confidence intervals, and four predetermined annotated test examples.
The terminal checkpoint is used throughout; test results did not select it.

The registered protocol is `.data/pilot-02/protocol.toml`. The cumulative ceiling
is 7,200 command-wall seconds for generation, CUDA profiling, feature preparation,
training, and evaluation. CPU compilation and report preparation are outside this
compute ceiling. The workstation is an RTX PRO 6000 Blackwell with 95.6 GiB VRAM;
training uses Burn 0.21.0 CUDA fusion and F32.

## Selection before the main run

Matched 64-update screens used the same six room seeds and the final decoder
shape. Their last 32 updates supplied the sizing measurements; cold compilation
spikes occurred outside that window.

| Resolution | Target examples/s | Median step | Peak process VRAM |
| --- | ---: | ---: | ---: |
| 256×256 | 130.07 | 62.14 ms | 7.35 GiB |
| 384×384 | 60.64 | 129.63 ms | 15.57 GiB |

The preset rule required 384 throughput to reach 65% of 256 throughput and peak
VRAM below 50 GiB. Its measured ratio was 46.6%, so the main run uses 256×256.
This was a throughput choice, not a quality comparison.

The 384 capture initially hit the dataset reader's old size ceiling after all
six rooms rendered. The reader was repaired and the complete capture explicitly
revalidated without rendering it again. The failed command remains in the
compute ledger. All later checks use the repaired reader.

## Data and training configuration

The published generator pins remain `bevy_zeroverse=0.22.0` and
`bevy_zeroverse_burn=0.5.0`. The main cache contains 2,048 training, 64 validation,
and 64 test rooms, each with three static views. It occupies 6,200,455,335 bytes
of raw compressed shards (5.77 GiB) and covers all ten procedural indoor layouts.
Room seeds and files are disjoint across splits. There are no humans in this
cohort.

The geometry audit checks 13,056 directed view pairs and 427,819,007 valid source
pixels. Maximum self-reprojection error is 0.00699 pixels and maximum depth error
is 0.000104 m. Geometry enters audit/evaluation only, never training inputs or
loss targets.

The main preset is in `configs/archive/pilot-02/pilot02-main.toml`: frozen local V-JEPA 2.1 Base,
256-wide/four-layer/eight-head shared decoder, two unordered reference views,
75% target masking, batch eight, and a GPU-resident full-view feature cache.
Room/target pairs are deterministically shuffled each epoch. AdamW uses a 3e-4
peak learning rate, 1,000-step warmup, cosine decay to 5% of peak, weight decay
0.05, and true global gradient clipping at 1.0.

The schedule was fixed at 65,000 steps from the settled 256 screen p90 timing,
with 360 seconds of preparation allowance and a 10% timing margin. The trainer
has a 5,428-second internal wall limit; an external watchdog allows 30 seconds
for orderly finalization. Profiling and main capture used 1,111.58 command
seconds, leaving a 600-second evaluation reserve plus finalization margin.
The terminal budgeted checkpoint is the reporting endpoint.

## Recorded results

| Measure | Result |
| --- | ---: |
| Training command / preparation | 70.51 min / 158.7 s |
| Generation, profiling, training, and evaluation commands | 92.22 min |
| Warm throughput / median / p90 update | 128.3 examples/s / 58.82 ms / 68.22 ms |
| End-to-end training throughput | 122.9 examples/s |
| Sampled peak process VRAM | 22.77 GiB |
| Mean device activity / power after first 10 updates | 85.9% / 343.3 W |
| Target examples / room-view passes | 520,000 / 84.6 |
| Validation total loss, initial → final | 5.0132 → 0.9422 |
| Test total loss, initial → final | 5.1704 → 1.0116 |
| Validation pooled pixel AUROC, initial → final | 0.5014 → 0.6117 |
| Test pooled pixel AUROC, initial → final | 0.5014 → 0.5988 |
| Test AP / visible-pixel prevalence baseline | 0.8822 / 0.8432 |

The final 5,000-update training window improved total loss by only 0.31% and
cross MSE by 0.42% versus the preceding 5,000 updates. The preset had reached a
practical plateau; this is not a proof of full optimization convergence.
Gradient clipping occurred on 78.7% of updates. Concurrent Rust compilation was
observed during throughput dips near step 45,000 and later. Device activity is
shared with desktop work, and no process priorities or GPU settings were changed.

On the complete test split, cross MSE is **0.47149**, target-only MAE MSE is
**0.48356**, and unrelated-reference cross MSE is **0.57837**. Correct references
reduce MSE by 18.48% versus unrelated references and 2.50% versus the target-only
branch. Paired room bootstrap differences are:

- Unrelated minus correct cross MSE: **0.10688 [0.09130, 0.12644]**.
- Target-only minus correct cross MSE: **0.01207 [0.00920, 0.01502]**.

The target-only branch changes by exactly zero under the intervention. Test
macro room AUROC is **0.5998 [0.5829, 0.6166]**; validation macro AUROC is
**0.6152 [0.5997, 0.6316]**. With original labels held fixed, unrelated references
reduce pooled test RI AUROC to 0.5315; this is a sensitivity diagnostic, not
visibility ground truth for the unrelated scene.

A post-hoc constant-zero prediction in normalized RGB has test MSE **0.98027**,
so the learned reconstruction is materially better than simply suppressing
random initial outputs. About 35.9% of test patches have low spatial contrast
under the stated 0.01-sRGB threshold. This descriptive check did not affect data,
hyperparameters, or checkpoint selection.

The four predefined visual examples include target-view AUROCs 0.487, 0.565,
0.458, and 0.568. Reconstructions remain blocky, and thin occlusion boundaries
are poorly resolved by the RI maps. The reference benefit and modest ranking
signal do not establish accurate correspondence, calibrated visibility, pose,
pointmaps, or real-data transfer.

## Evaluation and report

The completed endpoint analysis compares initial and terminal weights on all
three target views of every validation/test room. A terminal test intervention
replaces references with views from the next held-out room, while fixing target,
mask, and weights. Its primary metric is reconstruction error; original geometric
labels are not relabeled as visibility to the unrelated room.

Room-level bootstrap intervals use 2,000 resamples and a fixed analysis seed.
Pooled pixel AUROC and AP, AP prevalence baseline, reconstruction losses, and
paired reference interventions are separate endpoints. The four annotated test
rooms are selected at fixed evenly spaced indices, before seeing predictions.
Normalized reconstruction displays use ground-truth patch mean/std and are
explicitly marked as oracle visualizations.

Validation passed: 43 Rust tests during implementation, focused continuation and
annotation-export checks, strict root CUDA/capture-tool Clippy, formatting, and
two CPU report contract tests. Final artifact checks verify 20 runtime source
and binary digests, decoder/AdamW checksums, all 6,144 training room/target pairs
visited 84–85 times, and five complete evaluations. Independent NumPy
recomputation of all four annotated samples' MSEs agrees with Rust/CUDA within
5.97e-8. These checks validate this run's contracts; they do not qualify original
encoder checkpoint parity.

Artifacts:

- `.data/pilot-02/`: protocol, sizing decision, command plans/ledgers, telemetry,
  geometry/inventory, analysis outputs, and the final PDF.
- `.data/runs/pilot-02-main/`: resolved TOML config, step metrics, fixed probes,
  initial/periodic/terminal model and AdamW checkpoints, held-out evaluations.
- `.data/datasets/730e17376a84e3294707986df36700d1311accc1e76faaca45aaebeeb0c8fb9a/`:
  immutable source shards, capture config/log, and hashed split manifest.
- `.data/pilot-02/machine-source.json`: hardware/software identity and runtime
  source/binary SHA-256 digests recorded before the main run.
- `.data/pilot-02/bin/gekko` and `source-snapshot.tar.gz`: exact runtime binary and
  source snapshot; `analysis-environment.txt` records report package versions.
- `.data/pilot-02/verification.json`: numerical, coverage, source, and checksum
  verification receipt. `pdf-review/` contains the rendered pages reviewed for
  layout and annotation integrity.

Regenerate the completed report using:

```sh
.data/analysis-venv/bin/python tools/legacy/render_pilot_report.py \
  --study .data/pilot-02 --run .data/runs/pilot-02-main \
  --dataset .data/datasets/730e17376a84e3294707986df36700d1311accc1e76faaca45aaebeeb0c8fb9a
```

This is one synthetic-data training seed, not a reproduced Gekko benchmark.
Official-checkpoint native-image parity and real-world transfer remain open.
Further work should qualify encoder parity, repeat seeds, and add matched-compute
MAE-only/CroCo-only and one-reference controls before a paper-level comparison.
No additional training was launched after the registered endpoint.
