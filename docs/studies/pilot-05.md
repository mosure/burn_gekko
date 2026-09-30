# Pilot 05: end-to-end reconstruction without NC pretrained weights

**End-to-end training is implemented and verified; the blur is not resolved.**
No candidate meets the reconstruction-quality gates, and learned co-visibility
is approximately chance. The selected checkpoint is diagnostic only.

[PDF report with all comparisons, annotated test samples and errors](../../.data/pilot-05/burn_gekko_pilot_05_report.pdf).
[Machine-readable results](../../.data/pilot-05/quality-summary.json).
[Training and checkpoint guide](../e2e-pipeline.md).

## Scope and provenance

Every fusion decoder, reconstruction head and relative-improvement head starts
randomly. One main arm also starts its Base image encoder randomly. The other
arms initialize only the image encoder from the audited MIT V-JEPA 2.1 package.
Released Gekko and pilot-04 hybrid checkpoints are excluded from initialization,
continuation and teacher supervision. DINOv3 is not imported in this study.

The V-JEPA package is `c408f68dd18a38824d0fa1d615e6f9f9f04f111d71a6f7c846f41dc187a8795f`.
Its 158 encoder tensors were compared with the official EMA checkpoint in pilot
03, after the documented F16 storage conversion. The pinned upstream MIT license
and source/checkpoint hashes are retained in `.data/pilot-05/licenses/`.

The implementation recomputes encoder features on each update, uses separate
AdamW states for encoder and decoder, and resumes both states together with the
absolute sample/mask schedule. Pretrained models first train fusion, then the
last two image blocks, then all image blocks and the patch stem. Validation gates
require a minimum 500 updates per stage and measured improvement. The unused
video tokenizer stays frozen. The random arm trains jointly from its first step.

## Dataset and experimental boundaries

Published `bevy_zeroverse=0.23.0` and `bevy_zeroverse_burn=0.6.0` generated a new
256px, three-camera dataset: 2,048 training / 32 validation / 64 reserved-test
rooms. All 2,144 room seeds are disjoint from every previous dataset cache.
The 6.11 GB cache took 743.0 seconds to generate and remains on disk at
`.data/datasets/69c9a724baf69f1a610c122d2b3b3599a8c63fca4701ccdc6bdc8e6dcbbf667e`.

Main models use 12-block, 768-wide image encoders and fresh 6-block, 384-wide
fusion decoders. Each target has 25% visible patches and two dense references.
The target is sparse before encoder attention. Training includes RGB, normalized
content, predicted statistics, hidden-pixel gradients and detached RI targets;
renderer geometry is evaluation-only. Main runs request 4,000 updates, batch 8.
The scratch encoder learning-rate ratio is 0.5; pretrained encoders use 0.1.
These schedule differences preclude attributing all differences to initialization.

Recorded exploratory amendments retain every result:

- Early validation exposed repeated, unaligned detail with the initial .02
  gradient-energy penalty. Subsequent arms remove it. Energy alone is not a
  quality metric; this failure is retained as a separate arm.
- A fresh attention-normalization ablation adds per-head Q/K LayerNorm in both
  decoder attention slots. Released Gekko uses it in self-attention; extending it
  to cross-attention here is a separate design choice, with no weight transfer.
- Selection prioritizes validation detail and reference controls, then hidden
  edge cosine. If no model qualifies, the highest-edge-cosine checkpoint is
  evaluated on reserved rooms as a diagnostic, without a success claim.
- One-room and optional 16-room fit diagnostics are excluded from selection.
  No tuning follows the reserved-test readout.

Capture, shader compilation, setup, training, checkpoints and GPU evaluations
share a cumulative 7,200-second command budget. CPU compilation and reporting
are separate. Device-wide utilization includes other workstation activity;
process VRAM is recorded separately. Other `indoor_validate` processes were
observed and left untouched.

## Full validation comparison

Every main arm completed 4,000 updates on the same 2,048 training rooms, then
exported all 96 targets from 32 validation rooms. No candidate meets the required
0.4 edge cosine and 0.5 minimum gradient-energy ratio.

| Initialization / recipe | Hidden MSE | Edge cosine | Edge-energy ratio | Monocular MSE | Metered command |
| --- | ---: | ---: | ---: | ---: | ---: |
| MIT V-JEPA, energy penalty .02 | 0.003971 | 0.0108 | 0.8922 | 0.003884 | 24.6 min |
| Entirely random, no energy penalty | 0.003769 | 0.0570 | 0.0421 | 0.003951 | 22.1 min |
| MIT V-JEPA, no energy penalty | 0.003505 | 0.0633 | 0.0531 | 0.003638 | 18.2 min |
| MIT V-JEPA, no energy penalty + Q/K norm | 0.003371 | 0.0703 | 0.0505 | 0.003549 | 21.8 min |

The energy penalty produces substantial unaligned detail: near-target gradient
energy does not imply useful reconstruction. Removing it lowers error and improves
alignment, but most within-patch detail remains missing. The Q/K variant leads
this validation comparison, with only 1.47% of target within-patch gradient energy.
Its cross-view error is 5.0% below its monocular branch, below the 10% requirement.

Warm full-encoder training measured 21.0, 25.9, 29.4 and 23.9 target examples/s,
respectively (first 50 updates of each stage excluded). Peak process VRAM was
about 53.0, 24.9, 53.0 and 53.7 GiB. These are observed runs with different staging
and other workstation activity, not isolated kernel benchmarks. Every pretrained
arm executed 500 frozen / 500 tail / 3,000 fully unfrozen updates.

## Verified implementation and memorization

The 57-test workspace suite and strict CUDA Clippy passed for the training changes.
An additional CPU batch-equivalence contract also passed (58 verified tests total).
Tests cover gradient routing and parameter updates under frozen/tail/full stages,
hidden-target isolation, reference permutation on CPU, exact CPU optimizer
continuation, and rejection of unreviewed or NC-listed checkpoint provenance.
The pinned encoder source files remain unchanged. A real CUDA smoke trained the
image stem, saved both optimizers and exported all validation targets.

A smaller random 192-wide/four-block encoder and decoder fit one fixed training
room/mask in 1,129 updates before its five-minute cap. Hidden RGB MSE was
0.0001917, edge cosine 0.8851, and gradient-energy ratio 0.7324. Training target
RGB is exported for inspection. This establishes the ability to express and
optimize detailed reconstruction on that input. It does not validate transferable
cross-view correspondence or generalization; held-out quality is assessed
separately.

Both training executable generations are preserved in `.data/pilot-05/bin/`,
with source archives and hash receipts. The v2 loader reads v1 model records with
a measured smoke-output maximum difference of 6.85e-6 on CUDA. This is approximate
load compatibility, not bitwise equivalence. Use the archived v1 executable for
its original checkpoint configuration/code identity.

## Reserved test and acceptance

The validation-selected Q/K-normalized checkpoint was frozen before testing all
64 reserved rooms / 192 target views. All predictions were exported and scored
independently from raw float32 arrays. The checkpoint SHA256 is
`010ad6de1eb0812199e95cfe8e0065345f5093409a98cf19e2285daf670be0b8`.

| Readout | Reserved test |
| --- | ---: |
| Hidden RGB MSE | 0.004292 (95% CI 0.003726–0.004951) |
| Hidden edge cosine | 0.0759 (0.0701–0.0823) |
| Gradient-energy ratio | 0.0541 |
| Within-patch gradient-energy ratio | 0.0158 |
| Monocular hidden MSE | 0.004416 |
| Unrelated-reference hidden MSE | 0.005673 |
| Improvement over monocular | 2.82% (required: 10%) |
| Pooled co-visibility AUROC | 0.4998 |
| Co-visibility AP / constant-score prevalence | 0.8437 / 0.8430 |

Mean per-target AUROC is 0.4904 (room-bootstrap 95% CI 0.4843–0.4962).
These scores do not establish useful co-visibility. Approximately 83.8% of known
hidden pixels are visible in a reference, so missing detail is not explained
solely by occlusion. Reference-visible RGB MSE is 0.004165 versus 0.005519 on
reference-absent pixels (room/view means, unknown labels excluded).

The hidden-target intervention changes RGB by exactly zero. CUDA reference
permutation changes RGB by 1.291e-4, exceeding the predeclared 1e-5 gate; the
cause was not isolated and the tolerance was not relaxed. CPU reference-order
and batched-forward properties pass. Monocular MSE differs by at most 2.80e-9
between correct- and unrelated-reference evaluations.

The 16-room varying-mask diagnostic completed **895 of 1,000 requested updates**
before its 300-second training cap. All 48 training targets were exported and
independently checked: MSE 0.005218, edge cosine 0.0150. Validation MSE was
0.004955, edge cosine 0.0108. This run did not yet fit its training targets;
it cannot be described as successful multi-room overfit or a generalization-only
failure. It is excluded from selection. Both optimizer records were retained.

Total capture and experiment command time was **6879.5 seconds
(114.7 minutes)** of the 120-minute ceiling. All scheduled commands
completed; the two memorization diagnostics stopped internally at their wall
limits and saved checkpoints. No GPU jobs from this study remain running.

## What this establishes, and what remains

The project can now train image encoders and fusion end to end from random
initialization or audited MIT V-JEPA weights, with no released Gekko parameters
or teacher. This does not establish an accepted reconstruction model. The
small varying-mask diagnostic also underfits, so a simple claim that only
held-out generalization is missing would be too strong.

The reconstruction objective is an adapted recipe: raw cross-view MSE has weight
10, monocular MSE 5, and RI 0.1, alongside normalized content/statistics and
cross-view edge terms. It is not an exact reproduction of the released Gekko
training loss. Unequal branch weights can confound error-comparison utility;
this study does not isolate their effect. Prediction content is explicitly
normalized before learned RGB calibration, another unisolated choice.

The next bounded study should first establish full-size, varied-mask 16-room
convergence; compare matched branch objectives and delaying RI until RGB is
stable; and test unconstrained normalized content or direct RGB heads. Only then
scale training duration/data. The CUDA reference-order residual needs a separate
precision diagnosis. These are proposed experiments, not proven quality fixes.
Frozen-encoder controls, independent training seeds, real-image transfer and
sparse-reference qualification remain open. Prior NC baseline scores use other
rooms and masks and are not a matched percentage comparison here.

## Reproduction artifacts

User recipes are `configs/pilot05-*.toml`; capture uses `capture-pilot05.toml`.
Study protocol, amendments, selection rule, immutable command plans and telemetry
are under `.data/pilot-05/`. Full RGB/monocular exports and model/optimizer records
are under `.data/runs/pilot-05-*/`. `tools/legacy/e2e_report.py` independently checks raw
float predictions against recorded MSE, calculates room-bootstrap intervals, and
renders unenhanced comparisons, detail crops, error maps and geometry annotations.
