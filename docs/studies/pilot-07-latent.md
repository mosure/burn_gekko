# Pilot 07: latent prediction becomes primary

The user confirmed V-JEPA 2.1 latent prediction as the primary task, with RGB
reconstruction optional. The new training pipeline is implemented and its first
bounded CUDA screen is complete. This is a change in task; prior RGB blur remains
unresolved and is not a success criterion for the latent experiment.

Read the [eight-page annotated PDF](../../.data/pilot-07/latent-report.pdf),
[machine-readable comparison](../../.data/pilot-07/latent-report.json), and
[pipeline/objective specification](../latent-pipeline.md).

## Completed experiment

- 256 training rooms, 16 scene-disjoint validation rooms / 48 targets, 256×256,
  two reference views, 75% hidden target patches. Existing immutable data from
  published Zeroverse 0.25.0 and Burn adapter 0.8.0; no new capture or test use.
- A fixed MIT V-JEPA 2.1 teacher supplies normalized dense targets only to loss
  and evaluation. The student starts from that same audited package. Fusion and
  the shared cross-view/monocular latent head start randomly. No released Gekko,
  noncommercial teacher, or previous RGB checkpoint initializes this model.
- 1,000 updates / 16,000 target exposures. Student frozen for 400 updates, last
  two blocks train for 400, full image encoder for 200. Both transitions use
  observed validation stability. The original teacher never enters an optimizer.
- Config: [pilot07-latent-screen.toml](../../configs/archive/pilot-07/pilot07-latent-screen.toml).
  Binary and source archive are preserved under `.data/pilot-07/`.

| Validation metric | Result |
| --- | ---: |
| Initial hidden latent MSE | 1.36024 |
| Final cross-view MSE | 0.21008 |
| Matched monocular MSE | 0.21561 |
| Unrelated-reference MSE | 0.23972 |
| Training-only constant position predictor MSE | 0.30642 |
| Cross-view cosine similarity | 0.88853 |
| Cross-view MSE reduction versus monocular | 2.56% |
| Prediction / teacher spatial variance | 0.352 |
| Hidden-feature effective rank, prediction / teacher | 22.36 / 101.09 |
| Raw latent-gain / learned-RI co-visibility AUROC | 0.604 / 0.596 |
| Learned RI AP / prevalence baseline | 0.885 / 0.843 |

The paired room mean of monocular minus cross-view MSE is 0.005529,
95% bootstrap CI [0.003731, 0.007361]. Unrelated minus related reference MSE
is 0.029645, CI [0.026669, 0.032653]. These support a small reference benefit.
They do not establish dense correspondence or strong visibility prediction.
Rank is measured on 2,048 deterministic hidden tokens from exported validation
samples; reduced rank and variance show that feature detail is still limited.

Geometry evaluates hidden patches with at least 128 known pixels, using majority
visibility in any reference. All 9,216 hidden validation patches were evaluable;
7,771 were positive. This patch protocol is different from earlier pixel-level
RGB RI studies, so AUROCs are not directly comparable. No geometry is used for
training. Sample pages show unaltered RGB context, one shared teacher-fitted PCA
basis, latent errors, error gain, and renderer visibility. PCA panels are not
decoded RGB images.

## Efficiency and correctness

The training command took 836.8 seconds including dataset verification,
initialization, checkpoints, and evaluation. Warm update medians by stage were
0.570s frozen, 0.639s partial encoder, and 0.817s full encoder. The aggregate
median was 0.628s / 25.5 targets per second. Process VRAM peaked at 77,272 MiB
(75.5 GiB); future full-batch changes need to account for this measured peak.
Raw data remain cached on disk in `.data/`; selected RGB is resident on the host.
Both teacher and student features are currently computed online.

All 78 workspace tests passed. Strict CUDA Clippy and native builds passed;
the strengthened exact-resume test exercises both frozen and nonzero student
gradient stages and restores the two AdamW states and sampling. Hidden RGB has
zero measured influence on masked latent prediction in CPU tests and the native
audit. Teacher isolation uses a non-autodiff backend; its sampled first-block
parameter delta is zero. Student first/last QKV deltas are nonzero.

Default attention fails the strict `1e-5` reference-order tolerance. An additional
16.8-second, unchanged-weight audit uses F64 softmax/value accumulation: on the
first validation target, maximum/RMS permutation differences change from
0.000954 / 0.000117 to zero. Latent MSE changes only from 0.20106477 to 0.20106626.
This is an isolated inference verification; the reported training run still
used the default policy. Full-batch gradient/throughput and broader numerical
qualification are required before changing the training default.

Follow-up: the full CUDA Fusion evaluator rejected that float64 policy with
`Unsupported precision for fusion: f64` before any updates. The narrow result
above remains recorded, but does not qualify the policy for production.
The [controlled continuation](../sota-evidence.md) uses float32 and retains the
numerical order-sensitivity audit.

Final model SHA256:
`de8656ebd1911a10a61b090e43a0ced1c4add9cd328373f1d03e1eb0ba72e8c8`.
Checkpoint: `.data/runs/pilot-07-latent-screen/final`.
All GPU command time is charged to the existing Pilot 07 cumulative budget.

The next controlled work is to test contiguous/harder masks and same-room
spatially shuffled references, expand the training rooms, and measure held-out
correspondence and visibility on fresh scenes. Keep the frozen teacher for
comparable targets while testing those changes. A momentum teacher or
hierarchical targets should be a separately labeled ablation. The current result
supports the latent direction but remains a single-seed diagnostic.
