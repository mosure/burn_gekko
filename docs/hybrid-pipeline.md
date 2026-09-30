# Hybrid reconstruction pipeline

**Historical noncommercial comparison baseline.** The current project trains
encoders and reconstruction without NC pretrained weights; see
[the end-to-end pipeline](e2e-pipeline.md). This hybrid and its derived checkpoints
are excluded from new candidate initialization and teacher supervision.

The user approved adding Gekko's appearance encoder alongside V-JEPA 2.1 to
prioritize reconstruction quality. This pipeline is independently implemented in
Burn. It transfers a large pretrained reconstruction prior; it is not a
V-JEPA-only model or a from-scratch comparison at matched pretraining compute.
Measured results and qualification are tracked in [pilot 04](studies/pilot-04.md).

## Modules and artifacts

| Component | Implementation | Contract |
| --- | --- | --- |
| V-JEPA 2.1 | `crates/burn_jepa`, `src/encoder.rs` | Six pinned upstream encoder files; frozen image path; strict package hashes |
| Appearance encoder and decoder | `src/released.rs` | All 712 released F32 tensors consumed once with exact shapes; sparse target masking before attention |
| Fusion and RGB calibration | `src/hybrid.rs` | Shared pairwise decoder, reference-order invariant RGB averaging, learned patch mean and scale |
| Bounded training and evaluation | `src/hybrid_pilot.rs`, `src/bin/hybrid_pilot.rs` | TOML configuration, fresh output path, local `.data` inputs, wall/step limits |
| Published room generation | `crates/gekko_data`, `tools/zeroverse_capture` | Immutable cached RGB/annotation shards, hashes, split seeds and generator identity |
| Qualification | `examples/{released_audit,hybrid_audit,hybrid_covisibility}.rs` | Official-model parity, hidden-pixel/reference-order interventions, separate dense RI evaluation |
| Independent analysis and PDF | `tools/{transport_diagnostics,hybrid_report}.py` | Raw float32 metrics; no inference enhancement or target-statistic inversion |

The checkpoint source is
[Gekko ViT-L, 500k steps](https://huggingface.co/thibautloiseau/gekko-vitl-500k),
revision `79fba28dd59ec54fffa0134fae681d88ed084513`. Its SHA256 is
`ce415f674dfcbf9d66ba91ebf213abb0010bd115cba72586829e7f5205df6e08`.
Its weights use **CC-BY-NC-SA-4.0**, separately from this workspace's code license.
The recorded reference code revision is
`63f0ec9957885ea82cc2f4637d502003fbf9afcb`. The imported V-JEPA package and its
provenance remain unchanged.

## Inputs and inference

Each camera is encoded independently. In the 256px study, a target supplies
64 of 256 patch tokens; both references supply all 256 tokens. Both encoders
remove hidden target patches before their attention blocks. The V-JEPA adapter
receives semantic features concatenated with RGB from the same observed patches.
Reference features can be cached because both encoders are frozen. Sparse target
features are recomputed for every mask.

The appearance encoder has 24 blocks at width 1024. The pretrained cross-attention
decoder has 12 blocks at width 768. A residual MLP maps the 1536-wide V-JEPA/RGB
input through width 256 into the appearance features. Its last layer starts at
zero. A separate width-128 MLP predicts patch mean and log-scale from decoder
features. Output content is normalized using its own predicted values and combined
with these predicted statistics, then ImageNet normalization is inverted.

The RGB API never receives a dense target representation, target patch statistics,
depth, camera pose or geometric visibility. Dense RI is a separate API operation.
Visible input patches are copied only when constructing a displayed completion;
all reported reconstruction metrics score raw predictions on hidden pixels.

The bounded implementation accepts one to three references. It runs a shared
pairwise decoder and averages their RGB predictions; it is not joint set attention.
The dense co-visibility diagnostic takes the maximum pairwise RI to approximate
visibility in any reference. RI is a utility score, not a calibrated probability.
The study does not establish a sparse-reference speedup or isolated V-JEPA benefit.

## Training

```sh
cargo build --profile pilot --features cuda --locked --bins --examples
target/pilot/hybrid_pilot --config configs/archive/pilot-04/pilot04-hybrid-smoke.toml \
  --run .data/runs/new-hybrid-smoke
```

The larger staged configs are `pilot04-hybrid-screen.toml`,
`pilot04-hybrid-main.toml` and `pilot04-hybrid-detail.toml`. Each stage uses a
new optimizer. `checkpoint` is a weight initialization, **not** an optimizer
resume. Change checkpoint and run paths when reproducing the stages. Defaults
never launch these studies automatically. Use `tools/study/run_study.py` with a fresh
output directory to retain external wall limits and GPU telemetry.

Both encoders remain frozen. The screen trains 758,402 adapter/calibration
parameters. `trainable_decoder_blocks = 2` additionally trains the final two
decoder blocks and final normalization, for 19,667,074 trainable parameters.
The initial decoder blocks and packed RGB/RI output heads remain frozen; training
reports assert the expected unchanged/updated weights. AdamW uses global gradient
clipping, 100-step warmup and a stage-specific cosine schedule.

Training losses use hidden RGB labels: patch mean, log-scale and normalized
content; raw cross-view RGB; optional monocular RGB; and optional adjacent-pixel
gradient error. `gradient_energy_weight` adds a training-only per-image gradient
contrast term. For each direction, it penalizes
`(sqrt((predicted_energy + 1e-7)/(target_energy + 1e-7)) - 1)^2`.
Only adjacent pairs whose two pixels are hidden contribute. This term does not
add target-derived values or sharpening to inference. Alignment and image quality
must be evaluated alongside energy to reject sharp, misplaced structures.

The new adaptation stages do not retrain the dense RI objective. Their RI head is
an inherited diagnostic and must be qualified separately. The original
`gekko train-pilot` trainer still implements the three-path Gekko objective and
optimizer resumption.

## Evaluation and data retention

```sh
target/pilot/hybrid_pilot --config configs/archive/pilot-04/pilot04-hybrid-screen-eval.toml \
  --run .data/pilot-04/new-validation --evaluate validation
# Add --unrelated for a reference intervention, or --disable-adapter to remove
# the combined semantic/RGB adapter (not an isolated semantic ablation).
```

Set `checkpoint` to the selected weights in an evaluation TOML. Set
`export_all_views = true` for independent full-split metrics and failure analysis.
Exports contain target, prediction, references, optional monocular prediction,
the visible mask, and measured MSE. Run `examples/hybrid_audit.rs` on a sample
for a full two-encoder hidden-pixel intervention. Geometry is read only by the
separate co-visibility evaluator after its model forward.

Generation, training and evaluation datasets are cached persistently under
`.data/datasets/<fingerprint>`. The fingerprint includes generator binary and
capture configuration. Reusing a capture verifies hashes and seed/split contracts;
it does not render again. Frozen feature caches are currently GPU-resident per
process, not persistent disk caches. All weights, run logs, checkpoints, telemetry,
raw evaluations and reports stay under `.data`; Cargo artifacts stay in `target`.

For the older decoder evaluator, a fresh dataset requires an explicit
`training_dataset = ".data/datasets/<original-id>"` option. The evaluator verifies
checkpoint/training identity and rejects **any** seed shared with the entire
original dataset. Default evaluation retains strict dataset identity checks.
Use a separate copied evaluation run directory to preserve historical outputs.
