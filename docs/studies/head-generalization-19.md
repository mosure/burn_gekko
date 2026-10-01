# Pilot 19: conditional output-head generalization study

Registered before the Pilot 19 foundation outcome. Execute only if that candidate
passes every unchanged synthetic retention criterion. A failure does not redirect
this study to another checkpoint. This study does not change the foundation's
selection decision or unfreeze its encoder/fusion weights.

The existing small head study demonstrates numerical stability but substantial
camera overfitting. This follow-up expands supervision on the accepted candidate
to the first **512 training rooms** and **64 validation rooms** in the immutable
8,192-room training cache. All three 256-pixel views are included: **1,536 training
targets and 192 validation targets**. Validation is reused development data;
neither fitting nor checkpoint selection may consume reserved test rooms.

## Frozen features and independent heads

Keep 90% random target masking, two references and mask seed 857. Export sparse
completion and references-disabled latents plus separate dense, bidirectional
camera-pair features. Cameras/depth/RGB targets remain labels loaded after model
inference. The native cache exporter streams one room at a time; its explicit
volume limit increases to 1,024 training / 128 validation rooms, without changing
inference math. Pin the candidate checkpoint, source, recipe and cache manifest.

Train the existing 64-wide RGB and camera heads from scratch with seeds **857,
858 and 859**, in that order. **857 is the primary reporting seed**, fixed before
outcomes; retain all three. Each run has **8,192 updates**, 128-step warmup, cosine
decay to 10% of the initial learning rate, learning rate 0.001, weight decay 0.001,
256 sampled hidden RGB patches and 16 camera pairs per update. Keep separate
AdamW optimizers, global clipping at 1 per head, regression representations and
loss weighting unchanged. Evaluate every 1,024 updates for diagnostics only.
Use the final endpoint, never the best validation probe. Each CPU run has a
4,000-second cap; a shortened or failed run is incomplete and cannot be omitted.

Three seeds examine head initialization and sample-order sensitivity, not
foundation-training seed variance. This is an expanded head experiment, not an
isolated causal comparison of dataset size against the old 32-room study.

## Budget and scoring

Require the completed foundation selection and no active GPU experiment. One
cache-export command has a **900-second cap**, charged to the same
`.data/pilot-18/budget.json`; admit it only with at least 910 seconds plus a
**3,600-second reserve** remaining. CPU head fitting/scoring does not consume GPU
allowance. Preserve process/board telemetry and do not rerun failed exports.

Require all nine existing numerical stability checks, complete populations,
checksum-bound raw prediction replay, and training/validation separation. Native
head diagnostics recompute RGB and camera scores, fit a constant calibration
predictor using training labels only, and bootstrap paired **whole rooms**.
Report hidden-pixel sRGB PSNR in dB, paired reference benefit with a 95% interval,
rotation and signed direction error in degrees, focal relative error in percent,
pose AUC@10, train/validation gaps and all three seed outcomes. Keep invalid
rotations and zero-direction failures in the angular metrics.

A useful primary development result requires positive lower 95% bounds for RGB
reference benefit and each camera-error reduction against the training-label
constant, plus primary mean rotation <=10 degrees, signed direction <=30 degrees,
focal relative error <=35%, and pose AUC@10 >=10%. All seeds must complete with
stable optimization and positive point gains against the same controls. These
are engineering progress criteria, not SotA thresholds or independent evidence.
Retain every failure; do not revise thresholds or pick a different seed.

The single-run page/paper uses only this foundation and seed 857's attached heads,
with the other seeds documented as sensitivity results. A failed calibration
criterion must remain explicit. Real-view qualification and deployment remain
separate from synthetic head fitting; do not automatically replace the public
model or consume unused compute allocation.
