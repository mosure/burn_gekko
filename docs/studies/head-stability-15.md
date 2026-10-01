# Camera and RGB output-head stability

This experiment trains independent camera-calibration and RGB reconstruction
heads from scratch on a frozen Pilot 13 foundation. It does not change the
encoder or fusion weights. No noncommercial weights or teachers are used.

The head study uses 32 existing training rooms (96 targets) and 8 disjoint
validation rooms (24 targets), all at 256 by 256 pixels. These are synthetic
development cohorts already used by the foundation study. They are not an
independent test of generalization. Head validation rooms never enter head
fitting. The primary seed is 857 and the final 800-update checkpoint is used;
no checkpoint is selected from validation results.

## Implemented routes and stability checks

The camera head consumes a separate dense RGB-pair fusion route in both
directions. Each direction is pooled into 4 by 4 spatial cells. A shared
16-channel projection and 64-wide MLP predict two rotation columns, signed
translation direction and log normalized focal lengths. Rotation columns use
the continuous representation of [Zhou et al.](https://arxiv.org/abs/1812.07035).
Regression avoids differentiating through angular acos or normalization at zero.
Scoring separately converts predictions to SO(3), retaining degenerate fits as
180-degree failures. Focal decoding is positive and clamp events are counted.
The principal point is fixed at the image center; translation scale is not learned.

The RGB head maps 768-dimensional predicted latents through a 64-wide hidden
layer to 16 by 16 RGB patches with sigmoid output. It never receives the dense
target image, teacher features or target image statistics. Cross-view and
references-disabled branches share this decoder and receive equal training
weight. Only hidden patches contribute to RGB loss and reported PSNR. PSNR uses
sRGB values in [0,1], data range 1, and equal target-view weighting.

Both heads have separate AdamW optimizers, 32-step warmup, cosine decay, initial
learning rate 0.001, weight decay 0.001 and global gradient clipping at 1 per
head. Each update samples 128 hidden RGB patches and 8 camera pairs. Model and
optimizer records, deterministic sampling, source identity, feature hashes and
configuration are checked across resume. Camera/RGB labels are loaded after the
frozen feature export, and no label is an argument to either head's forward API.

All nine bounded stability gates pass: schedule completion, finite losses and
gradients, both heads changing parameters and reducing fixed training objectives,
valid final rotations, no focal clamps, and exact saved-weight replay. Peak
pre-clipping camera/RGB gradient norms are 4.1301 / 0.05449. The 800-update CPU
training loop takes 64.70 seconds (12.37 updates/second); total head command time
reported by the native runner is 66.44 seconds. This is small-head cached-feature
throughput, not whole-model training throughput or GPU efficiency.

The later native `head-diagnostics` audit independently reproduced all raw-output
metrics and the training-label camera constant. Paired whole-room bootstrap
intervals on these same eight validation rooms give an RGB reference benefit of
**0.394 dB [0.021, 0.722]**. Rotation-error reduction versus the constant is
**1.90 degrees [-0.16, 4.63]**, signed-direction reduction is **52.61 degrees
[38.41, 66.76]**, and focal-error reduction is **1.29 percentage points
[-10.68, 11.00]**. Rotation and focal improvements are uncertain on this small
cohort. These 95% development intervals do not include training-seed variation.
Recipe: `configs/eval/heads15-uncertainty.toml`; output:
`.data/pilot-19/heads15-uncertainty.json`. No weights changed or GPU work was used.

## Results and remaining weaknesses

| Development measure | Initialization | Final head |
|---|---:|---:|
| Hidden-pixel RGB PSNR | 13.46 dB | 21.09 dB |
| References-disabled RGB PSNR | 13.46 dB | 20.70 dB |
| Relative rotation error | 13.44 degrees | 11.64 degrees |
| Signed translation-direction error | 180.00 degrees | 32.68 degrees |
| Focal relative error | 53.73% | 44.35% |
| Pose AUC at 10 degrees | 0.00% | 3.18% |

The initialization's zero translation counts as a failed prediction. A constant
camera predictor fitted only on training labels gives 13.54 degrees rotation,
85.29 degrees translation and 45.64% focal error on the same validation targets.
The learned camera head improves on that control, but absolute calibration is
poor. Camera regression loss is **0.000131 on training versus 0.387406 on
validation**. Training pose AUC is 89.72% versus 3.18% on validation. This is strong
overfitting, not reliable camera estimation. Numerical stability must not be
reported as calibration quality.

RGB error falls, and reference information adds 0.39 dB over the same decoder's
monocular route. PSNR alone does not establish sharp or geometrically coherent
completion. The deterministic annotated examples and all failed legacy transfer
gates remain visible. Full encoder/fusion/head joint training, larger synthetic
coverage, additional seeds and independent real data remain unqualified.

## Reproduction and artifacts

- Cache recipe: `configs/experiments/head-stability15-cache.toml`.
- Training recipe: `configs/experiments/head-stability15-train.toml`.
- Fixed protocol/binary pins/budget: `.data/head-stability-15/`.
- Native models, optimizers, per-update records and full prediction exports:
  `.data/runs/head-stability-15/`.
- [Project page](../../www/project/index.html) and [PDF](../../www/project/paper.pdf).

The GPU export takes 44.993073 command seconds. Combined usage is
43,027.256993 of 43,200 authorized seconds, leaving **172.743007 seconds**.
All head optimization and scoring are CPU work. Desktop processes remain
untouched, and observed board load includes them.

Three-initialization learnability fixtures, zero-baseline/label-detachment tests,
camera-coordinate checks, invalid-rotation/PSNR tests and exact interrupted-resume
tests cover the new paths. Initial 50- and 150-update RGB fixtures did not reach
the fixed 75% loss-reduction assertion; 500 updates pass for every declared seed
with the same threshold. These logs remain available. One scheduler setup attempt
started before the CPU binary was built and stopped before GPU work; its zero-cost
receipt is retained. These are validation development failures, not omitted
training runs or changed scientific tolerances.

## Repository and publication verification

All 164 workspace tests pass, including the three-initialization head fitting and
exact interrupted-resume checks. Warnings-denied workspace and CUDA Clippy,
Rustdoc, all-features compilation, formatting and the encoder import audit pass.
The final reporter tests were rerun after correcting gradient metric counts to
800 optimizer updates; accuracy counts remain 24 validation targets.

The 34-page PDF and self-contained page were regenerated from the pinned run.
Native validation checks 120 file hashes, 103 decoded images and 48 local links.
Browser checks cover 390, 768 and 1440 pixel widths, all six selectable latent
samples, six output-head panels, image decoding and JavaScript errors. Camera/RGB
tables and annotated examples were visually reviewed in the PDF and page. CI now
validates the committed `www/project` bundle without local training artifacts.
