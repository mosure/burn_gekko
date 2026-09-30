# Known-transform correspondence study

Status: registered development preflight, 2026-09-30. No additional GPU allowance
has been assumed. The existing Pilot 07 ledger has 565.22 command seconds remaining
before this preflight. Capture, training and GPU evaluation all consume it.

## Hypothesis

The preceding affinity-sharpening head lacked verified geometric targets and
failed real-viewpoint transfer. This experiment replaces that auxiliary loss with
correspondences obtained from known transforms of training RGB. It tests geometric
learning, while sparse V-JEPA latent completion remains the primary task.

[SiLK](https://arxiv.org/abs/2304.06194) provides precedent for descriptor supervision
from transformed single images. This is an independent Burn implementation, not a
SiLK reproduction: no SiLK code, weights or keypoint detector are used.

## Architecture and objective

- Start from the audited native-spatial-refinement checkpoint
  `f0d79d2cb53c9c3cf4ddb870efb6167898db42bb5b67e982a56ffdd5f3fa0d82`.
  Reset both optimizers and initialize the bounded residual descriptor head at zero.
  Do not inherit the failed sharpening phase.
- Freeze the V-JEPA encoder for this diagnostic. Final and block-6 features feed
  the existing six-block fusion trunk. The head's correction norm is bounded by
  25% of the centered block-6 descriptor norm.
- Make a projectively warped, photometrically modified copy of each target RGB.
  Known homographies generate bilinear patch-grid label distributions in both
  directions. Whole query patches must remain inside the other image, and their
  centers must remain inside its descriptor grid. Invalid rows have zero weight.
- Pixel-edge coordinates are explicit: image-pixel centers are `(x+0.5,y+0.5)`;
  descriptor centers are `16*(column+0.5,row+0.5)`. Inverse RGB sampling and forward
  labels use the same transform. Transform parameters never enter fusion inputs.
- Normalize descriptor cosines and apply bidirectional valid-row NLL at 0.07.
  Train pair-conditioned descriptors and a shared-weight same-image-conditioned
  control equally. Auxiliary loss is `0.1*(pair_NLL+self_NLL)/2`.
- Keep sparse completion, monocular completion and RI losses unchanged. Dense
  augmented images do not enter the sparse completion forward path. No scene
  depth, camera, pose, geometry or noncommercial teacher is used in training.
- A feature-only fusion forward omits unused attention diagnostic score matrices.
  CPU forward and gradient parity qualify this refactor; CUDA behavior is measured
  during the bounded preflight, not inferred from CPU timing.

## Fixed preflight

`configs/experiments/pilot07-equivariance-preflight.toml`: 128 training rooms,
256x256 RGB, batch 16, two references, 90% random masking, seed 733, 256 updates,
235-second internal wall limit. Eight validation rooms, probes at 128/256 updates.
The final endpoint is used; no checkpoint selection on benchmark geometry.
An early wall stop is reported as incomplete, never silently relabeled a full run.

Predeclared follow-up evaluation, within the existing allowance: 16 validation
rooms under fixed known transforms (seed 1000733); complete HPatches if sufficient
budget remains; 16-room latent assessment. HPatches and ETH3D are development data
because earlier decisions used their results. No fresh held-out claim is made.
The preflight is a small-data convergence and efficiency check, not a full study.

## Readouts and acceptance boundaries

The known-transform diagnostic measures NLL, endpoint error in model-input pixels,
PCK8 and PCK16, equally weighted by room and direction. It compares the same frozen
checkpoint's pair-conditioned head, self-conditioned head and centered block-6
encoder. Display the first four rooms without selecting favorable samples.
Geometric augmentation success alone does not establish 3D viewpoint transfer.

For a subsequent fully budgeted study, predeclare the primary external readout
`spatial_residual_conditional` and controls `spatial_self_conditional` and
`student_l06_centered_conditional`. A fusion claim requires positive paired
scene/sequence-bootstrap AEPE gains against both controls, without PCK3 degradation,
on complete ETH3D and HPatches viewpoint data. Also require latent completion not
to regress on a fresh room cohort. Do not choose readouts per image or adjust the
objective using external benchmark outcomes within the registered study.

Encoder unfreezing is deferred until objective stability and geometric learning
are established. A larger phase needs an explicitly authorized new ledger and
enough reserved time for complete external evaluation and fresh-room assessment.
No state-of-the-art claim is supported by a preflight, one seed, patch-grid readouts
or development-only benchmark gains.

## Evidence and reproducibility

Runtime artifacts live under `.data/pilot-07/equivariance-study/`; resolved configs,
checkpoint provenance, sample identities and augmentation losses are recorded by
the Rust trainer. Augmentation depends on the absolute update index, preserving
exact resume. The run identity includes transform, objective and training sources.
Record command wall time, batch preparation, update latency, throughput, memory,
and device energy. Shared desktop activity contributes to board telemetry; it is
not process-attributed energy or a measurement of SM occupancy.

Nothing is committed, pushed, deployed or published by this protocol.
