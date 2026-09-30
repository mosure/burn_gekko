# Bounded spatial descriptor experiment

Registered before GPU preflight or training. Continue the existing Pilot07
ledger: 39,905.4188 / 43,200 seconds consumed, 3,294.5812 seconds remaining.
All GPU checks, capture, training and inference count. No additional allowance,
noncommercial teacher, geometry supervision, commit or publication is authorized
by this protocol. CPU engineering and scoring are outside the command ledger.

## Hypothesis and implementation

The primary latent target and matching descriptors should have distinct heads.
Preserve the independently encoded block-6 descriptor and learn a bounded
correction from the pair-conditioned fusion representation. For centered
encoder token x and fusion token z, use

`descriptor = x + stop_gradient(norm(x)) * 0.25 / sqrt(D) * tanh(W z)`.

W has no bias and starts at zero. Initialization therefore exactly recovers the
centered encoder feature, with correction norm bounded by 25% of its norm. The
head is independent of sparse completion: dense target features never enter the
masked completion path. The existing six-block fusion trunk is shared.

The head learns bidirectional RGB-only soft affinity targets from the fixed
warm-start encoder's block 6. Teacher temperature 0.035 and student temperature
0.07 provide a sharpening objective, rather than a zero-gradient copy of the
identical starting encoder. These are semantic pseudo-targets, not geometric
correspondence labels. The risk is confidently learning incorrect pseudo-matches;
real-image evaluation determines whether the correction is useful.

## Fixed training recipe

- Warm-start checkpoint: `f0d79d2cb53c9c3cf4ddb870efb6167898db42bb5b67e982a56ffdd5f3fa0d82`.
- Both optimizers reset; the new head begins at zero. Record this as a new phase.
- 8,192 training rooms, 64 validation rooms, 256×256, two references, batch 16,
  random 90% masking, seed 719; retain the fixed MIT V-JEPA latent teacher.
- Freeze the entire encoder throughout. Train fusion, latent/RI and spatial heads.
- Learning rate 1e-4, 50-update warmup, 3,000-update cosine horizon, weight decay
  0.05. Descriptor weight 1.0; attention and dense-latent auxiliary weights zero.
  Existing completion/visible/RI weights are unchanged.
- Target exactly 3,000 updates; native wall stop 1,850 s and command ceiling
  2,000 s reserve finalization. A wall-limited endpoint must be labeled partial.

First run CPU contract tests, a real-checkpoint CUDA neutral-initialization probe,
and a separate 16-update diagnostic. Diagnostic weights never initialize the
main run. New optional named-record fields must preserve old checkpoint loading;
missing or unexpected trained heads must fail instead of silently disappearing.

Preflight amendment before training: the first 60-second neutral probe timed out
during the full 8,192-room cache audit, with no inference result. Its charged
receipt is retained. The retry uses the existing 130-room cache, one validation
room for the neutral probe, and its one train/one validation room for the separate
16-update numerical check. All used files still receive checksum verification.
This changes only diagnostic data size; the main 8,192-room recipe and selected
parent remain fixed, and diagnostic weights are discarded.

## Selection and evaluation

Select the final endpoint before evaluating external benchmarks. Primary readout
is `spatial_residual_conditional` at fixed temperature 0.07. Compare with the same
checkpoint's `student_l06_centered_conditional`, and retain unconditioned versions,
raw fusion decoder and attention readouts as diagnostics. No per-example choice,
post-hoc temperature sweep or mixture-weight tuning is allowed.

ETH3D and HPatches remain development benchmarks. Export all 3,365 ETH3D pairs and
all 580 HPatches pairs; the 295 viewpoint pairs are HPatches' primary population.
Use native scoring and paired scene/sequence uncertainty for the readout contrast.
The transfer gate requires reduced AEPE with a positive paired interval on both
benchmarks and no PCK3 regression. A gain only against the weaker final-layer
encoder or an unconditioned operator does not pass.

If time permits after checkpoint selection, capture a new disjoint 128-room
four-camera cohort. Otherwise label the prior cohort development. Latent MSE,
cosine, variance, information-use controls and RI AP/AUROC remain required. The
paper/page binds only to this one endpoint and must show both passed and failed
gates, camera-head status, resource use and deterministic annotated samples.
This small single-seed experiment cannot by itself qualify a SOTA foundation model.
