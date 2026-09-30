# Initial training pipeline implementation

Historical record of the initial preflight. The subsequent package upgrade,
batched trainer, feature cache, and completed small GPU study are documented in
[pilot 01](studies/pilot-01.md). Versions, batch limits, and remaining gates below describe
the earlier implementation, not the current pilot.

Implemented and locally checked on 2026-09-27 (America/Chicago). This is a
bounded integration preflight; no competence, convergence, or paper result is claimed.

## Code organization

| Location | Implemented responsibility |
| --- | --- |
| `crates/burn_jepa` | Six verbatim encoder/core files from revision `939abcea4648fd2ad0e12cb6d7bf4874f6bdf871`, minimal manifest, strict sharded Burnpack adapter, native-image checkpoint round trip |
| `crates/gekko_data` | Bounded capture contract, process wrapper, content-addressed immutable cache, raw/zstd Safetensors reader, RGB-only training samples, separate geometry loader and audit |
| `tools/zeroverse_capture` | Separate locked Cargo workspace linking published `bevy_zeroverse=0.21.0` and `bevy_zeroverse_burn=0.4.0`; no dependency on the dirty source checkout |
| `src/encoder.rs` | Native independent image forwards, encoder normalization, deterministic sparse target masks, explicit inner-backend/frozen features |
| `src/model.rs` | Shared self/cross-attention decoder, separate MAE mask and RGB head, joint reference context, full-target raw RI head |
| `src/loss.rs` | Pixel-major RGB patch targets, unbiased per-patch normalization, released-code RI loss with epsilon 0.01 and stopped RGB targets |
| `src/train.rs` | AdamW, true global gradient clipping, finite/nonzero-gradient checks, JSONL metrics, held-out reconstruction validation, atomic model+optimizer checkpoints |
| `src/eval.rs` | Held-out raw-RI ranking against geometry, AP/AUROC with grouped ties, unknown-pixel exclusion |
| `tests/`, `.github/workflows/ci.yml` | CPU contracts and CI definition; no automatic GPU job or data generation |

The encoder import preserves upstream SPDX attribution and individual file hashes.
`tools/interop/verify_encoder_import.py` checks those hashes. The local Burnpack adapter
and checkpoint test adapt upstream `model_package.rs` and `numerical_parity.rs`
without pulling in viewer/RAC/AnyUp code. Canonical upstream root license notice
files were absent; see `crates/burn_jepa/UPSTREAM.md` before redistributing.

## Current contracts

Training is RGB-only. Depth, position, camera transforms, and the scene AABB are
available to evaluation through a separate type. Each camera is encoded as an
image, never as a timestep in a synthetic video. One random target mask is shared
between MAE and cross-view reconstruction and applied before encoder attention.
The dense target encoder output reaches only the RI branch. Features cross from
the non-autodiff encoder backend into decoder autodiff as constants.

The loss is the masked pixel mean of
`e_mae + e_cross + (stopgrad(e_mae - e_cross) - max(stopgrad(e_mae), 0.01) * RI)^2`,
where RGB errors average the three channels. RI is an unconstrained score, not
a calibrated visibility probability. Only the released-code clamp variant is
implemented; the alternative unclamped paper equation remains an ablation.

The decoder uses pre-norm attention and 2D sinusoidal position embeddings. Each
reference resets its spatial grid; a common reference-role embedding permits
joint attention without introducing a camera-order signal. There is no reference
view-index embedding, pose conditioning, dropout, or online feature cache. MAE
uses target self-attention in the shared cross-attention slot. The full-size
CroCo architecture and published Gekko checkpoint are not reproduced here.

The current batch size is one. Room and target-view selection are deterministic
functions of step; a ChaCha8 mask is a function of seed and step. Resume saves
decoder parameters **and AdamW state**, rejects mismatched identities, and
continues from the completed step. Current sampling is a deterministic diagnostic
cycle, not the eventual overlap/baseline curriculum. No AMP is enabled.

Generation uses static published camera settings, portable lighting, no humans,
one capture worker, one room per shard, and one timestep. It writes zstd-compressed
raw F32 RGB/depth/position plus metadata. A staging directory is renamed into the
cache only after all expected shards validate. Process failure and timeout leave
an incomplete staging directory; retry never silently promotes it. Binary hash,
schema, package identity, and config determine the cache key. Every cache open
checks shard hashes, dimensions, sRGB encoding, annotation precision, expected
seed sequence, and split membership. Only one room's arrays are loaded at a time.

## Local verification evidence

The recorded dataset is
`.data/datasets/b85c4bc17381421bca5acbf16d1dcad6a5b22738c3adb6a4ff811ae9d34f323f/`:
four independent rooms, seeds 20260927–20260930, two cameras, 64×64, split 2/1/1.
Its dataset files and receipt occupy less than 1 MB. This is a fixture-sized
capture, not the planned 32-scene qualification set.

The geometry audit covers 32,768 valid source pixels. Maximum self-reprojection
error is **0.0025396151 pixels**, maximum camera-Z depth error
**0.00000667572 m**. Directed cross-view counts are 4,792 visible, 2,152 occluded,
24,557 out of view, and 1,267 unknown. Position exports are AABB-normalized and
are decoded back to world coordinates before projection. Cameras are column-major
Bevy world-from-view with local forward -Z; pixel centers use `(x+0.5,y+0.5)`.
Visibility uses nearest-pixel depth with `max(0.02 m, 1% z)` tolerance. A point
in front of a different reference surface is unknown, not automatically negative.
These tolerances and independent-camera overlap need larger qualification.

The initial CPU diagnostic and CUDA pretrained preflight each performed **two
optimizer steps** with finite three-branch losses, finite nonzero global gradient
norms, changing decoder weights, held-out validation, and saved optimizer state.
The CUDA run used the local V-JEPA 2.1 Base Burnpack package: 12 checksum-verified
shards, 219,446,784 weight bytes, F16 storage converted to F32, on an NVIDIA RTX
PRO 6000 Blackwell workstation. The encoder identity is
`c408f68dd18a38824d0fa1d615e6f9f9f04f111d71a6f7c846f41dc187a8795f`.
Original Meta checkpoint provenance and independent image-forward parity have
not been established by this package-level verification.

The first CUDA step took 156.6 s, including cold compilation/autotuning, and the
second 2.03 s. These are diagnostic observations, not a throughput benchmark.
The initial CUDA report is `.data/runs/cuda-vjepa-base-preflight/report.json`;
the CPU report is `.data/runs/cpu-preflight/report.json`. Later verification
artifacts are listed in `.data/preflight/verification.json` when present.

All **33 CPU tests passed**, workspace formatting and strict Clippy passed, and
the isolated capture wrapper passed strict Clippy. Encoder import hashes match
the pinned revision. The real cache was reopened without restarting rendering.
Held-out validation evaluation completed for both the CPU diagnostic and CUDA
pretrained checkpoint; the reports are `evaluation-validation.json` in each run.
Neither run establishes useful co-visibility prediction after only two steps.

The CPU suite covers masked-pixel leakage, branch isolation, reconstruction/RI
head gradients, a scalar RI loss/gradient oracle, patch ordering and normalization,
reference permutation invariance, deterministic masks, RGB/geometry round trips,
cache reuse, corruption rejection, incomplete generation, over-budget rejection,
optimizer resume equivalence, checkpoint tensor round trips, duplicate shard
rejection, visibility states, and AP/AUC ties/degenerate cases. Imported encoder
tests also execute. CI is defined but has not been run remotely in this session.

## Remaining qualification gates

- Independent official-checkpoint native-image numerical parity, including F16
  conversion error; strict encoder-only exports and original checkpoint provenance.
- CPU/CUDA numerical comparison and CUDA resume equivalence. WGPU is feature
  wired but not runtime-qualified. Sparse patch kernels are imported, not exercised.
- Sparse reference masks, token-budget selectors, padded multi-example batching,
  view dropout, and measured memory/throughput studies.
- Larger geometry qualification, overlap-aware sampling, depth-boundary masking,
  render-quality variants, geometry sidecars, and real-data loaders.
- Tiny overfit, numerical stress/AMP tests, objective controls, multiple seeds,
  transfer evaluation, and all research/paper claims in the roadmap.
- Production-scale capture, dataset indexing/sharding, distributed training,
  checkpoint retention, interrupts/recovery orchestration, and experiment scheduling.

Current preflight limits are deliberate and enforced in config validation. The
broader roadmap is not complete, and no long GPU training or large generation
run was launched.
