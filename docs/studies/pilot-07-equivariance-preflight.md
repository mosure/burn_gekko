# Pilot 07: known-transform correspondence preflight

Completed 2026-09-30. **The new objective improves local matching; the combined
fusion-transfer gate is still not passed.** This is a completed diagnostic,
not an accepted state-of-the-art foundation model or a resolution of RGB blur.

- [Private project page](../../.data/publications/pilot07-equivariance-preflight/index.html)
- [Annotated 17-page PDF](../../.data/publications/pilot07-equivariance-preflight/paper.pdf)
- [Single-experiment publication TOML](../../configs/publish/pilot07-equivariance-preflight.toml)
- [Registered protocol](../equivariance-protocol.md) and
  [ETH3D execution amendment](pilot-07-equivariance-eth3d-amendment.md)

## What changed

Known projective image transforms replace semantic-affinity sharpening as the
descriptor auxiliary objective. Bilinear patch-grid labels supervise bidirectional
cosine NLL; cropped queries are excluded. Augmentation is deterministic by absolute
update index and replays across optimizer checkpoints. No transform, renderer
geometry or camera label is supplied to inference. Primary latent completion and
RI remain active, and dense augmentation cannot enter the sparse completion path.

The same trunk/head is trained both with the other image as context and with only
its own image as context. This supplies a trained control for cross-image
conditioning. Matching uses centered block-6 descriptors plus a correction bounded
by 25% of their norm. The fresh correction starts at zero. The audited parent is
the native-spatial-refinement checkpoint, not the failed sharpening checkpoint.
Both optimizers restart; all pretrained ancestry is audited MIT V-JEPA 2.1 or
this project's own checkpoints.

The phase completes **256 updates**, batch 16, **128 rooms / 384 distinct target
room-view pairs**, 4,096 logged target exposures, 256x256 RGB, two references,
90% random masking, seed 733. The encoder stays frozen, with measured first/last
encoder and fixed-teacher parameter deltas exactly zero. Final checkpoint:

`996fedf858d6c5e1fe0cdaabd8539e87ff3bd7e6041625ea2981b41056be5e73`

## Matching results from one checkpoint

Conditional readouts use the same fixed temperature, 0.07. All external datasets
are development evidence; none of these hard-patch protocols establishes parity
with published full-resolution refinement pipelines.

| Readout | Known-transform AEPE | HPatches viewpoint AEPE | HPatches PCK3 | ETH3D AEPE | ETH3D PCK3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Centered block-6 encoder | 13.7957 px | 25.4874 px | 11.7268% | 36.7477 px | 2.2871% |
| Pair-conditioned descriptor | 12.2629 px | 23.1493 px | 12.0376% | 36.2859 px | 2.2881% |
| Same-image-conditioned descriptor | 12.2724 px | 23.1508 px | 12.0356% | 36.4323 px | 2.2838% |

Known transforms cover 16 validation rooms in both directions. Pair conditioning
reduces mean error by **11.11%** relative to the encoder, with NLL 2.8894 versus
3.1795. The same-image control is nearly identical. This tests augmentation
learning, not real 3D viewpoint transfer.

HPatches includes all 580 pairs; the 295 viewpoint pairs / 59 sequences are
primary. Relative to the encoder, the head gains **2.3382 px AEPE**, paired 95%
interval **[1.7795, 2.8874]**, and **0.3108 percentage points PCK3**, interval
[0.1554, 0.4634]. The encoder comparison passes the registered readout gate.
Relative to the same-image control, AEPE gain is only **0.0016 px**, interval
**[-0.0538, 0.0581]**. A cross-image conditioning gain is not established here.

ETH3D includes all **3,365 pairs**, with equal scene/interval weighting. Relative
to the encoder, AEPE gain is **0.4618 px**, interval **[-0.4519, 1.2612]**: uncertain.
Relative to the same-image control, gain is **0.1463 px**, interval
**[0.1046, 0.1988]**, with PCK3 gain **0.00436 percentage points**, interval
[0.00246, 0.00651]. This supports a small conditioning benefit on this development
protocol, but does not clear the encoder comparison or the combined gate.
Intervals resample whole scenes/sequences 10,000 times; training-seed uncertainty
is not measured. No readout or checkpoint was selected per example.

## Latent completion and convergence

On 16 validation rooms / 48 target views, cross-view MSE is **0.189829** versus
**0.212278** with references disabled: **10.58%** lower. Paired room-bootstrap
gain is 0.022449, interval [0.019872, 0.025228]. This cohort is reused development
data; its first eight rooms supplied training probes.

Latent cosine is 0.89954; spatial variance is only **42.87%** of the teacher,
so oversmoothing remains. RI AP is 0.92085 and AUROC 0.70202, over 11,040 known
hidden patches, 9,300 positive. These depend on the cohort and its prevalence.
Hidden-RGB perturbation gives exactly zero prediction change. Reference-order
permutation has maximum difference 0.003796 and RMS 0.0001685, failing the strict
1e-5 tolerance. Camera, depth and RGB reconstruction heads remain untrained.

First versus last 32-update training means: pair NLL **3.2402 to 2.9497**,
same-image NLL **3.2404 to 2.9491**, cross-view MSE **0.19249 to 0.18354**.
These changing minibatches show optimization, not independent generalization.
This short phase does not establish convergence of a large-data study.

## Efficiency and engineering qualification

Training command: **205.48 seconds**, including 24.92 seconds of preparation and
in-command evaluation. Warm update median **0.5235 seconds**, p95 **0.5647**;
about **30.56 targets/second**. Augmentation averages 29 ms in the first window
and 35 ms in the last. Peak training-process VRAM is **24,372 MiB**. Observed
board energy is **18.11 Wh**, mean board power **318.94 W**, telemetry coverage
99.49%, median device utilization 90%. Shared desktop processes contribute;
these are not process-specific energy or SM-occupancy measurements.

The focused ETH3D exporter avoids repeated encoder passes and unused readouts.
Its first eight pairs match the broad exporter exactly in indices and mutual
flags on CUDA. Warm p95 pair time is 47.76 ms; the complete export finishes in
**156.23 seconds**, within the registered 210-second internal ceiling. This
qualifies the focused path for these three readouts and this backend, not every
export mode. Quality scoring stays in Rust on CPU.

CPU tests cover transform/image consistency, fractional labels, masked-row loss,
detached targets, descriptor gradients, feature-only forward/gradient parity,
exact training resume, scoring and report provenance. The 119-test workspace
suite and subsequent affected-module tests pass; CUDA and workspace Clippy are
clean. Page/PDF use native scoring and rendering, with pinned hashes and explicit
capability gaps. No permanent Python analysis program was added.

Runtime evidence is under `.data/pilot-07/equivariance-study/`, including command
ledgers, telemetry, selected inputs, predictions, scores and sealed v24/v25 source
and binaries. This work uses the existing 12-hour allowance. The final ledger and
remaining allowance are recorded in `.data/pilot-07/budget.json`: 43,112.81 of
43,200 command seconds consumed, **87.19 seconds remaining**, no active GPU job.

## Prepared continuation, not launched

[`pilot08-equivariance-proposed.toml`](../../configs/experiments/pilot08-equivariance-proposed.toml)
prepares a 6,000-update phase over all 8,192 rooms, retaining both conditioning
modes and the fixed teacher. It restarts from the audited native parent and
allows tail-encoder unfreezing only after the existing validation-stability gate
and 2,000 updates. This is a proposal for a new allowance, not an authorized
extension of Pilot 07. It has not been GPU-qualified or run.

A two-hour study would first qualify tail-stage throughput, then reserve time
for complete ETH3D/HPatches and a newly generated room cohort. The configured
86-minute training ceiling leaves evaluation/finalization reserve; a wall stop
before 6,000 updates must be labeled incomplete. Retain the same-image control:
simply lowering local descriptor loss cannot establish multi-view foundation-model
quality. The combined gate and independent evidence are required before promotion.

No staging, commit, push, publication or deployment was performed.

Subsequent work: the separately budgeted
[Pilot 08 continuation](pilot-08-equivariance.md) has since completed 6,000 updates
and full evaluation. This preflight and its original ledger remain unchanged.
