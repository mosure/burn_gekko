# Pilot 07: bounded spatial descriptor study

Completed 2026-09-30. **The registered transfer gate fails.** References help
latent completion, but the learned spatial correction does not beat the same
checkpoint's strongest declared encoder control. This endpoint is retained as a
negative experiment, not promoted as a SOTA or RGB-quality result.

- [Single-run project page](../../.data/publications/pilot07-spatial-descriptor/index.html)
- [Annotated PDF](../../.data/publications/pilot07-spatial-descriptor/paper.pdf)
- [Publication manifest](../../configs/publish/pilot07-spatial-descriptor.toml)
- [Protocol registered before training](../spatial-descriptor-protocol.md)

## Model, data and selection

The new matching head preserves centered block-6 encoder features and adds
`0.25 * stop_gradient(norm(base)) / sqrt(768) * tanh(W * fused)`.
Its bias-free matrix starts at zero, exactly recovering the encoder readout.
The dense matching path is separate from sparse completion. The affinity teacher
is the own frozen parent encoder, with temperature 0.035 against the student's
0.07; these semantic pseudo-targets contain no geometric correspondence labels.

The fixed recipe completes 3,000 updates over 8,192 cached procedural rooms at
256×256, batch 16, two reference views, random 90% target masking and seed 719.
Both optimizers restart from the audited own parent; the new head starts at zero.
The encoder stays frozen throughout this phase. The fixed MIT V-JEPA 2.1 teacher
and commercial-compatible ancestry are retained; no noncommercial weights teach
or initialize this run.

Native coverage analysis finds 48,000 target exposures, all 8,192 training rooms
and all 24,576 room/view identities. This is 5.859 exposures per configured room,
or 1.953 cycles over its three target views. It is a bounded refinement phase,
not evidence of full model convergence or encoder pretraining from scratch.

The final checkpoint was frozen before external inference and the new capture:
`b87c11bfb8482118280a0081d80c2e9f4983860287b8a7a1480e8e0f61f41ba7`.
No checkpoint/readout/temperature was selected from these results. Validation
MSE decreases from 0.193884 to 0.192141 within the phase. Teacher and monitored
first/last encoder parameter deltas are zero, with no encoder gradients logged.

The new capture uses published bevy_zeroverse 0.25.0 and bevy_zeroverse_burn 0.8.0,
seed 2610040000, four cameras, baseline 0.5 and density 0.35. Its one train and one
validation room are unused; the 128 test rooms yield 512 target views. Dataset:
`9ce1ee777326018a0c88aebae9bbb141704b4009467b2e8a94b08bb7f4229c4d`.
All artifacts remain on disk under `.data/`.

## Fresh-room completion and information use

| Measurement | Result |
| --- | ---: |
| Cross-view hidden latent MSE | 0.203794 |
| Room-bootstrap MSE 95% interval | [0.200404, 0.207145] |
| Matched monocular MSE | 0.214045 |
| Paired reference MSE reduction | 0.010250 / 4.79% |
| Paired reduction 95% interval | [0.008616, 0.011937] |
| Unrelated-reference MSE | 0.258742 |
| Shuffled-reference-token MSE | 0.212977 |
| Training-position-mean MSE | 0.309260 |
| Cross-view cosine | 0.891745 |
| Prediction / teacher spatial variance | 0.382943 |
| Learned RI co-visibility AP / AUROC | 0.884248 / 0.696030 |

Co-visibility scoring has 117,701 known hidden patches, 91,289 positive. RI is an
error-improvement regression score, not a calibrated geometric probability.
Intervals resample 128 whole rooms with 10,000 deterministic draws. They do not
measure variability across training seeds. Shuffling already encoded reference
tokens is a layout-sensitivity diagnostic, not complete removal of positional
information inside features.

On the common first 32 rooms / 128 target views, MSE is **0.213161, 0.204278 and
0.201376** with one, two and three references. The native scorer checks checkpoint,
mask, dataset and target identities and verifies monocular isolation. The
two-reference number here differs from the full-cohort mean because its population
is restricted to that intersection.

Hidden-RGB perturbation and monocular reference permutation cause exactly zero
output change. Cross-view permutation has maximum difference 0.0036235 and RMS
0.0001612, failing the predeclared strict 1e-5 tolerance. Representation
oversmoothing remains: switching to a latent target did not solve RGB blur.

## Real-image transfer: declared primary readouts

All rows below belong to this one checkpoint. Inputs are RGB only; geometry is
loaded by the separate Rust scorer after inference. Both benchmarks are
development evidence, and the hard-patch readout is not published refinement parity.

| Conditional readout | ETH3D AEPE ↓ | ETH3D PCK3 ↑ | HPatches viewpoint AEPE ↓ | HPatches PCK3 ↑ |
| --- | ---: | ---: | ---: | ---: |
| Block-6 encoder control | 36.7477 px | 2.2871% | 25.4874 px | 11.7268% |
| Spatial residual head | 37.6953 px | 2.2615% | 25.5399 px | 11.4207% |
| Fusion decoder features | 43.0673 px | 2.1455% | 29.7094 px | 9.9749% |

ETH3D uses all 3,365 pairs, equal scene/interval weighting and a ten-scene paired
bootstrap. Its spatial-head AEPE gain is **−0.9476 px**, interval
**[−1.4913, −0.3971]**: a supported regression. PCK3 changes by −0.0256 percentage
points, interval [−0.0427, −0.0070].

HPatches exports all 580 pairs; the 295 viewpoint pairs / 59 sequences are
primary. Its AEPE gain is **−0.0525 px**, interval **[−0.3062, 0.2080]**. PCK3
changes by **−0.3061 percentage points**, interval **[−0.4350, −0.1811]**.
Positive gains favor the head. Neither benchmark passes the registered gate.

The bounded head is closer to its encoder than the raw fusion descriptor, but
that does not establish a useful learned fusion gain. Comparing only against
unconditioned encoder cosine would hide this failure. Sharpened semantic
pseudo-affinities do not supply verified new geometric information; the result
is consistent with that limitation, rather than proof of a unique causal failure.

## Execution and engineering qualification

Training takes 1,863.33 command seconds, including 158.80 seconds of preparation.
Warmed median/p95 update times are 0.46963 / 0.49604 seconds, or 34.07 target
examples/s at the median. Peak process VRAM is 21,140 MiB. The observed board
energy is 207.22 Wh, 15.54 J per logged target, with 99.94% telemetry coverage.
Mean board power is 400.57 W; median device activity is 93%. These device-wide
numbers include substantial desktop activity and do not quantify SM occupancy
or training-only energy. No power settings or other processes were changed.

The cumulative Pilot07 ledger ends at **42,634.7785 / 43,200 seconds**
(11.8430 / 12 hours), leaving 565.2215 seconds. It includes the failed 60-second
full-cache probe, the successful neutral/16-update checks, training, new capture,
and all GPU evaluations. CPU engineering, scoring and publication are outside
that declared command-wall ledger. No study GPU jobs remain active.

CPU contracts and a real-checkpoint CUDA probe establish exact zero-head feature
and match parity. Named-record tests cover legacy checkpoints and reject dropping
a trained optional head. The native training test reproduces the next updates,
both optimizer states and sample order after resume. Workspace tests, strict
workspace/CUDA Clippy and formatting checks pass.

Current training/assessment loaders verify and decode selected shards once;
full-cache audit remains explicit. Corrupt-shard/membership tests and exact resume
pass. The sealed v23 GPU binaries predate that loader repair, so these timings
do not qualify a startup speedup. See [GPU efficiency](../gpu-efficiency.md).

Publication verification checks 97 bundled files, 84 decoded images and 29 local
links. All six gallery selections work at 390, 768 and 1,440 pixels without page
overflow or JavaScript errors; reduced-motion handling passes. The 17-page PDF
has no overfull boxes or undefined references. Its first page and matching figures
were visually inspected. Receipts and screenshots are in
`.data/engineering/spatial-head/`. The paper now includes the primary V-JEPA 2.1
reference alongside the existing related-work links.

## Decision and next experiment

Do not promote this affinity-sharpening recipe as the quality fix. Retain the
encoder control and optional head implementation, keep the failed endpoint and
all predictions, and reserve new independent evidence after further development.
The project page and PDF show only this experiment, with explicit FAIL labels,
deterministic annotated examples, input provenance and camera/RGB/depth gaps.

The next accuracy study should change the supervision question rather than tune
the failed head on these benchmarks: test known image-transform equivariance as
an RGB-only correspondence objective, with a matched target-only adapter control.
Measure whether reference conditioning adds value beyond that adapter. Camera-
supervised or geometry-guided variants must be separately named and compared to
the self-supervised track; they are not implemented or qualified by this study.
Full-resolution matching/refinement parity and multiple seeds remain necessary
before a SOTA claim. No further training or hyperparameter selection was performed
after these gate failures.

Rebuild into a fresh output directory with:

```sh
target/pilot/gekko-report build \
  --experiment configs/publish/pilot07-spatial-descriptor.toml \
  --output .data/publications/spatial-descriptor-review --pdf
target/pilot/gekko-report validate --bundle .data/publications/spatial-descriptor-review
```

Nothing was committed, staged, pushed, deployed or published.
