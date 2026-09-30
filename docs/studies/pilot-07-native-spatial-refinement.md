# Native spatial refinement and publication study

The fixed 3,000-update experiment is complete. It learns useful reference-dependent
latent completion, but **does not pass the fusion-transfer quality gate**. It does
not establish SOTA or sharp RGB reconstruction. The repository now produces the
page, annotated evaluations and paper through native Rust crates from one selected
checkpoint, without a private old/new model comparison.

- [Project page](../../.data/publications/pilot07-native-spatial-refinement/index.html)
- [15-page PDF](../../.data/publications/pilot07-native-spatial-refinement/paper.pdf)
- [Resolved measurements and provenance](../../.data/publications/pilot07-native-spatial-refinement/results.json)
- [Publication manifest](../../configs/publish/pilot07-native-spatial-refinement.toml)
- [Protocol registered before training](../native-refinement-protocol.md)

## Training and data

The selected endpoint is `f0d79d2cb53c9c3cf4ddb870efb6167898db42bb5b67e982a56ffdd5f3fa0d82`.
This weights-only continuation reset both optimizers explicitly and retained its
audited MIT V-JEPA / own-model ancestry. No public Gekko weights or predictions
entered training. The fixed teacher and early encoder blocks remained unchanged.

The phase completed 3,000 updates / 48,000 target exposures on 8,192 cached rooms,
256×256 inputs, two references, batch 16 and 90% random target masking. The encoder
was frozen for 2,000 updates; the validation gate enabled the final two blocks for
1,000 updates. No full unfreeze occurred. Smaller updates and preserved early
features did not suffice to close the downstream matching gap.

After freezing selection, published `bevy_zeroverse 0.25.0` and
`bevy_zeroverse_burn 0.8.0` captured a new 128-room test cohort with four cameras and
a wider camera baseline. Its seed interval is 2610030002–2610030129. The dataset
cache is `c3bdc80be000d76cbed85ac95f7e1a5f736e6ed068a66df5a56ed69fe127ebd6`.
The cache contains one unused train and one unused validation room in addition to
the 128 test rooms. Existing ETH3D and HPatches results had already informed prior
research decisions, so both benchmarks are explicitly **development evidence**.

## Completion and information use

All full-cohort values use the same 512 masked target views and one checkpoint.

| Measurement | Value |
| --- | ---: |
| Cross-view hidden-token MSE | 0.204602 |
| Room-bootstrap 95% interval | [0.201542, 0.207816] |
| References disabled MSE | 0.215685 |
| Reference positions shuffled MSE | 0.213380 |
| Unrelated reference room MSE | 0.257334 |
| Training-set position mean MSE | 0.309469 |
| Hidden-token cosine | 0.891288 |
| Prediction / teacher spatial variance | 0.382860 |
| Full-target RI co-visibility AUROC / AP | 0.686712 / 0.881668 |

References reduce aggregate MSE by 5.14%. The paired room gain is 0.011083 with
95% interval [0.009211, 0.012861]. Intervals resample rooms, not correlated pixels,
and do not measure uncertainty across training seeds. RI uses a separate
full-target branch; its AP has 78.1% positive prevalence and is not sparse inference.

On exactly the same first 32 rooms / 128 targets, one, two and three references
give MSE **0.210573, 0.204395 and 0.201018**. The monocular control is identical
across these runs within 1e-7. Gekko-style fusion here is therefore not restricted
to two views. This measures reference-count sensitivity of one model, not separate
trained versions.

Hidden-RGB isolation is exact. Reference permutation changes the float32 output
by up to 0.003803 (RMS 0.000171), failing the strict 1e-5 invariance threshold. Low
feature variance and visible smoothing remain. The page uses a common teacher-fit
PCA, deterministic sample selection and signed reference-benefit maps so harmful
reference effects are visible. Feature colors are not reconstructed RGB.

## Real-image transfer

These are alternative readouts of this checkpoint. All 3,365 ETH3D pairs and all
580 HPatches pairs were exported; HPatches' primary viewpoint subset has 295 pairs.
The native scorer preserves each benchmark's coordinate and aggregation protocol.

| Readout | ETH3D AEPE ↓ | HPatches viewpoint AEPE ↓ |
| --- | ---: | ---: |
| Fusion decoder features | 39.2341 | 27.5824 |
| Fusion decoder, conditional matching | 38.0780 | 26.4834 |
| Reciprocal fusion attention | 40.0379 | 25.6878 |
| Own encoder block 6, centered | 38.0565 | 26.5389 |
| Own encoder block 6, conditional matching | **36.7477** | **25.4874** |

Conditional fusion PCK3 is 2.2536% on ETH3D and 11.0036% on HPatches viewpoint;
the matched block-6 conditional control reaches 2.2871% and 11.7268%. Thus improved
synthetic completion does not establish preserved real correspondence. Published
refinement/SOTA results are not directly comparable to this local hard-patch
readout. The report includes cluster intervals and deterministic annotated real
image pairs, with geometric labels loaded only after inference.

## Efficiency and budget

Training command time was **1,936.19 seconds / 32.27 minutes**. Steady median
updates were 0.4673 seconds frozen and 0.5359 seconds partially unfrozen: 34.24 and
29.86 targets/second, respectively. The corresponding p95 times were 0.4922 and
0.5663 seconds. Whole-command throughput, including setup, validation and saving,
was 24.79 target exposures/second.

Observed board energy was 214.586 Wh / 16.094 J per target with 99.95% telemetry
coverage. Observed average board power was 399.2 W and median device activity 93%.
Peak process VRAM was 22.86 GiB. Shared desktop activity remains included in board
power/utilization; these figures do not establish process-only energy or SM
occupancy. No power settings or unrelated processes were changed.

Compatibility checking, this training phase, fresh capture and all GPU evaluation
used **44.41 minutes** of the approved remaining allowance. The cumulative shared
Pilot07 ledger is **39,905.42 / 43,200 seconds** (11.085 hours), with **54.91 minutes
unspent** and no training/evaluation process left running. No new allowance was
created. Further unchanged-objective training was not scheduled after the failed
transfer gate. CPU engineering, scoring and report generation are outside this
GPU-command ledger.

## Native implementation and verification

`burn_gekko` contains models; `gekko_train` owns training and inference;
`gekko_data` owns capture/cache contracts; `gekko_eval` owns CPU scoring;
`gekko_report` owns publication. The imported encoder remains separate. Historical
Python analysis and configs are archived away from active commands; the bounded
process monitor and weight-interoperability bridges remain explicit exceptions.

Verification covered the workspace tests, strict workspace and CUDA Clippy,
encoder import integrity, exact real-checkpoint reorganization parity on eight
views / 80 arrays, and complete native benchmark scoring parity within recorded
floating-point tolerances. Publication tests reject mixed checkpoints, altered
evidence, private model-comparison lists and mislabeled populations. Camera metric
tests cover rotations, signed translation, baseline eligibility and pose AUC;
camera scoring is implemented, but no camera head is claimed trained.

The final bundle validates 97 hashed files, 84 decoded images and 29 local links.
All six gallery selections worked at widths 390, 768 and 1440 without JavaScript
errors or horizontal overflow; reduced-motion behavior also passed. The PDF has
15 pages, no overfull boxes, and visually inspected text/sample layouts. These
are local checks; no remote CI, publication, commit or push was performed.

## Next controlled research gates

1. Separate a spatial descriptor head from the final-layer completion target.
   Use a gated residual over the preserved block-6 descriptor, initialized to
   reproduce that encoder control. Verify equality before training; evaluate
   whether learned fusion adds value beyond the preserved route.
2. Test relative-camera prediction and camera-aware fusion in an explicitly
   separate supervision arm. Synthetic intrinsics/poses are guidance during
   training; inference inputs, camera gauge, pose losses and geometry usage must
   be declared. The existing camera evaluator can score the resulting head.
   Image-grid RoPE alone is not calibrated 3D geometry.
3. Gate further scale on both completion benefit and preserved/improved
   correspondence under matched readout operators. Register a new independent
   real-image holdout and multiple training seeds before a SOTA claim. Neither
   the present development benchmarks nor the just-inspected synthetic cohort
   may be reused as untouched confirmation data.

These are prospective experiments, not implemented or qualified capabilities.
