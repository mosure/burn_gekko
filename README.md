# burn_gekko

Sparse-view latent completion and multi-view representation learning in **Burn**.
Independently encoded V-JEPA 2.1 views feed a shared fusion decoder and task heads.
Training uses disk-cached procedural rooms from published **bevy_zeroverse 0.25.0 /
bevy_zeroverse_burn 0.8.0**. Runtime artifacts live in `.data/`.

This is a research prototype. **SOTA performance and sharp RGB reconstruction are
not established.** Latent prediction is primary; feature-color figures are not
RGB reconstructions. Training excludes noncommercial Gekko weights, including as
teachers. Separate camera-calibration and RGB reconstruction heads can be fitted
from scratch on cached foundation features. Numerical stability and prediction
accuracy have separate qualification gates.

## Workspace

| Crate | Responsibility |
| --- | --- |
| `burn_gekko` | Model library: encoders, fusion, sparse inputs, heads and tensor objectives |
| [`burn_vjepa`](crates/burn_vjepa) | Audited imported V-JEPA encoder and provenance/license notices |
| [`burn_gekko_data`](crates/burn_gekko_data) | TOML, immutable disk cache, RGB/geometry readers and capture contracts |
| [`burn_gekko_train`](crates/burn_gekko_train) | Trainers, optimizers, checkpoint ancestry, Burn assessment and prediction exports |
| [`burn_gekko_eval`](crates/burn_gekko_eval) | Native metrics, camera protocols, benchmark scoring, uncertainty and efficiency |
| [`burn_gekko_report`](crates/burn_gekko_report) | One-experiment project page, annotated figures, LaTeX and PDF |

The model library has no CLI, renderer or dataset-reader dependency. Scoring and
reporting need neither a GPU nor Python. Capture keeps its [isolated Bevy
workspace](tools/zeroverse_capture). Historical analysis is under `tools/legacy`;
remaining import/parity and process-monitor bridges are described in
[tools/README.md](tools/README.md).

## Install

All crates start at version **0.1.0**, using Rust 1.98 and Burn 0.21.

```toml
[dependencies]
burn_gekko = "0.1.0"
burn_vjepa = "0.1.0"
```

```sh
cargo install burn_gekko_train --locked --bin gekko
cargo install burn_gekko_eval --locked --bin gekko-eval
cargo install burn_gekko_report --locked --bin gekko-report
cargo install burn_gekko_capture --locked
```

The renderer has a separate Bevy build. Add `--features cuda` when installing the
trainer for CUDA. Read the [release notes](CHANGELOG.md) and
[release procedure](docs/releasing.md) for validation and source-identity changes.

## Main workflow

Run from the workspace root. User inputs are TOML; machine outputs are JSON/JSONL
or typed tensors. Every training command has an explicit ceiling.

```sh
cargo test --workspace --locked
cargo build --profile pilot --locked -p burn_gekko_eval -p burn_gekko_report
cargo build -p burn_gekko_train --profile pilot --features cuda --locked --bin gekko

# Cache/reuse a small procedural dataset.
cargo build --manifest-path tools/zeroverse_capture/Cargo.toml --locked
target/pilot/gekko capture --config configs/data/capture-preflight.toml

# Primary training; use a fresh output path and an explicitly budgeted recipe.
target/pilot/gekko train-latent --config configs/experiments/pilot07-native-spatial-refinement.toml \
  --run .data/runs/my-experiment --backend cuda

# Assessment/scoring/publication manifests name one selected checkpoint.
target/pilot/gekko assess-latent --config configs/experiments/my-assessment.toml \
  --output .data/evaluations/my-experiment --backend cuda
target/pilot/gekko-eval score --config configs/eval/my-experiment.toml
# Optional training-only summary, with command-bound power telemetry.
target/pilot/gekko-eval training --config configs/eval/my-training-summary.toml
target/pilot/gekko-report build --experiment configs/publish/my-experiment.toml \
  --output .data/publications/my-experiment --pdf
```

The `my-*` files are manifests you create for the chosen run; see the
[publication guide](docs/publication.md) and [configuration layout](configs/README.md).
The historic `latent_pilot`, `latent_assess` and export binaries remain adapters
inside `burn_gekko_train`.

## Evaluation and publication

The generator accepts **one run and one checkpoint**, verifies artifact hashes,
recomputes displayed completion metrics from float arrays, shares one teacher-fitted
color projection, and retains explicit capability gaps. It does not compare two
private model versions. Monocular/reference controls and readouts of the same
checkpoint are within-experiment diagnostics. The public Gekko baseline remains
isolated from training and is not in the default publication generator.

Metrics include feature signal/error in dB, reference benefit and spatial variation
as percentages, latent MSE/cosine, and co-visibility AP/AUROC. Correspondence shows
mean pixel error and the percentage of matches within three pixels, with cluster
intervals. Camera metrics include rotation/translation-direction/focal error and
pose AUC. Feature dB is not RGB PSNR; RGB PSNR needs an evaluated RGB decoder.
The page and paper explain each metric's scale and direction. Future heads
register checkpoint-bound records with units, counts,
aggregation and limitations. Missing results are not represented as zeros.

Report generation is local and never deploys a page. Read
[publication](docs/publication.md), [native evaluation](docs/native-evaluation.md),
[crate organization](docs/repository-plan.md) and the
[registered spatial-head experiment](docs/spatial-descriptor-protocol.md).

The latest [output-head study](docs/studies/head-stability-15.md) trains camera and
RGB heads for 800 CPU updates on a fixed foundation checkpoint. All bounded
stability gates pass, including exact checkpoint replay and finite gradients.
On 8 development rooms / 24 targets, hidden-pixel RGB PSNR is **21.09 dB**
(**20.70 dB** with references disabled). Camera rotation error is **11.64 degrees**,
translation-direction error **32.68 degrees**, and focal relative error **44.35%**.
The camera head overfits this small dataset; stable optimization does not establish
reliable calibration. The [project page](www/project/index.html) and
[PDF](www/project/paper.pdf) include actual RGB predictions and camera diagrams.

The preceding [Pilot 14 diagnostics](docs/studies/pilot-14-information-diagnostics.md)
reuse the fixed Pilot 13 checkpoint. A third reference reduces latent error by
**1.66% on 512 identical masked targets**, with a positive paired room interval.
Native analysis shows that missing spatial structure, rather than a simple
amplitude mismatch, dominates the remaining smoothing: teacher-variance
rescaling worsens MSE by about 21%. Camera gates are sensitive to solver seeds;
raising the RANSAC budget improves mean AUC@10 from **7.99% to 9.42%**, but both
control gates each pass 2 of 8 seeds; only one seed passes both together. These are input/solver diagnostics,
not new training improvements. The [page](.data/publications/pilot14-information-diagnostics-reviewed/index.html)
and [32-page PDF](.data/publications/pilot14-information-diagnostics-reviewed/paper.pdf)
show one checkpoint and its own controls. All 157 workspace tests passed at that
closeout. After the new head study, the previous GPU allowance has **2.88 minutes
remaining**; longer training awaits a new ceiling. SOTA and resolution of the
smoothing issue remain unestablished.

The completed [Pilot 13 study](docs/studies/pilot-13-view-geometry.md) adds
renderer-supervised correspondence across actual camera views, using geometry
only for training labels. Its matched 384-update experiment over 2,048 rooms
reduces viewpoint error by **26.9%** while retaining completion within the
registered 1% limit. External mean matching error falls **12.4% on HPatches
viewpoint / 16.3% on ETH3D** versus the matched control. All declared spatial
matching contrasts pass, including the previously failing coarse ETH3D gate.
The selected local readout reaches **26.23% / 5.68% within-three-pixel accuracy**
in those benchmarks' different scoring frames. Camera transfer remains mixed:
one per-sequence gate fails and overall AUC@10 regresses despite higher recall.

The single-run [project page](.data/publications/pilot13-geometry-reviewed/index.html)
and [30-page PDF](.data/publications/pilot13-geometry-reviewed/paper.pdf) include
annotated completion, matching, camera metrics, learning curves and limitations.
On 512 targets from 128 newly generated test rooms, references reduce latent
error by **7.18%**, but only **43.56%** of teacher spatial variation remains.
Feature signal/error is **7.38 dB**, not RGB PSNR. Strict numerical reference-order
invariance still fails; RGB, camera and depth heads remain untrained. No
noncommercial weights enter training. Neither SOTA nor an RGB blur fix is
established. All 152 workspace tests pass. The added decoder work costs 12.6%
more observed board energy per target; shared-load limitations and a failed
supplementary process monitor remain documented. Combined GPU-command usage is
**11.900 / 12 authorized hours**, with **6.02 minutes unused** and no running
model jobs. Older studies below retain their original budget snapshots.

The preceding [Pilot 12 study](docs/studies/pilot-12-feature-preservation.md)
adds a training-only frozen feature anchor. Its matched 2,048-update screen
retains completion MSE while reducing known-transform error from **7.44 to
4.78 pixels**. A fixed 4,096-update continuation over all 8,192 rooms then passes
the refined readout's matching and calibrated-motion transfer gates against its
own equally processed controls. The coarse ETH3D readout still fails its
encoder-control precision requirement. The refined readout reaches **24.00% / 5.15% within-three-pixel
accuracy** on HPatches viewpoint / ETH3D, in their distinct scoring frames.
TUM pose recall within 10 degrees is **19.48%**; this uses a geometric solver,
not a trained camera head. Absolute accuracy remains limited.

The single-run [project page](.data/publications/pilot12-main-reviewed/index.html)
and [30-page PDF](.data/publications/pilot12-main-reviewed/paper.pdf) include
learning curves, annotated predictions, uncertainty, units and retained failures.
On 512 targets from 128 newly captured rooms, references reduce latent error by
**7.47%**, but only **42.52%** of teacher spatial variation remains. Strict
numerical reference-order invariance still fails. No noncommercial weights enter
training; RGB, camera and depth heads remain untrained. SOTA and an RGB blur fix
are not established. All GPU jobs have finished: Pilot 12 uses 3.817 hours from
the prior remainder, bringing combined usage to **11.172 of 12 authorized hours**
and leaving **49.71 minutes unused**. The study retains its numerical amendments,
failed optimized export and corrected capture-wrapper receipt.

The completed [Pilot 11 main phase](docs/studies/pilot-11-full-adaptation.md)
trains for 12,000 updates over all 8,192 rooms. References reduce error by
**6.42% across 512 fresh target views**. Matching passes all registered transfer
gates: 20.78% of HPatches viewpoint matches and 4.37% of ETH3D matches fall within
three pixels of truth, using their distinct scoring scales. Camera motion is
still weak: 14.61% recall within 10 degrees at 256px, and 25.97% at 512px;
both resolutions fail their per-sequence fusion gates. The single-run
[page](.data/publications/pilot11-main-reviewed/index.html) and
[29-page PDF](.data/publications/pilot11-main-reviewed/paper.pdf) include learning
curves, annotated predictions, readable units and retained failures. Spatial
variation remains only 43.30% of the teacher's, and strict numerical reference
permutation fails. The matched 12,000-update full-encoder recovery improves
synthetic transform matching but leaves completion error 4.99% higher, failing
its 1% retention limit. The tail recipe is retained. The completed
[batch-capacity probe](docs/studies/pilot-11-batch-probe.md) finds only 14.64--16.33%
more target throughput at batch 32 with 1.965 times the process VRAM, below its
20% threshold; batch 16 stays unchanged. Pilot 11 closed at
**7.355 of the authorized 12 additional GPU-command hours**; Pilot 12 uses its
4.645-hour remainder as described above. No accuracy or energy-saving claim
follows from the performance probe.

The preceding [Pilot 10 qualification](docs/studies/pilot-10-real-pose.md)
first evaluated the frozen model on 186 real-image TUM pairs. Its immutable
[page](.data/publications/pilot10-real-pose/index.html) and
[PDF](.data/publications/pilot10-real-pose/paper.pdf) remain historical evidence.
Reuse of this camera cohort is now development. Pose estimates use known
intrinsics and a native geometric solver; the camera head remains untrained.

The preceding [Pilot 09 study](docs/studies/pilot-09-local-readout.md) completes
2,700 continuation updates over all 8,192 rooms and **passes both registered local
precision gates and all matching fusion-transfer gates**. Within the selected
checkpoint, fixed local refinement raises matches within three pixels from
13.26% to **19.69% on HPatches** and 2.33% to **4.19% on ETH3D**, while reducing
mean pixel error. The fusion descriptor also beats equally refined encoder and
trained same-image controls. These benchmarks remain development data; their
pixel coordinate scales differ.

A fresh 128-room / 512-target cohort has **7.30 dB feature signal/error** and
**5.99% lower latent MSE with references**, but retains only 42.27% of teacher
spatial variation. Fine correspondence, strict reference-order invariance and
untrained RGB/camera/depth heads remain limitations. Read the
[single-run project page](.data/publications/pilot09-local-readout/index.html)
and [annotated 23-page PDF](.data/publications/pilot09-local-readout/paper.pdf).
The study uses **39.10 GPU-command minutes** from Pilot 08's remaining allowance;
combined usage at that closeout was 111.05 of 120 minutes. Its jobs completed. This is a
development milestone, not an established SOTA model or an RGB blur fix.

The preceding [Pilot 08 study](docs/studies/pilot-08-equivariance.md) established
the hard spatial readout's four transfer gates after 6,000 updates. Its
[page](.data/publications/pilot08-equivariance/index.html) and
[PDF](.data/publications/pilot08-equivariance/paper.pdf) retain that single-run
evidence and the unchanged original compute ledger.

The preceding [known-transform preflight](docs/studies/pilot-07-equivariance-preflight.md)
completes 256 updates over 128 rooms, with a new RGB-only geometric descriptor
objective and a trained same-image conditioning control. HPatches viewpoint AEPE
improves by 9.17% relative to the same checkpoint's encoder. ETH3D shows a small
conditioning benefit, but its encoder comparison remains uncertain; the combined
fusion-transfer gate still fails. The [project page](.data/publications/pilot07-equivariance-preflight/index.html)
and [annotated PDF](.data/publications/pilot07-equivariance-preflight/paper.pdf)
include full benchmarks, known-warp examples and native efficiency measurements.

The preceding [spatial-head study](docs/studies/pilot-07-spatial-descriptor.md)
includes the [local project page](.data/publications/pilot07-spatial-descriptor/index.html)
and [annotated PDF](.data/publications/pilot07-spatial-descriptor/paper.pdf).
Its 3,000-update phase covers all 8,192 training rooms. References reduce held-out
latent MSE by 4.79%, but the bounded spatial head **fails** its registered ETH3D
and HPatches transfer gates against the same checkpoint's encoder control.
This is a retained negative experiment, not the accepted quality fix. Native
paired intervals, exposure coverage and explicit capability gaps are included.
The [earlier native refinement study](docs/studies/pilot-07-native-spatial-refinement.md)
remains available. Generate each bundle from its own experiment manifest.

## Reproducibility

Completed run artifacts and sealed source/binary archives remain immutable under
`.data/`. Historical recipes are byte-preserved in `configs/archive/pilot-XX/`;
reports are in `docs/studies/`. The [path map](docs/archive-paths.json) records moves.
Source moves change exact-resume identity: use the original sealed binary for an
old optimizer resume, or an audited weights-only phase with explicit optimizer reset.

Reorganization reproduced eight real-checkpoint target views and all 80 raw sample
arrays bit-for-bit. Native benchmark scoring tolerances are recorded in
[native evaluation](docs/native-evaluation.md). [SOTA qualification](docs/sota-evidence.md)
remains open; the [fusion-transfer study](docs/studies/pilot-07-fusion-transfer.md)
and [GPU investigation](docs/gpu-efficiency.md) retain the earlier evidence.
