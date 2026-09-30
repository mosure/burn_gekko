# burn_gekko

Sparse-view latent completion and multi-view representation learning in **Burn**.
Independently encoded V-JEPA 2.1 views feed a shared fusion decoder and task heads.
Training uses disk-cached procedural rooms from published **bevy_zeroverse 0.25.0 /
bevy_zeroverse_burn 0.8.0**. Runtime artifacts live in `.data/`.

This is a research prototype. **SOTA performance and sharp RGB reconstruction are
not established.** Latent prediction is primary; feature-color figures are not
RGB reconstructions. Training excludes noncommercial Gekko weights, including as
teachers. Camera metrics exist; the current latent checkpoint has no trained
camera head.

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

Metrics cover latent MSE/cosine/variance, co-visibility AP/AUROC, ETH3D/HPatches
AEPE/PCK, cluster intervals, camera rotation/translation-direction/focal error and
pose AUC. Future heads register checkpoint-bound records with units, counts,
aggregation and limitations. Missing results are not represented as zeros.

Report generation is local and never deploys a page. Read
[publication](docs/publication.md), [native evaluation](docs/native-evaluation.md),
[crate organization](docs/repository-plan.md) and the
[registered spatial-head experiment](docs/spatial-descriptor-protocol.md).

The latest [Pilot 08 study](docs/studies/pilot-08-equivariance.md) completes
6,000 updates over all 8,192 rooms and **passes all four registered spatial-readout
transfer gates**. HPatches viewpoint AEPE is 20.32 versus 25.47 for the same
checkpoint's encoder; ETH3D is 33.50 versus 36.74. Both improve over the trained
same-image control as well. A fresh 128-room cohort shows 5.23% lower latent MSE
with references, but oversmoothing, strict precision and unevaluated heads remain
limitations. Read the [single-run project page](.data/publications/pilot08-equivariance/index.html)
and [annotated 20-page PDF](.data/publications/pilot08-equivariance/paper.pdf).
This is a development milestone, not an established SOTA model. The study uses
71.96 of its 120 GPU-command minutes; all jobs have stopped.

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
