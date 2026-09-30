# Implemented crate organization

```text
src/                           burn_gekko model library
  encoder.rs                   normalization, sparse features, contiguous uploads
  fusion/                      attention kernels, set decoder, 2D RoPE
  models/                      latent, RGB and explicitly historical compositions
  heads/                       appearance, matching and transport
  objectives/                  tensor losses and fixed-teacher affinity guidance
  sparse/                      masks and RGB-only curriculum
  tensor.rs                    checked diagnostic readback
crates/
  burn_vjepa/                   audited imported encoder; preserve upstream source map
  burn_gekko_data/              cache, chunks, geometry and TOML
  burn_gekko_train/src/
    training/                  trainers, schedules, checkpoints and ancestry
    data/                      dataset-to-tensor adapters and batching
    evaluation/                Burn inference, readouts and immutable exports
    bin/                       compatibility and benchmark entry points
  burn_gekko_eval/src/          scalar metrics, protocols and frozen-export scoring
  burn_gekko_report/src/        experiment binding, figures, HTML and paper
configs/                       data, train, experiments, eval, publish, archive
tools/                         isolated capture, interop, process monitor, legacy
docs/                          current guides, studies, archived original plans
.data/                         ignored datasets, models, runs and publication bundles
```

`burn_gekko_train` depends on the model/data/evaluation crates. `burn_gekko` does not
depend on its trainer or dataset reader. `burn_gekko_report` consumes immutable native
records without initializing Burn or a GPU. Bevy stays in the separate capture
process/lockfile. Do not create a new crate for each head or metric.

Canonical APIs include `burn_gekko::fusion::decoder`,
`burn_gekko_train::training::latent` and `burn_gekko_eval::camera`. Historical model/trainer
aliases remain to limit unnecessary API churn. Model record field names and
tensor layouts are retained. Long attention/configuration/training concerns have
separate files instead of sharing a flat root.

Inference exports remain with runtime loading because they audit model config and
ancestry. Scalar scoring, uncertainty and publication are independently buildable.
The original organization plan and README are under `docs/archive/`.

Update source-identity inputs when moving numerical code; never weaken resume
checks. Old exact resumes require their sealed binary. New weights-only phases
record optimizer resets. Historical artifact/config bytes are not rewritten.
