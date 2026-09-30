# burn_gekko_train

Bounded training and Burn prediction export. `training::latent` is the primary
fixed-teacher pipeline; `training::reconstruction` retains RGB diagnostics. Other
trainers are historical research implementations. Configuration/ancestry and long
training loops have separate files. `data` owns batching and dataset adapters;
`evaluation` owns model loading, readouts and exports. Scalar scoring belongs to
`burn_gekko_eval`, publication to `burn_gekko_report`.

The `gekko` binary exposes `capture`, `verify-dataset`, `audit-geometry`,
`train-latent` and `assess-latent`, plus historical preflight commands. CUDA is
explicit (`--features cuda`, then `--backend cuda`); CPU is the default. Runtime
outputs stay in `.data/`. Run commands from the workspace root.

Historic binary/API names remain thin compatibility entry points. Exact optimizer
resume verifies source/config/backend and checkpoint hashes. Moving source does
not waive those checks: use the sealed old binary, or explicitly start a weights-only
phase with recorded ancestry and reset optimizers. Noncommercial pretrained
Gekko weights cannot initialize or supervise the primary latent trainer.
