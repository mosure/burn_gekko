# Tool boundaries

New metrics, evaluation and publication belong in `burn_gekko_eval` and `burn_gekko_report`.
Do not add another study-specific Python report here.

| Directory | Purpose |
| --- | --- |
| `zeroverse_capture/` | Isolated Rust/Bevy capture process and lockfile; published generator dependencies |
| `demo/` | Static WASM packaging and headed Playwright browser verification; inference and scoring stay in Rust |
| `interop/` | Audited encoder import/parity and optional public Gekko baseline bridges |
| `study/` | Existing bounded process runner, shared budget ledger and GPU telemetry |
| `legacy/` | Historical analysis/report scripts retained for artifact reproducibility; not the active reporting API |

Legacy recipes are byte-preserved in `configs/archive/`. Their original working
paths are available in the sealed study source archives under `.data/`; use those
archives to reproduce an old command verbatim. `docs/archive-paths.json` maps the
old checkout layout to the current one. Never rewrite a completed run's config,
source identity, checkpoint or evaluation artifact during cleanup.
