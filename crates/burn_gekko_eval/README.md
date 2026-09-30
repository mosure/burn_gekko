# burn_gekko_eval

Native CPU scoring of immutable Burn prediction exports. No Python or GPU backend.

- `metrics`: masked completion, PSNR and correspondence errors.
- `ranking`: tie-aware AP/AUROC.
- `camera`: SO(3), signed translation direction, normalized intrinsics and pose AUC.
- `benchmark`: complete ETH3D/HPatches protocols and deterministic visual examples.
- `statistics`: cluster bootstrap with a specified portable RNG.
- `efficiency`: observed board energy and update timing, with coverage/shared-load limits.
- `training`: exposure coverage, scalar windows and verified encoder gradient stages.
- `training_export`: checkpoint/command-bound training summaries.
- `process_activity`: shared-GPU process counters with unavailable values preserved.
- `schema`: extensible checkpoint-bound capability records consumed by publication.

`gekko-eval score --config <TOML>` scores one checkpoint. `gekko-eval camera
--config <TOML>` scores post-inference `CameraRecord` JSONL with pinned record and
provenance hashes. Camera records must declare their coordinate frame and input
contract; they are not an input API for the training model.

`gekko-eval training --config <TOML>` summarizes a completed training command.
`gekko-eval activity --config <TOML>` scores an immutable NVIDIA pmon log. Both
record their source hashes and keep optimization evidence separate from accuracy.

See [metric/protocol definitions](../../docs/native-evaluation.md). Add numeric
oracles and leakage/provenance contracts here when introducing new heads.
