# Configuration layout

All user-authored inputs are TOML. Machine-produced metrics remain JSON/JSONL.

- `data/`: procedural capture recipes.
- `train/`: small preflight and encoder recipes.
- `experiments/`: current bounded training/evaluation plans.
- `eval/`: native single-checkpoint scoring protocols.
- `publish/`: one experiment per page/paper bundle.
- `archive/pilot-XX/`: byte-preserved historical recipes, not current defaults.

Run commands from the workspace root. Historical recipes may contain their
original paths; the sealed source archives are authoritative for replay.
