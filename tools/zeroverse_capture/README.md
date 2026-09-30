# burn_gekko_capture

An isolated Bevy rendering executable for `burn_gekko_data` capture recipes.
Uses published `bevy_zeroverse = 0.25.0` and `bevy_zeroverse_burn = 0.8.0`.
The library/training workspace does not link the renderer.

```sh
cargo install burn_gekko_capture --locked
gekko_zeroverse_capture --identity
gekko_zeroverse_capture --config configs/data/capture-preflight.toml \
  --output .data/capture-preflight --dry-run
```

Recipes are in [burn_gekko](https://github.com/mosure/burn_gekko/tree/main/configs/data).
`--dry-run` checks the resolved generator configuration without rendering. Actual
capture requires a graphics backend and the Linux display/graphics libraries used
by Bevy. For a content-addressed cache and capture receipt, use `gekko capture`
from `burn_gekko_train`, passing `--binary` with the installed executable path.
