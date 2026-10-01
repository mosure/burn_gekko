# burn_gekko_data

Renderer-independent contracts for reproducible multi-view room datasets:
TOML capture recipes, content-addressed on-disk caches, SHA-256 verification,
chunk/RGB readers, camera geometry and image transforms.

`real_views` defines strict external RGB-pair manifests and separate camera labels.
`tum` prepares fixed Freiburg 3 cohorts, timestamp associations, original-image
provenance and resized RGB caches. Preparation uses no model outputs to choose pairs.
Input resolution is explicit (default 256px); camera intrinsics stay in original
image pixels, with half-pixel coordinate conversion shared by scoring and reporting.

```toml
[dependencies]
burn_gekko_data = "0.1.0"
```

This crate has no Burn or Bevy dependency. The optional rendering process is
provided by `burn_gekko_capture`; training and evaluation consume its disk cache.
See [the repository](https://github.com/mosure/burn_gekko) for capture recipes and
provenance rules. Runtime data belongs in `.data/`.

## Renderer correspondence targets

`view_targets` prepares compact, immutable CPU caches from verified room geometry.
Each directed view pair stores patch-center projections and separate visible,
occluded, out-of-view and unknown states. Depth discontinuities and points outside
the descriptor-center hull do not receive labels. The cache binds the dataset,
room splits, shard checksums, labeling source and target bytes.

Prepare one with `gekko-eval prepare-view-targets --config configs/data/targets-pilot13.toml`.
These are explicitly renderer-supervised training/evaluation labels. They are
separate from the RGB-only model input API; inference does not need the cache.
