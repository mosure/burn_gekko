# burn_gekko_data

Renderer-independent contracts for reproducible multi-view room datasets:
TOML capture recipes, content-addressed on-disk caches, SHA-256 verification,
chunk/RGB readers, camera geometry and image transforms.

```toml
[dependencies]
burn_gekko_data = "0.1.0"
```

This crate has no Burn or Bevy dependency. The optional rendering process is
provided by `burn_gekko_capture`; training and evaluation consume its disk cache.
See [the repository](https://github.com/mosure/burn_gekko) for capture recipes and
provenance rules. Runtime data belongs in `.data/`.
