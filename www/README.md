# Project page and paper

`project/` contains the latest verified single-experiment bundle, including
annotated RGB predictions, camera diagrams, metrics, source identities and PDF.
Generate it with the native Rust reporter from
`configs/publish/head-stability15.toml`. Open `project/index.html` locally or
serve this folder using any static server.

The deployment workflow remains disabled; committing the reviewed bundle does
not enable GitHub Pages. CI validates the committed output hashes, images and
local links without starting training or requiring local `.data/` inputs.

See [the publication guide](../docs/publication.md).
