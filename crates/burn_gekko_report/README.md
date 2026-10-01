# burn_gekko_report

One verified experiment produces a self-contained project page and paper.

```sh
cargo run -p burn_gekko_report --bin gekko-report -- build \
  --experiment configs/publish/my-run.toml --output .data/publications/my-run --pdf
cargo run -p burn_gekko_report --bin gekko-report -- validate \
  --bundle .data/publications/my-run
```

`experiment` defines the singular run/checkpoint contract; `artifact` verifies
input binding and capability metrics; `figures` and `correspondence` make actual
prediction visualizations; `page`/`paper` share the resolved report; `validate`
checks output hashes, decoded images, local links and gallery assets. HTML/CSS/JS
is self-contained under `src/assets/`. PDF compilation uses `pdflatex` with
PGFPlots and without shell escape. There is no publishing or network operation.

`learning` produces training/validation CSVs, a page SVG and a vector PDF plot
from the same selected-phase points. Parent and post-checkpoint updates are excluded.

`latent_metrics` verifies feature SNR against every exported hidden-target array.
`display` shares plain-language labels and metric explanations between the page
and PDF. Feature SNR is not RGB PSNR. Reference benefit, spatial variation and
within-three-pixel matching accuracy are shown as percentages. Export all assessed
rooms when using the new signal/error fields; incomplete exports are rejected.

An active encoder-preservation objective binds the frozen anchor's weights and
metadata, verifies unchanged probes, and displays training feature drift as an
RMS percentage. This remains a regularization diagnostic, separate from held-out
completion, RGB error or correspondence accuracy.

See the [publication guide](../../docs/publication.md). Future heads add native
capability records, not new report scripts or private-model comparison panels.

## Actual-view validation

An optional `[view_geometry]` pinned artifact adds synthetic camera-view metrics
and annotated first-room examples to a single-checkpoint page/PDF. The plot shows
RGB inputs, renderer-projected target points, predicted points and pixel errors.
This evidence is labeled separately from image homographies and external transfer;
it does not imply a trained camera head or RGB reconstruction capability.
