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
is self-contained under `src/assets/`. PDF compilation uses `pdflatex` without
shell escape. There is no publishing or network operation.

See the [publication guide](../../docs/publication.md). Future heads add native
capability records, not new report scripts or private-model comparison panels.
