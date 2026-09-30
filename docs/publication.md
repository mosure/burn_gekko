# One experiment → page and paper

```sh
cargo build --profile pilot --locked -p burn_gekko_eval -p burn_gekko_report
target/pilot/gekko-report build --experiment configs/publish/pilot07-spatial-descriptor.toml \
  --output .data/publications/spatial-descriptor-review --pdf
target/pilot/gekko-report validate --bundle .data/publications/spatial-descriptor-review
```

`--pdf` uses `pdflatex` with shell escape disabled. Without it, the page links LaTeX.
Serve the output with any static server or open `index.html` locally. No external
JS, fonts or services are required.

The TOML manifest has singular `run`, `checkpoint`, `checkpoint_sha256` and
`latent` fields. Benchmark and future-head files must bind to that checkpoint.
Optional efficiency evidence binds telemetry to its exact training command.
An optional `[equivariance]` pinned file points to a native known-transform
`metrics.json` from that checkpoint. The generator checks its summary against
per-room records and adds geometric metrics plus annotated original/warped RGB
pairs. Image hashes are checked before rendering. These figures are explicitly
labeled as augmentation diagnostics rather than real viewpoint tests.
An old/new `models` list is rejected, as is a second checkpoint injected through
head or benchmark evidence.

The bundle contains HTML/CSS/JS, annotated PNGs, the training SVG, metric summaries,
input hashes, exact sample selection, a shared teacher-fitted latent projection,
LaTeX, optional PDF, the experiment manifest and output checksums. Page and PDF
share one Rust report object. Camera/RGB/future-head gaps remain visible.
Within-checkpoint paired readout intervals and explicit PASS/FAIL gate labels
are generated from native scores. Training coverage uses logged sample identities,
not only configured room counts. Negative experiments remain negative in the
page and PDF; they are not relabeled as accepted models.

The visual/evidence approach follows
[bevy_zeroverse](https://mosure.github.io/bevy_zeroverse/project/): real matched
samples, declared populations, a report and downloadable provenance. Templates
here are newly authored and self-contained.

## A new run

1. Freeze its selected checkpoint and classify cohorts as development or held out.
2. Run `gekko assess-latent` with one model and `export_rooms > 0`.
3. Export benchmark predictions in Burn; run `gekko-eval score` for native metrics
   and deterministic annotated examples.
4. Pin these artifacts in a publication TOML, state architecture/limitations, and
   build into a fresh directory.
5. Validate links, mobile/desktop gallery controls, PDF layout and bundle hashes.

Future head artifacts use `schema = 1`, the same `checkpoint_sha256` and one
validated `capability` object. Camera heads should include angular/focal errors,
pose AUC and baseline coverage. Different inputs, such as full-target camera
estimation, must be distinguished from sparse completion. No new Python report
script is needed for a head.

## Publication is pending

The repository is private. Building performs no commit, push, deploy or upload.
The workflow template ends in `.disabled`. After user approval, copy the reviewed
bundle to `www/project/`, review/enable the template, check distribution rights and
provenance, and complete the normal release process. The optional public Gekko
baseline remains isolated from training and is not included in the default page.
