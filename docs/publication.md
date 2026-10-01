# One experiment → page and paper

```sh
cargo build --profile pilot --locked -p burn_gekko_eval -p burn_gekko_report
target/pilot/gekko-report build --experiment configs/publish/pilot07-spatial-descriptor.toml \
  --output .data/publications/spatial-descriptor-review --pdf
target/pilot/gekko-report validate --bundle .data/publications/spatial-descriptor-review
```

`--pdf` uses `pdflatex` with PGFPlots (TeX Live's pictures package) and shell
escape disabled. Without it, the page links LaTeX.
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
2. Run `gekko assess-latent` with one model and `export_rooms = rooms` for complete
   array verification of the feature signal/error metrics.
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

## Reading the metrics

The page and PDF share a plain-language metric guide. Latent completion shows
feature signal/error in dB, reference benefit in percent, spatial variation
retained in percent, and the underlying MSE/cosine. Feature dB is calculated as
`10 log10(mean hidden teacher squared amplitude / hidden prediction MSE)` and then
averaged over target views. Rust rechecks all exported target arrays and their
identities before publication; a substituted score or incomplete export fails.
It is not RGB PSNR. RGB PSNR remains unavailable until a real RGB head is trained
and assessed with explicit range, color space and mask.

Correspondence tables show mean match error and the percentage within three
pixels. HPatches uses a 240-pixel scoring frame; ETH3D uses original image pixels.
These thresholds are not comparable across resolutions. Optional fractional
coordinates use the same displacement interpolation as the hard readout. The
fixed local readout uses a 3 by 3 probability centroid; it does not use labels.
Native paired contrasts distinguish the `transfer` gate (positive pixel-error
reduction interval, no mean precision regression) from the `local_precision`
gate (positive precision-gain interval, no mean pixel-error regression).

## Local artifacts and deployment

Building performs no commit, push, deploy or upload, regardless of repository visibility.
The authorized `project-page.yml` workflow publishes the committed `www/project`
bundle and builds the live demo on pushes to `main` or manual dispatch. It validates
the publication, compiles the viewer and inference worker for WASM, and retrieves
one SHA-256-pinned model release asset. The browser verifies every model part again.
Model files stay outside Git; licenses and checkpoint provenance accompany them.
The optional public Gekko baseline remains isolated from training and is not
included in the default page or demo. An optional `[demo]` table adds a link to both
page and PDF and must identify the experiment's foundation checkpoint.

## Actual camera-view diagnostics

The optional `[view_geometry]` table pins the `metrics.json` from
`view_geometry_export`. Reports keep this synthetic diagnostic separate from
image transforms, external matching and calibrated pose. They show the first
four validation rooms, eight evenly spaced visible queries per room, ground-truth
projections and RGB-model predictions. No examples are selected by quality. A
geometry-trained run also binds its training-label cache and provenance in the
report closure. Historical synthetic test rooms reused for development are
labeled `evaluation_use = "development"`; they are not presented as fresh tests.

## Attached calibration and RGB heads

The optional `[output_heads]` pinned file names one head-training phase attached
to the selected foundation checkpoint. The native reporter verifies both head
weight files, training records, feature-cache identity, split membership and all
raw validation predictions. It recomputes angular/focal and hidden-pixel RGB PSNR
scores, rejecting changed labels, incomplete populations or substituted weights.
The camera route uses dense RGB pairs; RGB completion uses sparse-target latents.
The page places these predictions before the feature gallery and the paper
contains deterministic RGB/camera panels. Numerical stability and camera accuracy
remain separate statuses. See `configs/publish/head-stability15.toml`.
