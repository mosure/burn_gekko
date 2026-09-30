# Repository organization and encoder import plan

Status: future layout. This planning change creates documentation only. Source
paths below name either inspected upstream files or proposed destination files.

## 1. Workspace boundaries

Keep the model library usable without Bevy, a renderer, training dependencies, or
dataset access. Keep capture out of the trainer process. This lets the published
generator use its own dependency lockfile and prevents a graphics migration from
silently changing an ML experiment.

```text
burn_gekko/
  Cargo.toml                    workspace + root burn_gekko library
  Cargo.lock                    pinned ML/runtime dependency resolution
  rust-toolchain.toml            verified toolchain, initially compatible with 1.92+
  src/
    lib.rs
    config.rs
    encoder.rs                  narrow adapter over copied burn_jepa
    fusion/                     target decoder, set attention, positions, masks
    heads/                      reconstruction, RI; later probe heads
    losses/                     paired and set error-comparison objectives
    sparse/                     selection policies and budget accounting
  crates/
    burn_jepa/                  imported encoder slice; publish=false initially
      UPSTREAM.md               source revision, copied paths, deviations
      LICENSE-MIT
      LICENSE-APACHE
      src/...
      tests/...
    gekko_data/                 renderer-independent shard reader and sampler
    gekko_train/                trainer, optimizer, checkpoint, run ledger
    gekko_eval/                 geometry, matching, calibration, reports
  tools/
    zeroverse_capture/          separate Cargo workspace and lockfile
    checkpoint_convert/        isolated conversion/parity utilities
  configs/
    data/                       capture cohorts, split definitions
    model/                      pair/set, Base/diagnostic, encoder mode
    train/                      schedules and bounded run budgets
    eval/                       immutable protocols and metric settings
    experiments/                resolved experiment intentions and controls
  tests/
    fixtures/                   tiny licensed/generated analytic data
    contracts/                  geometry, schema, gradients, leakage
  benches/                      encoder, fusion, full-step, dataloader timings
  scripts/                      parity bridge and analysis/report generation
  docs/                         this design, ADRs, protocols, model/data cards
  paper/
    main.tex
    sections/
    references.bib
    figures/                    generated vector figures
    tables/                     generated tables with run IDs
    claims.csv                  claim -> evidence mapping
  manifests/                    small immutable split/source/run registries
  data/                         ignored mount/symlink; bulk artifacts elsewhere
  runs/                         ignored local run artifacts
```

`tools/zeroverse_capture` is excluded from the ML workspace and declares its own
`[workspace]`. It depends on the published crates by exact version. The data
crate consumes the frozen storage contract without importing Bevy. If initially
using the published reader is simplest, put that adapter in the capture tool and
export a stable training shard format; do not pull its renderer dependencies
into `burn_gekko` as a default.

Start with these boundaries as modules if a separate crate has no independent
build/dependency purpose. Avoid creating a crate per loss or attention block.
The copied encoder is a distinct crate because upstream synchronization and
feature selection require that boundary.

## 2. Exact encoder import scope

Source: `/home/mosure/repos/burn_jepa`, root package, proposed starting revision
`939abcea4648fd2ad0e12cb6d7bf4874f6bdf871`. Destination:
`crates/burn_jepa`, preserving its package identity and notices while documenting
the reduced API. Reverify the revision and file hashes when the copy happens.

| Source file or family | Initial disposition | Reason |
| --- | --- | --- |
| `src/config.rs` | Copy; preserve checkpoint configuration semantics | Model shape, modality, positional and normalization options |
| `src/model.rs` | Copy initially; expose encoder through a narrow adapter | Encoder, attention, patch projection, masks; avoid rewriting numerics while importing |
| `src/tokens.rs`, `src/positional.rs` | Copy | Original token IDs, grid shape, batched masks, RoPE support |
| `src/safetensors_io.rs` | Copy with strict encoder loading wrapper | Required tensor mappings and upstream checkpoint handling |
| `src/sparse_patchify.rs` | Copy, feature-gated | Reference sparse planning and later CUDA/WGPU kernels |
| `src/pipeline.rs` | Extract only required preprocessing helpers, recording changes | Whole pipeline has broader temporal/predictor concerns |
| `src/feature_memory.rs`, `src/sparse_feature_memory.rs` | Defer until a temporal-memory experiment | Stateless multi-view fusion needs no interframe cache |
| `src/model_package.rs` | Extract a minimal encoder manifest/load path later | Whole file couples multiple unrelated models and packaging systems |
| `tests/numerical_parity.rs` and associated fixtures | Import relevant cases and pin Python source | Preserve numerical evidence; add native-image coverage |
| Sparse patchify tests | Import when their backend feature is enabled | Validate an actual optimization, including gradients if trained |
| `src/lib.rs`, `Cargo.toml` | Author a reduced surface, with recorded upstream origin | Current whole-crate dependencies include unrelated workspace crates |
| Viewer, AnyUp, RAC/autocode, SC-TTT, DSA/GDN consumers | Exclude from initial import | No dependency in the chosen encoder/fusion contract |

The first import can retain the JEPA predictor types inside `model.rs` because
the existing loader builds `VJepa2_1Model`. Conversion may instantiate the full
upstream container, verify all required **encoder** tensors, then export its
encoder. Production fusion should load that encoder-only artifact. Predictor
keys may be explicitly irrelevant to an encoder-only package; missing encoder
keys may never be excused by a permissive global load option.

The source package declares `MIT OR Apache-2.0`, but root license-text files were
not present in the inspected revision; the located files belong to `bevy_burn`.
Resolve the encoder's canonical notices during import and populate the planned
destination license files from that provenance, not from unrelated crate notices.

Potential direct dependencies are Burn, burn-store, anyhow, serde/serde_json and
safetensors; conversion and GPU kernels add their actual dependencies. Retain
only the features reached by the imported modules. Initially pin the Burn 0.21
family consistently and add `burn_flex_gmm` only for a qualified sparse-patchify
lane. Do not copy dirty workspace feature forwarding into unrelated AnyUp crates.

## 3. Import and provenance procedure

1. Inventory source revision, dirty status, selected files, licenses, and feature
   dependencies. Copy from the immutable revision, not the ambient dirty tree.
2. Create `UPSTREAM.md` and a machine-readable source map containing source path,
   destination path, SHA-256, source revision, and local modifications.
3. First retain numerical behavior and test it; separate later API cleanup from
   numerical changes in reviewable commits.
4. Pin the official Python constructor revision and the original weight checksum.
   Specify Base/Large `ema_encoder` selection where applicable. A loader may not
   silently fall back to a different teacher or online encoder.
5. Compare required tensor names, shapes and dtypes; retain conversion reports.
   Fail on missing, unexpected-critical, nonfinite, or incompatible tensors.
6. Establish float32 parity for dense/native-image and selected-mask inputs,
   including non-square grids. Existing video micro-parity is useful but insufficient.
7. Export an encoder package containing model config, preprocessing, selected
   layer, tensor checksum, source notices, and conversion version.
8. Validate reload on the intended CPU/reference and GPU/runtime backends. Add
   lower precision only after establishing the float32 reference.

A model-loading fallback must be an explicit new configuration, never random
initialization disguised as a loaded checkpoint. An ordinary CI test that skips
external weights cannot satisfy the real-weight qualification gate.

Use the upstream local fixture's `5e-4` maximum absolute error as the initial
float32 parity threshold on identical deterministic inputs. Also report mean
absolute error, relative error, cosine similarity and per-layer discrepancies;
extend beyond micro grids to representative image resolutions and multiple masks.
Register separate lower-precision bounds from the float32 reference before
training. Do not relax a failing tolerance without diagnosing and recording the
source of the discrepancy.

## 4. Component APIs

Proposed responsibilities rather than committed Rust signatures:

| Component | Accepts | Returns / owns |
| --- | --- | --- |
| `VisionEncoder` | Preprocessed image, original-grid token selection | Features, token IDs, grid, weight/preprocess identity |
| `GekkoFusion` | Target features/query masks, reference groups | Fused features, RGB patches, RI maps |
| `RgbTrainingSample` | RGB plus view/sampling identity | No geometry-bearing fields or methods |
| `GeometryEvalSample` | RGB identity plus annotation sidecars | Calibration, valid depth, visibility, labels |
| `PairSetSampler` | Split manifest, run seed, step index, curriculum | Reproducible target, refs, augmentations and masks |
| `Trainer` | Fully resolved config and explicit stop budget | Checkpoints, optimizer/RNG state, metrics and event log |
| `Evaluator` | Immutable checkpoint, protocol and dataset IDs | Prediction shards, metric JSON, uncertainty and failure counts |
| `ReportBuilder` | Validated evaluation artifacts | Tables/plots carrying run and protocol IDs |

Feature boundaries: CPU reference, CUDA, WGPU, training/autodiff, conversion, and
optional sparse kernels. Initial inference has no filesystem or CLI requirement;
web support remains a later portability check rather than a training prerequisite.

## 5. Planned command surface

These commands do not exist yet; they specify the intended workflow.

```text
gekko-data validate <dataset-manifest>
gekko-data make-splits <split-spec>
gekko-data audit <dataset-manifest> --protocol <audit-spec>
gekko-convert encoder --spec <checkpoint-spec>
gekko-train run --config <experiment> --max-steps <ceiling>
gekko-train resume --run <run-id>
gekko-eval run --checkpoint <artifact> --protocol <protocol>
gekko-report build --study <study-manifest>
```

Keep configuration declarative and typed. Resolve defaults into a saved complete
config, reject unknown fields, and record CLI overrides. Validate incompatible
combinations before allocating GPU memory: for example, dense cached targets
with masked reconstruction, GT overlap filtering in the RGB-only track, or
temporal memory without a camera/time identity.

## 6. Reproducibility and artifact ownership

Every run directory should hold:

```text
run.json                   IDs, status, start/end, budget, hardware, source state
config.resolved.toml
environment.json           GPU/driver/backend/toolchain, dependency lock hash
sources.json               import, checkpoint and dataset hashes
metrics/train.jsonl
events.jsonl               resumes, failures, OOM, budget exits, config rejection
checkpoints/<step>/        weights + optimizer + schedule + sampler/RNG state
eval/<protocol>/           predictions, metric JSON, scene rows, bootstraps
artifacts.json             checksums and external storage locations
```

Use atomic checkpoint/shard publication and completion markers. Checkpoint
resume restores the next sample, mask, augmentation and optimizer update; it does
not merely load weights. Record determinism limitations of a backend explicitly.
Keep raw logs append-only. Derived tables are regenerable; never manually edit a
paper number after extraction.

Large datasets, pretrained weights and run checkpoints stay outside Git. Commit
small manifests, configs, analysis code and generated paper tables/figures whose
source artifacts can be retrieved. Avoid secrets or machine-specific absolute
paths in public manifests; retain configurable storage-root aliases.

## 7. Validation and CI design

| Lane | Required checks |
| --- | --- |
| Ordinary CPU CI | Formatting/lint, typed config validation, tiny tensor contracts, loss gradients, no-GT loader boundary, synthetic geometry fixtures, save/reload |
| Encoder qualification | Pinned real weights plus official Python parity; required before accepting a converted encoder |
| CUDA qualification | Forward/backward, optimizer update, mask leakage, sparse/dense selected-input parity, mixed precision, resume |
| WGPU qualification | Inference parity and sparse-kernel qualification where supported; no untested training claim |
| Capture qualification | Published package build, analytic/rendered labels, codec round trip, seed/split/resume/process tests |
| Study verification | Dataset and checkpoint hashes, equal-budget controls, metric aggregation, table provenance |

Autodiff for custom sparse kernels must be demonstrated before unfreezing through
them. Frozen inference can use a kernel whose backward is absent; a partially
trainable encoder needs a supported fallback or an independently checked backward.
Do not infer support from a successful forward benchmark.

Run only the checks appropriate to an actual code change. This documentation
change needs link, consistency, and whitespace checks rather than training tests.
