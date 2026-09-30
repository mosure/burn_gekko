# Procedural indoor dataset and data pipeline

Status: proposed recipes and contracts, using published Zeroverse packages. The
training process consumes saved data; it does not own a live Bevy engine.

## 1. Published generator boundary

Pin `bevy_zeroverse =0.25.0` and `bevy_zeroverse_burn =0.8.0` in an isolated
capture environment. Save the exact dependency lockfile, registry archive
checksums, binary hash, driver/backend, and exported capture-engine identity.
See [the artifact audit](source-audit.md) for verified package hashes.
Pilot 06's live registry receipts are under `.data/pilot-06/registry-versions.json`.
Its active 8,192/128/128-room capture and 256px recipe are described in
[pilot 06](studies/pilot-06.md); the larger recipe tables below remain planning options.

Use `procedural-indoor`, preserving its seeded room grammar. The inspected source
supports conference, open-office, lounge, training, and mixed layouts, furnishing
and human density, room materials/lighting, camera calibration, semantic output,
and scene manifests. Keep the generator's original manifests as provenance.

Start with static rooms and disabled humans, then include static people as a
declared diversity cohort. Motion is a later extension requiring synchronized
capture and dynamic-geometry validation. Keep GI quality fixed inside a cohort;
changes in rendering quality are versioned domain factors.

The published camera policy includes primary-room confinement, trajectory
length controls, and a connected `multiview` rig in 0.22.0/0.5.0. The original
0.21.0/0.4.0 planning audit lacked that option. The bounded pilot enables it and
audits rendered overlap for all directed pairs. Neither its proxy overlap
threshold nor independent same-room cameras guarantee a pixel overlap floor.
See [pilot 01](studies/pilot-01.md) for measured distributions and the new `position=2`
unclamped affine position contract.

## 2. Qualification capture recipe

The following is a **future invocation of the published binary**, after building
and verifying it in the isolated environment. It has not been run for this plan.
Use a fresh output path for every incompatible capture contract.

```sh
zeroverse_gen \
  --scene-type procedural-indoor \
  --output data/qualification/static_v1 \
  --samples 32 --workers 1 --chunk-size 2 --seed 100000 \
  --indoor-layout mixed --indoor-density 0.65 --indoor-human-density 0 \
  --indoor-quality auto --indoor-gi-rays 256 \
  --indoor-camera '{"primary_room":true,"path_length_min":0,"path_length_max":0,"long_path_fraction":0}' \
  --width 320 --height 240 --cameras 4 \
  --playback-steps 1 \
  --render-modes color depth normal semantic position \
  --color-codec raw --compression zstd --no-ui
```

Use a second small cohort with odd image dimensions and multiple trajectory
steps to validate indexing and resizing. Use a third with short trajectories,
for example path length 0.25–2 m, nine progress samples with increment 0.125,
and two cameras. Those frames provide strided RGB pairs within a camera track
even when independent cameras have little overlap. Validate the proposed
settings on the registry build before adopting them.

`playback-step` is normalized trajectory progress, not seconds or an assumed
video frame interval. Preserve both camera and progress IDs. Different times are
valid multi-view observations of a static scene; moving-scene experiments require
matching physical time or an explicitly temporal correspondence task.

Keep one generator worker initially. Increase only after profiling memory and
throughput; the CLI's larger defaults are unsuitable assumptions for a single
GPU sharing a workstation. Measure training efficiency after capture has stopped.
Capacity diagnostics may overlap capture if their timing is labeled as shared
GPU work and both full command durations are accounted against the study cap.

## 3. Dataset sizes and capture cohorts

These are proposed ceilings to cost before generation, not mandatory purchases
of compute/storage or claims that the data already exist.

| Tier | Proposed independent scenes | Captures and purpose |
| --- | --- | --- |
| Analytic/qualification | 32 primary scenes plus small edge-case cohorts | Full RGB/geometry; reprojection, codecs, process identity and previews |
| Model diagnostic | 64 scenes, separate from held-out sets | Tiny overfit and debugging; never headline evaluation |
| Pilot | 512 train, 64 validation, 64 sealed test | Four views at one time plus a separately costed short-trajectory cohort |
| Initial study | Up to 10,000 train, 1,000 validation, 1,000 test | Four simultaneous views per scene; additional short trajectories for a fixed 20% of scene families; 384-square model input |
| OOD suite | Initially 256 independent scenes per selected factor | Controlled material, layout-composition, camera and lighting changes |
| Scale extension | Only after P4/P5 gates | More unique scenes, views or temporal observations, one axis at a time |

Choose the trajectory subset by a fixed scene-ID hash before inspecting labels;
capture two cameras at nine progress samples for those families. Keep all of a
family's static and trajectory observations in the same split. The S0 sampler can
oversample these trajectory pairs without claiming more independent scenes.
Track family-level exposure counts to expose that concentration. Cost the extra
18 observations per selected family separately and shrink the cohort if the
registered capture/storage ceiling requires it.

The initial study ceiling does not force all training scenes to carry five
annotation planes. Retain all modalities for qualification and evaluation plus
a deterministic training audit subset, initially 5%. Large-scale RGB-only loss
training can store RGB for the remaining train scenes. Geometry-assisted
curation requires a separately costed geometry-bearing capture cohort.

Model input resolution and capture resolution are distinct. Start with 384-square
capture for direct token-grid qualification; then add controlled rectangular
captures with a documented letterbox/crop transform. Never stretch images or
modify intrinsics implicitly.

## 4. Source data semantics and canonical schema

Published sample views are time-major. Chunk axes include sample, time, camera,
height, width and channels as appropriate; inspect field-specific dimensions
rather than assuming every tensor has six axes or the same channel count.

| Source quantity | Required interpretation |
| --- | --- |
| Color | Tone-mapped sRGB with its storage range and codec recorded; lossless raw is authoritative |
| Depth | Linear camera-axis depth in metres, not Euclidean ray distance |
| `world_from_view` | Column-major camera-to-world transform; right-handed Bevy camera faces `-Z` |
| `fovy` | Vertical field of view in radians, with image aspect ratio |
| Near/far | Camera clipping planes in metres |
| Position | World position normalized by exported scene AABB; invert using that AABB |
| Normal | View-space normal encoded as `(n+1)/2`; decode before using |
| Semantic | Exact class palette data; convert to IDs with the recorded palette |
| Flow, if enabled later | Forward temporal displacement plus validity/visibility; not an arbitrary cross-camera map |
| Annotation precision | Render attachment precision, separately from the stored tensor dtype |

All conventions require confirmation in the rendered qualification set, including
background/alpha validity semantics. A zero-depth or invalid attachment pixel is
not a negative co-visibility example. Lossy JPEG previews are never geometric or
semantic ground truth.

Proposed canonical manifest fields:

```text
dataset_id, schema_version, generator_version, capture_engine_identity
source_archives_sha256, binary_sha256, cargo_lock_sha256, renderer_environment
scene_family_id, scene_id, root_seed, manifest_sha256, cohort_id, split
view_id, camera_id, progress_id, trajectory_progress, optional_physical_time
original_hw, stored_hw, color_encoding, annotation_precision
world_from_view, fovy, near, far, derived_intrinsics, coordinate_convention
modalities -> {shard, tensor_key, dtype, shape, checksum}
rgb_transform_id, pixel_transform, transformed_intrinsics
annotation_protocol_id, validity_policy, source_generation_config_sha256
```

Do not treat pseudocode field names as existing exports. The adapter creates
stable IDs and supplements source metadata without rewriting its meaning.
Preserve calibration in float64 during CPU label derivation where practical.

Keep geometry sidecars out of the RGB training sample type. Store pair/set
identities independently of geometry labels. Pair-index manifests must reveal
whether they were generated from RGB/time IDs or filtered with annotations.

## 5. Geometric co-visibility labels

Create directional labels `g_(t<-r)(p)` on the target image grid: does the first
surface observed at target pixel `p` also appear in reference `r`? This is not
semantic class overlap or intersection of camera frusta.

Use an explicit conversion to one canonical camera convention. One convenient
choice is OpenCV axes (`+X` right, `+Y` down, `+Z` forward) with
`C = diag(1,-1,-1,1)` converting Bevy camera coordinates. If `T_wb` is exported,
then `T_wc = T_wb * C` maps canonical camera coordinates into world space.

With square pixels and the exported centered perspective camera, derive
`fy = H / (2*tan(fovy/2))` and `fx = fy`; store the principal point and the
pixel-center convention explicitly. For continuous pixel centers `(u+0.5,v+0.5)`,
the centered principal point is `(W/2,H/2)`. Equivalent integer-center formulas
need the corresponding half-pixel shift. Verify with known projected points.

For each valid target pixel:

```text
X_t = z_t * inverse(K_t) * [u+0.5, v+0.5, 1]
X_r = inverse(T_wc,r) * T_wc,t * [X_t, 1]
q_r = project(K_r, X_r)
```

Require positive depth and the reference clipping/frustum conditions. Compare
projected depth with the rendered reference depth using a documented surface
sampling rule and tolerance. A provisional starting tolerance is
`tau(z) = 0.01 m + 0.002*z`; calibrate it on analytic fixtures before freezing it.
Use nearest valid reference depth initially to avoid interpolating across object
boundaries; report sensitivity to tolerance, boundary exclusion, and sampling.

Use at least three label states:

- **Visible:** reprojection is supported and its depth agrees within tolerance.
- **Non-visible:** a valid target surface lies outside the reference view or
  behind a nearer valid reference surface.
- **Unknown/invalid:** no valid target surface, missing reference depth,
  inconsistent foreground mismatch, or ambiguous boundary/material behavior.

Exclude unknown pixels from metrics and record their count. Behind-camera and
out-of-bounds projections can be known non-visible; do not confuse that with the
different validity policy of a temporal-flow tensor. Derive both directions
independently. They generally have different masks and overlap fractions.

For set labels, visible in any valid reference is a positive. A negative needs
non-visibility in every reference; unresolved references leave an unknown unless
another reference already supplies a positive. Never label unknown as negative
to make set metrics easier.

Also export valid correspondences, occlusion reason, baseline, triangulation
angle, per-view overlap and pair IDs. Measure camera-center separation and
surface-dependent parallax separately. A near-zero-baseline pair can have high
overlap but weak depth information.

Depth/position are two independent paths for validating reprojection. Check
normal transforms and AABB inversion on analytic fixtures. Glass/transmission and
specular RGB may disagree with the first annotation surface: retain dedicated
material/boundary strata and document the opaque-annotation policy instead of
claiming exact photometric correspondence there.

## 6. Pair/set sampling and curricula

Maintain explicit experimental regimes:

| ID | Loss inputs | Pair/set selection | Permitted description |
| --- | --- | --- | --- |
| S0 | RGB only | Camera/time IDs, fixed stride and random same-scene selection | RGB-only self-supervised fusion training |
| S1 | RGB only | Depth/pose-derived overlap or parallax strata | Self-supervised losses with geometry-assisted curation |
| S2 | RGB plus geometric targets | Declared geometric supervision | Supervised auxiliary or downstream experiment |

Synthetic scene creation itself uses geometry. Even S0 should not be described
as an entirely geometry-free end-to-end pipeline; its learning signals and pair
selection avoid annotation lookup.

For S0, begin with a fixed mixture of short same-camera strides and simultaneous
different-camera pairs, initially 75%/25% in the pilot. After a fixed warmup,
increase stride and the fraction of different-camera/reference-set examples.
Freeze the schedule before confirmation; do not adjust individual pairs using
their GT overlap. Log temporal and cross-camera performance separately.

For S1, use five overlap bins `[0,.05), [.05,.25), [.25,.5), [.5,.75), [.75,1]`,
based on the minimum directional visible fraction. Report the two directional
fractions as well. Include low/zero-overlap cases; choose a registered mixture
instead of keeping only easy pairs. Keep rejected examples and reason counts in
the curation ledger. No silent seed resampling or threshold relaxation.

For sets, sample one target and one to three unique references. Include
complementary, redundant, low-overlap and mixed sets. Do not enforce pairwise
overlap between every reference. In S1, set selection may use geometry and must
remain labeled accordingly. In S0, content diversity and connectivity are
measured after sampling, not enforced by hidden label access.

## 7. Splits and domain diversity

Split at the independent underlying scene-family level before enumerating views,
time steps, pairs, relighting variants or masks. A stable root identity groups
all variants of one geometry seed, including camera resampling and appearance
counterfactuals. Keep those variants in the same split. Hash canonical geometry
manifests to detect accidental duplicates across seed/config namespaces.

Use disjoint seed namespaces and freeze explicit scene lists. Select complete
lists without inspecting model outcomes. Validation serves tuning and includes a
separate calibration subset; test remains sealed until a specification is locked.
Bootstrap statistics over scenes, not over millions of correlated pixels.

The main in-distribution study can include all five layout families with recorded
proportions. Separate OOD suites vary material/lighting ranges, room/furniture
composition, density, FOV, baseline, viewpoint height/roll, and corruption.
A leave-one-layout-family-out study is a separate training split/run, not an
OOD label attached to a family already used in training.

Keep geometry-matched appearance counterfactuals as a diagnostic when their
geometry was seen in training; label them accordingly. Domain-shift claims
require independent held-out geometry. Report planned object counts separately
from objects actually visible in the selected images.

## 8. Storage, throughput, and restartability

Archive original lossless chunks and metadata. For training, create indexed shards
with fast random sample access; choose compression after a decode-throughput
pilot. Do not recompress the entire corpus into an assumed optimal format first.
Use bounded reader queues and deterministic shuffling, with checksums, schema
versions and explicit corruption failures.

Conservative capture-memory estimate:

```text
bytes per scene = T * V * H * W * sum(bytes per captured plane per pixel)
```

Five RGBA32 planes, `T=1`, `V=4`, `H=W=384` require 45 MiB per scene before
compression and metadata. Twelve thousand such scenes would be about 527 GiB.
At `T=3` that becomes about 1.54 TiB. RGB alone at the same RGBA32 capture width
is 9 MiB per four-view scene. These figures cover the static cohort only; the
additional trajectory captures, OOD cohorts and analysis sidecars add to them.
Actual stored sizes depend on exported channels and compression; measure them
and do not promise compression ratios.

Small chunk sizes control the renderer/writer overlap peak; account for two
resident uncompressed batches plus compression workspace, render assets and GPU
buffers. Process recycling may bound worker lifetime, but does not itself prove
the absence of memory growth. Track RSS/VRAM per scene and across restarts.

Generate to immutable shard identities with atomic completion. Resume verifies
capture identity, source version, seed ordering, image size, modalities and
cohort configuration. An incompatible renderer creates a new dataset version.
Record failed seeds; repairs retain provenance rather than quietly changing the
population. Data workers must preserve uneven final shards without duplicating
or silently dropping them.

## 9. Data acceptance gates

1. Analytic planes/occluders and known camera transforms recover expected
   projection, depth, visibility, asymmetry and out-of-frustum labels.
2. Actual rendered scenes agree under depth and world-position reprojection,
   with numerical thresholds justified before setting a dataset pass rate.
3. All modalities share scene/camera/progress identity; calibration survives
   safetensors/NPZ or canonical-shard round trips.
4. Odd resolutions, image transforms, empty humans, terminal time, invalid depth,
   glass, thin geometry, and non-square FOV cases have explicit outcomes.
5. Single/multiple worker schedules preserve scene identities and metadata;
   deterministic pixel equality is qualified only for the tested environment.
6. Histograms cover layouts, density, materials, visible classes, overlap,
   baseline, parallax, invalid pixels, luminance and texture. Store sampled RGB
   montages with fixed selection seeds; previews do not substitute for statistics.
7. No shared scene-family IDs or geometry fingerprints across splits; all
   training pair/set indices obey the declared S0/S1 regime.
8. Throughput, disk bytes, decode time, RSS/VRAM and resume behavior pass the
   bounded pilot before increasing generation scale.
