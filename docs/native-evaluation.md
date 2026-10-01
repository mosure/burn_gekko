# Native evaluation

`gekko assess-equivariance --config <TOML> --output <.data/path> --backend cuda`
evaluates one checkpoint on known transforms of validation RGB. It exports
bidirectional NLL, AEPE, PCK8/PCK16, point annotations and the first four image
pairs. The same checkpoint supplies pair-conditioned descriptors, a shared-weight
same-image-conditioned control, and centered intermediate encoder features.
`burn_gekko_eval::warp` scores the known patch-grid point locations on CPU. These
development diagnostics do not establish real 3D viewpoint transfer.

HPatches/ETH3D export configs can enable `self_conditioned_readouts = true`.
The ETH3D selection record binds that flag before inference. HPatches computes
each self-conditioned descriptor once per image, then reuses it across pairs.
Readouts `spatial_self` and `spatial_self_conditional` provide a control for the
benefit of cross-image conditioning with the same learned trunk and head.

ETH3D also supports `focused_spatial_readouts = true` with a fixed spatial layer,
one model and the same-image controls enabled. Only the three predeclared spatial
readouts are computed. The first eight pairs must exactly match the broad exporter
on the actual backend; a 64-pair throughput qualification checks the optional
`max_seconds` ceiling before continuing. Qualification data are retained, and the
CPU scorer still requires the complete benchmark population.

If the optimized path fails its exact backend parity check, retain that failure.
`canonical_spatial_readouts = true` selects the broad calculation order while
emitting the same three spatial methods and their local refinements. It requires
focused mode and local refinement, and must also appear in the sealed selection
record. Canonical hard and local matches share the original score arrays; exact
hard-index and mutual-flag consistency is checked on every pair. The 64-pair
throughput qualification and complete-population requirement still apply. Its
receipt records zero optimized parity pairs, not a passed optimized audit.
Pilot 12 uses this path after a CUDA index mismatch; that numerical discrepancy
remains unresolved.

`local_refinement = true` adds a fractional counterpart to each of those three
controls. Enable the same flag in the ETH3D selection record; HPatches requires
the fixed spatial layer and same-image controls, and ETH3D requires focused mode.
The known-transform audit supports the flag as well. Each method preserves its
hard indices and mutual flags and adds an optional `coordinates` array; integer
coordinates identify patch centers, not image corners.

The independently implemented readout uses the same reciprocal conditional
log scores as the hard matcher. For each query, exponentiate after subtracting
the row maximum, then take the probability-weighted coordinate in a clipped
3 by 3 neighborhood of the hard match. This produces a local fractional location
without adding a trainable head, accepting a camera, or loading a label. Only one
score matrix crosses the device boundary for the hard and refined pair. A
quarter-patch numerical oracle checks its units; integer coordinates reproduce
the original bilinear displacement scorer exactly. Local inference retains
ETH3D's eight-pair exact hard-index/mutual parity audit.

Training summaries include nonoverlapping first/last 32-update scalar windows,
restricted to the selected phase. They describe optimization on changing
minibatches and are not substituted for validation or external measurements.

`gekko-eval training --config <TOML>` closes out a completed training command on
CPU. The manifest names `run`, `ledger`, `telemetry`, `command` and `output` (see
`configs/eval/training-pilot08-equivariance.toml`). It verifies the final checkpoint,
command/run association and telemetry PID, then exports source hashes, exposure
coverage, scalar windows, encoder gradient stages and board efficiency. Reports
also check actual encoder gradient counts against each logged freeze stage. A
configured unfreeze option is never presented as evidence that unfreezing occurred.
Named parameter probes measure QKV/head tensor changes, not whole-model norms.

Feature-preservation runs also bind the training provenance/source identity and
frozen anchor probes. `encoder_preservation_mse` is raw final-feature squared
error divided by each image's anchor energy, with a 1e-6 floor, averaged equally
over sparse target, full target and each reference route. Reports show 100 times
the square root of its first/last-window mean as an RMS drift percentage. This
describes training regularization; it is not RGB PSNR or a held-out task score.

`gekko-eval select-preservation --config <TOML>` checks the registered three-arm
screen's full sample/validation/transform populations, intended gradient stages,
common training source and teacher, pinned coefficient/anchor and unchanged
frozen probes. The fixed retention and geometry gates are described in
[Pilot 12](studies/pilot-12-feature-preservation.md). Real-view diagnostics are
scored only after that synthetic decision, including rejected endpoints.

`gekko-eval activity --config <TOML>` summarizes an immutable NVIDIA process log
from `nvidia-smi pmon -s um -o DT`. It preserves unavailable counters, separates
GPU/PID identities and rejects incomplete rows. Process counter percentages are
neither additive occupancy nor a basis for attributing board energy. The input
hash and observation counts remain in the result.

Both dated (`-o DT`) and undated process logs are supported. A standalone `#`
header marker is not a data column; unavailable counters remain unavailable.

`gekko-eval dispatch --config <TOML>` consumes checksummed Nsight
`cuda_gpu_trace` and `cuda_api_trace` JSON exports using `ts=ns:dur=ns` units.
The manifest supplies `description`, `output`, and `[gpu_trace]`/`[api_trace]`
tables containing `path` and `sha256`. No new dependency or Python analysis is
required. Exactly one traced device and API process are expected. Overlapping
kernel/memory intervals are merged within the first-to-last GPU event window;
API durations are clipped to that window and merged separately. Per-name sums
describe work and can overlap. Small-kernel counts and synchronization time
alone do not prove dispatch limitation; uncovered trace intervals may contain
other GPU processes, CPU/JIT work or profiling overhead.

`burn_gekko_train::evaluation` performs Burn inference. `burn_gekko_eval` scores immutable
exports on CPU. Neither scoring nor publication requires Python; geometric truth
is loaded after RGB-only inference.

The actual-view exporter also preserves dense `pose-predictions.json` for the
synthetic camera retention probe. `gekko-eval synthetic-pose --config <TOML>`
checks dataset/prediction identities and complete room/method populations before
using known camera intrinsics and extrinsics on CPU. It never filters model
matches using visibility truth. Every declared solver seed contributes, failed
fits remain in the denominator, and low-baseline exclusions are counted. See
[Pilot 16](studies/pilot-16-continuation.md) for the fixed thresholds and the
distinction between this geometric probe and learned camera heads.

`gekko-eval audit-view-targets --config <TOML>` verifies the training-label cache
and reports all directed pairs, empty targets and valid-query coverage per split.
It uses the same bilinear-label policy as training, without launching GPU work.

| Capability | Contract |
| --- | --- |
| Latent completion | Hidden-token/channel MSE, teacher-power feature SNR, cosine and spatial variance / fixed teacher; equal target-view weighting |
| Co-visibility | AP/AUROC with ties grouped, positive/known counts retained and unknown labels excluded |
| Correspondence | AEPE and inclusive PCK at 1/3/5 pixels; declared coordinate system and aggregation |
| Rotation | Geodesic SO(3) angle; invalid rotations and reflections rejected |
| Translation | Signed direction error in anchor-camera coordinates; zero truth baseline excluded; zero valid-baseline prediction counts as 180 degrees |
| Intrinsics | Relative error of fx/width and fy/height; centered principal point declared |
| Pose | Trapezoidal AUC of max(rotation, translation error) at 5/10/20 degrees with eligible-baseline counts |
| RGB | PSNR with declared data range; zero MSE has explicitly infinite PSNR |
| Efficiency | Observed board energy, coverage, joules/target and median/p95 update timing by encoder stage; shared desktop load included |

Camera metric implementation is **not evidence of a trained camera head**. Future
heads supply `schema::Capability` records with status, units, populations,
aggregation and limitations. Unevaluated capabilities cannot contain metrics.

Feature SNR is `10 * log10(signal_power / MSE)`, where `signal_power` is the mean
squared teacher feature on the same hidden-token/channel population. The report
averages the per-view dB values, rather than applying a log to population-mean
MSE. Zero signal or error has no finite dB value and is represented explicitly as
`None`. This is distinct from bounded-image PSNR: V-JEPA feature values have no
declared image peak range. A 3.01 dB increase means half the error at fixed signal
power; no universal RGB-quality threshold applies.

Reports with the new feature-SNR fields require all target arrays. Native Rust
recomputes signal power, MSE, cosine, SNR and variance for every declared identity
and mask, then verifies row and population summaries. Reference benefit is
`100 * (1 - mean_cross_MSE / mean_monocular_MSE)`. Spatial variance is displayed
as a percentage of teacher variation; matching 100% alone does not prove spatial
accuracy. The HTML and PDF share these definitions and preserve the raw units.
The same array verification produces an explicitly post-hoc breakdown of hidden
MSE by geometric majority-visibility label, pooling patches and excluding unknown
labels. This diagnostic has separate denominators and no acceptance gate.

`gekko-eval score --config <TOML>` binds one checkpoint, prediction JSONL and export
provenance to checksummed image/label manifests. Missing or duplicate observations,
wrong grids, unknown pairs, nonfinite values and mixed checkpoints are rejected.

ETH3D requires all 3,365 pairs, ten scenes and seven intervals. Target coordinates
use ties-to-even rounding; duplicate locations retain the last label. AEPE/PCK
average pairs, scenes, then intervals; point-weighted PCK3 is separate. Hard-patch
displacements use half-pixel bilinear interpolation at original image resolution.

HPatches requires 116 sequences / 580 pairs. The 59 viewpoint sequences / 295
pairs are primary; illumination is separate. Homographies and flow are scored at
240x240 after 256x256 model input. These local readouts do not reproduce published
ZeroCo refinement; comparisons to published SOTA numbers are not valid.

The native migration rescored three readouts on both full benchmarks. Maximum
AEPE difference from the archived Python oracle was below 2.6e-7 pixels; PCK
differences were below 3.2e-7 at floating-point boundaries. Registered tolerances
were 2e-5 pixels and 1e-6 PCK. See
`.data/engineering/organization/benchmark-parity.json`. Native float64 scoring does
not claim bitwise parity with NumPy/OpenCV float32 arithmetic.

Intervals use 10,000 deterministic SplitMix64 cluster-bootstrap draws: whole
ETH3D scenes, HPatches sequences or synthetic rooms. They measure sampling
uncertainty, not training-seed uncertainty. Reusing a benchmark for checkpoint
selection makes it development evidence in later experiments.

Training coverage comes from logged room/view identities up to the selected
endpoint. Rust distinguishes target exposures, unique rooms and unique room/view
pairs, validates update continuity and batch size, and reports coverage against
the configured room pool. These are counts for the selected phase, not inferred
pretraining exposure counts or a claim of independent samples.

Optional `[[contrasts]]` tables declare a candidate and control readout from the
**same checkpoint**. Scoring requires identical pair identities and label counts.
Differences are computed per pair, averaged equally over groups within a scene
or sequence, then bootstrapped over scenes/sequences. HPatches contrasts use only
the viewpoint population. Positive gains mean lower AEPE or higher PCK3 for the
candidate; these paired intervals are more informative than subtracting two
independent intervals. The default `gate = "transfer"` requires an entirely positive
AEPE-gain interval and nonnegative mean PCK3 gain. `gate = "local_precision"`
requires an entirely positive PCK3-gain interval and nonnegative mean AEPE gain.
Both intervals and the gate outcome are retained. These are not equivalence
tests. `visual_method` fixes which scored
readout supplies the annotated examples, without selecting it by example quality.

The spatial-descriptor experiment retains encoder, fusion and attention readouts
and adds `spatial_residual` / `spatial_residual_conditional`. Both apply the learned
bounded correction to centered encoder features. The conditional operator and its
0.07 temperature match the declared intermediate-encoder control.

Completion figures recompute hidden MSE from exported floats. One teacher-fitted
PCA and common color bounds apply to all displayed predictions. Benchmark figures
retain first/middle/last primary-subset pair identities and eight uniformly selected
valid labels, without selection by error. Ground truth is visualization-only.

`gekko-eval references --config <TOML>` verifies that one checkpoint, dataset,
mask, split and target identities are shared across reference-count assessments.
It reads the actual inference configs, intersects targets and checks monocular
isolation before reporting each count's MSE. See
[`references-native-spatial-refinement.toml`](../configs/eval/references-native-spatial-refinement.toml).

`gekko-eval camera --config <TOML>` scores camera JSONL records under a checksummed
prediction provenance record. `camera_export::CameraScoreConfig` declares the
coordinate frame and input contract; mismatches and duplicate sample IDs fail.
It emits the same checkpoint-bound head capability consumed by the publication
generator. Camera heads and future decoders can therefore extend the page without
adding a Python metric/report program. The current checkpoint remains untrained
for camera prediction.

## Calibrated real-image motion

`gekko-eval prepare-tum --config configs/data/prepare-pilot10-tum.toml` caches a
fixed Freiburg 3 cohort through `burn_gekko_data::tum`. RGB manifests reject camera
fields; ground-truth labels live separately. The `real_pose_export` trainer
binary verifies a sealed checkpoint, manifest and readout set before RGB-only
inference. It shares the fixed local matcher with HPatches/ETH3D.

`gekko-eval pose --config configs/eval/pose-pilot10.toml` fits camera motion on CPU
using known intrinsics, normalized eight-point essential RANSAC and cheirality.
The solver is in `burn_gekko_eval::pose::solver`; complete populations, failure
accounting, sequence aggregation and transfer gates are in `pose::benchmark`.
Errors use degrees, AUC/recall use fractions rendered as percentages. Every failed
fit contributes 180 degrees; baselines below the declared minimum are counted and
excluded only from translation/pose. Solver success is not pose accuracy.

This is a calibrated geometric probe, separate from a learned camera head or
intrinsics prediction. The report's optional `calibrated_pose` artifact retains
that distinction, rechecks summaries and gates against all rows, and renders
deterministic predicted-match overlays. Read the
[registered study](studies/pilot-10-real-pose.md) for calibration, temporal pairs,
association exclusions, dataset attribution and limitations.

The TUM cache recipe accepts `image_size` (default 256, maximum 512, divisible
by the encoder's 16px patch size). Original RGB is resized directly, and every
prediction carries its grid. Camera fitting maps half-pixel patch centers back
to original-image coordinates, retaining original intrinsics and pixel thresholds.
The report labels model input resolution separately from its 256px thumbnails.

`gekko-eval pose-replay --config <TOML>` verifies exporter engineering against
sealed predictions. Checkpoint, image/selection hashes, methods, pair identities,
grid, hard indices and mutual flags must agree; fractional coordinates use an
explicit tolerance. The receipt records differences even when the qualification
fails. The caller must require `passed = true` before promoting the exporter.

## Replays and controlled adaptation

`gekko-eval replay --config <TOML>` compares the overlapping prefix of two
checksummed training logs. Update numbers, sampled targets, encoder stages and
gradient counts must match before component losses and gradient norms are
compared with declared tolerances. Timing is excluded from numerical agreement.
An explicit `prefix_updates` can qualify a shorter engineering replay against a
completed run; both inputs must contain that entire prefix, and the receipt
retains the full original update count. Omitting it requires replay of every
original update.

`gekko-eval select-adaptation --config <TOML>` implements the registered Pilot 11
validation-only selection. Both 2,048-update screens must complete with verified
teacher and encoder probes, matching configurations except trainable stage, and
identical training sample order, validation masks and 32-room bidirectional warp
population. Pixel errors and PCK8 are
recomputed from every point. Full adaptation is accepted only when it improves
warp error without reducing PCK8, retains reference benefit and stays within 1%
of the tail arm's completion MSE. The immutable selection receipt records every
gate and source hash. It does not compare private runs on a publication page.
This is a study-specific selector, not a general hyperparameter search service.
Its original contract defaults to 2,048 updates. The registered longer recovery
uses `updates = 12000` plus `[parents.tail.summary]`, `[parents.tail.warp]` and
corresponding full-arm pinned inputs. It verifies the complete matched screen
parents and each continuation's warm-start path/checksum before permitting the
two parent identities to differ. All remaining recipes and populations must
still match; the 1% completion threshold is unchanged.

## Actual-view renderer supervision

`gekko-eval prepare-view-targets --config configs/data/targets-pilot13.toml`
creates a compact CPU target cache under `.data/`. The data crate averages the
central 2 by 2 source pixels only on a depth-continuous surface, checks source
reprojection and reference-depth visibility, and distinguishes visible, occluded,
out-of-frame and unknown points. Visible locations outside the descriptor-center
hull are excluded instead of clamped to an edge. The optional trainer objective
uses these labels in a separate dense pair branch; masked completion inputs
remain unchanged. This auxiliary is renderer-supervised, not self-supervised.

`view_geometry_export --config <TOML> --output <.data/path>` scores view 0 against
view 1 in both directions on fixed validation rooms. It retains hard and local
readouts for the pair-conditioned, same-image and encoder controls.
`gekko-eval select-view-geometry --config <TOML>` enforces the fixed short screen
contract in [Pilot 13](studies/pilot-13-view-geometry.md). The CPU integrity and
coverage audit is `cargo run -p burn_gekko_data --example view_targets_audit --
<CACHE> <DATASET> <OUTPUT.json>`. No empty pair was omitted in Pilot 13's cache.
## Diagnosing completion detail and camera-solver stability

The native `latent-detail` command consumes one complete assessment and checks
its metrics against the underlying arrays. Its TOML pins `directory`,
`metrics_sha256`, `provenance_sha256`, `checkpoint_sha256` and `output`.
Per-channel spatial centering gives an additive MSE decomposition: channel-mean
bias plus centered structure error. Neighbor differences use horizontal and
vertical hidden/hidden pairs, with no wraparound. Zero-power quantities remain
undefined with explicit counts. A teacher-assisted scalar-gain oracle and
teacher-variance matching test whether simple amplitude changes could explain
missing variation; neither is a legal inference transformation or improved
model result. The companion `.head.json` follows the ordinary checkpoint-bound
capability schema.

`latent-replay` validates fixed assessment prefixes using identical teachers,
checkpoints, masks, baseline means and reference counts. It checks complete
arrays and scalar metrics under declared tolerances. Prefix size is explicit;
an incomplete export cannot silently reduce the compared population.

`pose-stability` takes a pinned original development `PoseReport`, 2--16 distinct
RANSAC seeds including the original seed, and a new output directory. It repeats
the exact solver policy on all methods/pairs, checks original-seed reproducibility,
and retains every seed, failure and sequence contrast. Seed ranges and population
standard deviations are descriptive, not confidence intervals across scenes or
training runs. Changing solver policy requires a separate original report and
declared protocol. The main camera gate is not retroactively changed by a seed
panel or best-seed selection. See the [Pilot 14 study](studies/pilot-14-information-diagnostics.md).

Standalone assessment provenance now distinguishes scoring-only evaluation
geometry from the selected phase's training geometry objective and pins the
training config. Legacy assessment flags are retained in old artifacts and
explained separately; inference does not load a training-label cache.
