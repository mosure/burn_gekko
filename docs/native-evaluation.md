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

`gekko-eval activity --config <TOML>` summarizes an immutable NVIDIA process log
from `nvidia-smi pmon -s um -o DT`. It preserves unavailable counters, separates
GPU/PID identities and rejects incomplete rows. Process counter percentages are
neither additive occupancy nor a basis for attributing board energy. The input
hash and observation counts remain in the result.

`burn_gekko_train::evaluation` performs Burn inference. `burn_gekko_eval` scores immutable
exports on CPU. Neither scoring nor publication requires Python; geometric truth
is loaded after RGB-only inference.

| Capability | Contract |
| --- | --- |
| Latent completion | Hidden-token/channel MSE, cosine and spatial variance / fixed teacher; equal target-view weighting |
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
independent intervals. The publication gate requires an entirely positive AEPE
gain interval and nonnegative mean PCK3 gain; the PCK3 interval is also shown and
is not interpreted as an equivalence test. `visual_method` fixes which scored
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
