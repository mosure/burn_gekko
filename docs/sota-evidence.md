# Evidence required for a state-of-the-art claim

`burn_gekko` is a research prototype. The present objective is useful geometric
representation learning through multi-view V-JEPA latent prediction. RGB
completion remains an optional diagnostic. Its earlier blur has not been solved
by changing the prediction space.

The latest [Pilot 08 study](studies/pilot-08-equivariance.md) completes 6,000
updates and **passes all four registered spatial-readout transfer gates**.
HPatches viewpoint AEPE is 20.3217 versus encoder 25.4678 and same-image control
21.3770; ETH3D is 33.5004 versus 36.7393 and 34.8031. All paired AEPE-gain intervals
are positive and mean PCK3 gains are nonnegative. ETH3D PCK3 improvement over the
encoder remains uncertain. These development results qualify a cross-image
conditioning gain for the trained spatial head, not raw decoder features or a
SOTA ranking. Raw HPatches decoder AEPE still regresses to 28.3549.

The fresh 128-room / 512-target cohort shows 5.23% reference benefit with a
positive paired room interval, but teacher-relative spatial variance is only
40.98%. Strict numerical reference-order invariance fails; RGB, camera and depth
heads remain untrained. Independent real data, fine correspondence/pose accuracy,
training-seed replication and protocol-matched public baselines remain open gates.
The single-run [page](../.data/publications/pilot08-equivariance/index.html) and
[PDF](../.data/publications/pilot08-equivariance/paper.pdf) retain these limits.

The preceding [known-transform preflight](studies/pilot-07-equivariance-preflight.md)
uses a new geometric descriptor objective, with 256 updates over 128 rooms and
complete HPatches/ETH3D evaluation. HPatches improves over the same checkpoint's
encoder (AEPE 23.1493 versus 25.4874), but not the same-image conditioned control.
ETH3D shows a small supported conditioning gain (0.1463 pixels), while improvement
over the encoder is uncertain. The combined transfer gate is not passed. A
focused exporter qualifies exact CUDA prediction parity and completes all 3,365
ETH3D pairs inside the remaining budget. The single-run page/PDF retain all four
paired gates, co-visibility, latent diagnostics and known-transform visualizations.

The preceding [bounded spatial-head study](studies/pilot-07-spatial-descriptor.md)
completes 3,000 updates, covers all 8,192 training rooms, and evaluates a fresh
128-room cohort. Reference use and reference-count scaling help latent prediction.
The registered transfer gate nevertheless fails: ETH3D AEPE regresses against the
same checkpoint's conditional block-6 encoder, and HPatches viewpoint PCK3 drops.
The paired intervals and annotated examples are in its single-run page/PDF.
The completed Pilot 07 GPU-command ledger is 11.976 / 12 hours, with 87.19 seconds
left unused. Pilot 08 uses a separate two-hour ceiling, with 4,317.42 seconds
consumed and 2,882.58 seconds left unused. No study GPU job is running. The earlier
[native refinement study](studies/pilot-07-native-spatial-refinement.md)
and chronological studies below retain their original evidence and budgets.

The [benchmark registry](../configs/train/benchmark-registry.toml) records the exact
scope of implemented evaluations and external protocol gaps. A result on
procedural rooms does not establish a result on ETH3D, and patch-grid EPE cannot
be placed in a published full-resolution AEPE table without qualification.

## Completed masking and memory controls

Both masking arms begin from the same own 1,000-update latent checkpoint, with
audited MIT V-JEPA ancestry. Each gets 1,000 additional updates, the same 1,024
training rooms, batch 16, sample order, learning rates, evaluation cadence and
unfreeze gate. Both restart their two AdamW optimizers. This is a weights-only
new phase, not an exact optimizer continuation. The sole training difference is
random visible patches versus a connected compact visible island at 90% masking.
The island policy is not V-JEPA's multi-block mask sampler.

Both are scored on the same fixed random 90% mask. A separate standalone
assessment loads the old checkpoint with that same mask. A comparison between
the old 75% report and a new 90% report would confound model and task difficulty.
The first screen uses validation for selection and must be labeled accordingly.

The [completed controlled continuation](studies/pilot-07-latent-continuation.md) selects
random masking and continues it through its original 2,000-update phase horizon.
It records both the mask arms' different adaptive stage durations and recovery
of the compact arm from an intact update-800 checkpoint after allocation failure.
The selected lineage has 3,000 total updates, including its parent phase. Common
validation improves latent prediction and encoder/decoder correspondence, but
feature variance remains suppressed. Final checkpoint selection precedes fresh
synthetic and HPatches scoring; those results cannot silently become tuning data.

A separate native audit reproduces unused hierarchical-output memory growth and
removes it with a final-only forward path. This qualifies the memory repair,
not a new trained model or an accuracy gain. CPU mathematical equivalence and
native loss-trajectory agreement are reported with their different tolerances.

Controls include the matched monocular branch, unrelated-room references,
shuffled reference token positions, a training-only per-position constant,
teacher/prediction feature variance, and the unchanged V-JEPA encoder for
correspondence. Shuffling after encoding retains positional information inside
features; sensitivity is evidence of reference layout use, not complete proof
of geometric reasoning.

The float64 attention policy passed a narrow earlier inference audit but failed
the full CUDA Fusion evaluator with `Unsupported precision for fusion: f64`.
The failed preflight completed zero training updates and is retained in the
Pilot 07 budget. Production experiments use float32, retain measured numerical
reference-order sensitivity, and reject the unsupported policy before loading
data. CPU tests do not qualify a GPU numerical policy.

## Fusion transfer study

The [subsequent positional and objective audit](studies/pilot-07-fusion-transfer.md)
explicitly turns all HPatches sequences and the previously observed synthetic
test into development data. The fixed checkpoint does not recover encoder-level
matching through a simple readout change or cross-view RoPE removal. A matched
2x2 training screen finds that semantic affinity guidance with RoPE retained
substantially improves attention transfer, while decoder descriptors still
regress. A second matched screen tests direct descriptor affinity alignment,
expands the room pool to 8,192, and keeps the encoder frozen to isolate the loss.

The v14 evaluation adds equally normalized feature controls before refinement
endpoint selection: raw and centered teacher, student and decoder cosine scores
receive the same reciprocal conditional operator as attention, with temperature
fixed to 0.07. A gain over plain encoder cosine alone cannot establish a learned
fusion gain. Both families stay in the report; no per-example readout selection
or holdout temperature sweep is allowed.

A development-only probe compares trained hierarchical feature levels in the
fixed teacher and own adapted encoder. Block 6 reduces conditional viewpoint
AEPE from 30.09 to 25.22 and improves PCK3 from 8.76% to 11.76% in the own
encoder. This is a stronger baseline, not a fusion gain. The registered matched
affinity-level experiment changes only the teacher's feature level while
retaining final-layer latent targets. All subsequent external comparisons retain
eight block-6 controls alongside the original 18 readouts; a gain against the
weaker final layer alone is insufficient.

The [GPU execution qualification](gpu-efficiency.md) separately fixes contiguous
RGB upload. Identical 64-update losses and validation metrics establish numerical
preservation for the matched workload. A 1.63x steady throughput improvement and
8.84% lower whole-command board energy are systems results, not accuracy gains.
GPU activity percentages are not treated as SM occupancy or useful compute.

The complete ETH3D interval bundles now have a separate RGB exporter and CPU
scorer, covering 3,365 pairs across 10 scenes. Tests check half-pixel resize
coordinates against OpenCV, duplicate point rasterization, and unequal scene
weighting. This establishes a local benchmark implementation, not official
refinement parity. The exporter requires a hashed selection record before the
first model evaluation. A fresh 128-room four-view capture with a wider camera
policy provides an additional independent qualification set. Both remained
unscored until candidate selection was sealed; their completed results are in
the [final annotated report](../.data/pilot-07/fusion-audit/fusion-transfer-final-report.pdf).

The independent exporter also binds the actual model names, checkpoint paths,
weight hashes and spatial feature level to that sealed record before loading a
model. A focused test rejects changed weights, duplicate or renamed models and
a changed feature level. The CPU scorer separately checks the complete declared
readout set and export provenance.

The completed qualification improves ETH3D raw decoder AEPE from 53.765 to
39.161 and conditional attention from 100.826 to 40.298. Fresh-room latent MSE
improves 8.65%, with a positive paired room interval. However, the stronger
block-6 encoder still beats all three declared fusion readouts on ETH3D in both
AEPE and PCK3; the paired scene intervals exclude zero. Thus the weak original
fusion has improved, while encoder-level transfer preservation remains an
unmet requirement. The unsuccessful final loss-strength screen is retained,
and the selected checkpoint is fixed before these outcomes.

The selected lineage has 17,200 optimizer updates and 275,200 target exposures
over 8,192 distinct training rooms. The complete Pilot07 GPU-command ledger is
10.345 hours, including failed commands and profiling. One seed, a coarse patch
readout, imperfect float32 reference-order invariance, and the remaining strong
encoder regression prevent a SOTA claim. No RGB blur fix is inferred from the
latent or correspondence gains.

## Qualification sequence

1. **Objective and implementation:** sparse target tokens enter the encoder
   before attention; dense fixed teacher features enter only the loss; neither
   geometry nor homographies enter training or matching scores. Exact resume
   preserves both optimizers and sample order. Weights-only phases name and
   hash their parent and restart optimizer/gate state explicitly.
2. **Useful multi-view learning:** paired room confidence intervals for
   monocular, unrelated-reference and shuffled-reference contrasts; feature
   variance/rank; downstream frozen-encoder and fused readouts. Lower latent MSE
   alone does not qualify a geometry model. Compare aggregate EPE together with
   PCK and mutual-match coverage to avoid flattering a tiny subset of matches.
3. **Scale:** expand rooms and updates only after the screen shows learning.
   Preserve a fixed learning-rate horizon on exact resume. Record exposure
   counts, unique room coverage, stage throughput, process VRAM and elapsed
   command time under the existing 12-hour ceiling. More steps are not assumed
   to improve downstream geometry; measure that separately.
4. **External transfer:** HPatches uses all 116 sequences and 580 image pairs.
   Its primary viewpoint subset exactly matches ZeroCo's 59 sequences / 295
   pairs; the illumination and combined subsets are supplementary. An audit
   matched the official homography CSV to the archive within 4e-11.
   The earlier continuation selected checkpoints using synthetic validation
   before this evaluation. The fusion study now declares HPatches development use.
   Use paired sequence bootstrap and report illumination/viewpoint subsets.
   Keep the matched fixed V-JEPA control. Repeated optimization after seeing
   these results turns the benchmark into development data and must be declared.
5. **Benchmark parity:** retain the complete ETH3D local evaluation and separately
   qualify ZeroCo's official full-resolution readout/refinement variants. Audit current
   competing method weights individually; code licenses do not establish
   weight licenses. No noncommercial checkpoint may initialize or teach the
   candidate. Published numbers can be contextual references only until the
   data, resolution, supervision, architecture and aggregation are matched.
6. **Reliability:** at least three independent training seeds; a fresh synthetic
   holdout disjoint from every ancestor's training/selection seeds; more than one
   masking pattern/seed; camera-baseline and visibility strata; robustness to
   the number and order of reference views. Existing historical test sets are
   development data once repeatedly inspected.
7. **Paper claim:** report gains only for qualified tasks, including any
   regressions and the complete compute/data/initialization ledger. A systems
   or benchmark contribution is valid even when no accuracy record is set.

## Architecture experiments after the controlled screen

If the fusion trunk improves latent prediction but loses encoder correspondence,
test frozen versus adapted encoders under identical data and steps before
increasing capacity. Hierarchical affinity targets and direct block-6 input have
now been tested and help, while simply increasing descriptor weight fails the
registered final selection rule. Further candidates are explicit feature
preservation across fusion, augmentation-based correspondence objectives,
confidence-weighted matching and local refinement. These remain separate
hypotheses. Renderer geometry remains evaluation-only in the self-supervised
experiment; a camera-supervised variant must be separately named. Any new
accuracy development that uses these ETH3D or fresh-room outcomes must declare
that reuse and reserve new independent evidence.

For the paper, maintain a table of exact initialization, objective, teacher
update rule, mask policy, reference count, trainable blocks, dataset version,
target exposures and compute. Show paired latent maps in a common teacher-fitted
PCA basis and correspondence overlays on real RGB. Separate encoder, decoder
feature and cross-attention readouts, and do not select the best readout per test
example or use ground-truth visibility to filter matches before inference.

Primary references: [Gekko](https://arxiv.org/html/2609.01530v1),
[V-JEPA 2.1](https://arxiv.org/html/2603.14482v3),
[ZeroCo](https://github.com/cvlab-kaist/ZeroCo), and the
[HPatches dataset](https://github.com/hpatches/hpatches-dataset).

The September 2026 literature refresh also includes
[RoMa v2](https://arxiv.org/abs/2511.15706),
[RoMa-Omega](https://arxiv.org/abs/2609.09507), and
[MV-RoMa](https://arxiv.org/abs/2603.27542). These make an unrestricted SOTA claim
substantially broader than outperforming an old latent checkpoint. Their
supervision and matching heads differ from this experiment. In particular,
RoMa-Omega distinguishes raw patch matching from the utility of representations
with a trained head; a weak cosine readout is a diagnostic, not a definitive
judgment about every possible downstream use.
