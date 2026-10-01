# Evidence required for a state-of-the-art claim

`burn_gekko` is a research prototype. The present objective is useful geometric
representation learning through multi-view V-JEPA latent prediction. RGB
completion remains an optional diagnostic. Its earlier blur has not been solved
by changing the prediction space.

The [output-head stability study](studies/head-stability-15.md) fits separate
camera and RGB heads on the fixed foundation. Its 800-update schedule passes
finite-gradient, learning, valid-rotation and exact-replay checks. On 24 targets
from 8 development rooms, RGB PSNR is 21.09 dB, versus 20.70 dB for the same head
without references. Camera errors remain 11.64 degrees rotation, 32.68 degrees
translation direction and 44.35% focal relative error. Training camera loss is
0.000131 versus validation 0.387406: strong overfitting, despite stable numerical
optimization. These are bounded head-training results, not full end-to-end
joint-training qualification or state-of-the-art evidence. The latest
[page](../www/project/index.html) and [PDF](../www/project/paper.pdf) retain these gaps.

The completed [Pilot 14 diagnostics](studies/pilot-14-information-diagnostics.md)
do not train new weights. On the fixed Pilot 13 checkpoint, a third reference
reduces hidden-token error by 1.66% on the same 512 targets; the paired room
interval for absolute reduction is [0.002545, 0.003599]. Spatial structure remains
weak. Per-view teacher-assisted contrast correction barely helps, while matching
teacher variance raises MSE by about 21%. Raising variance alone is not a
demonstrated solution to missing detail.

Camera qualification must account for RANSAC variability. With frozen predictions
and eight solver seeds, the original 2,048-trial same-image / encoder gates pass
only 1/8 and 2/8 times. A separately registered 8,192-trial probe raises mean
AUC@10 from 7.99% to 9.42%, but each gate passes only 2/8 times and only one seed
passes both together. The original seed
passes both at the larger trial budget; selecting it would conceal instability.
No best-seed promotion, camera-head claim or new training improvement is made.
The [page](../.data/publications/pilot14-information-diagnostics-reviewed/index.html) and
[PDF](../.data/publications/pilot14-information-diagnostics-reviewed/paper.pdf) preserve
the original primary camera result alongside these diagnostics, all for one
checkpoint. All 157 workspace tests pass; exact 32-target GPU replay qualifies
the assessment provenance correction. Shared-process monitoring now runs with
a supported interval and explicit unavailable counters. There are 3.63 minutes
left in the prior 12-hour GPU allowance; a further training ceiling is pending.
All earlier study budget snapshots and frozen artifacts remain unchanged.

The completed [Pilot 13 study](studies/pilot-13-view-geometry.md) adds a detached,
renderer-supervised correspondence objective across actual camera views. In a
matched 384-update, one-epoch screen over 2,048 rooms, synthetic viewpoint error
falls 26.9%, with completion MSE only 0.35% higher, inside its 1% retention gate.
Selection precedes all external inference. The fixed local readout reduces mean
error by 12.4% on HPatches viewpoint and 16.3% on ETH3D versus the matched control.
Both endpoints complete all 580 / 3,365 pairs. The selected checkpoint reaches
14.8376 / 23.8085 pixels mean error and 26.23% / 5.68% PCK3 in their different
scoring frames. All ten declared within-checkpoint spatial contrasts pass.
The coarse ETH3D encoder-control precision gain is small and its interval still
includes zero; the existing transfer gate requires a nonnegative mean. This
qualifies the declared spatial readouts, not every decoder or attention feature.

Camera transfer remains incomplete. The selected checkpoint's calibrated TUM
probe gives 21.10% pose recall within 10 degrees but 7.65% AUC@10, below both the
matched control (8.35%) and parent (8.50%). Its encoder-control gate passes;
the same-image gate fails on `structure_texture_near`. Known intrinsics and
essential-matrix RANSAC are not a trained camera head. This accuracy tradeoff
must remain visible even though synthetic checkpoint selection passed.

A conditionally preregistered fresh cohort adds 128 new rooms / 512 targets,
with verified seed disjointness and no subsequent fitting. Reference benefit
is 7.18%, feature signal/error 7.38 dB, spatial variation retained 43.56%, and
co-visibility AUROC 0.7382. The paired room interval for MSE reduction is
[0.012480, 0.016027]. Hidden-input isolation passes; strict numerical reference
permutation fails. RGB/camera/depth heads remain untrained. Oversmoothing and
camera weakness remain, so SOTA and an RGB blur fix are not established.
The reviewed [page](../.data/publications/pilot13-geometry-reviewed/index.html)
and [PDF](../.data/publications/pilot13-geometry-reviewed/paper.pdf) show one
selected checkpoint with its own controls. Arm comparisons remain internal.

Pilot 13 consumes 43.69 GPU-command minutes. Combined Pilot 11--13 usage is
11.900 / 12 authorized hours, leaving 6.02 minutes; all model jobs are finished.
The objective adds 18.3% median update time and 12.6% observed board energy per
target. Shared desktop load remains included. A supplementary per-process
observer failed at startup; board telemetry and process VRAM remain valid, but
historical process SM activity is unavailable. All 152 workspace tests and CUDA
strict Clippy pass after repairing a concurrent capture-test fixture race;
failure logs remain retained. Native publication and browser checks pass.
Independent seeds, stronger pose accuracy and protocol-matched public baselines
remain open requirements. Historical studies retain their own budget snapshots.

The preceding [Pilot 12 study](studies/pilot-12-feature-preservation.md) addresses
the full-encoder adaptation tradeoff with a pinned, training-only feature anchor.
Its matched screen retains completion MSE while reducing known-transform error
by 35.8%. The selected recipe's fixed 4,096-update continuation passes all
registered refined-readout matching and calibrated-motion transfer gates against
equally processed controls from that same checkpoint. The coarse ETH3D readout
still fails its encoder-control gate: mean pixel error improves, but PCK3 drops
by 0.0606 percentage points. This failure remains in the paper. HPatches viewpoint / ETH3D
within-three-pixel accuracy is 24.00% / 5.15%, with mean errors 16.9561 / 28.3917
pixels in their different scoring frames. All 580 / 3,365 pairs are retained.

On the 186-pair TUM development cohort, pose recall within 10 degrees is 19.48%
across 185 pose-eligible pairs. Pose AUC@10 improves over both controls in every
sequence, with unchanged overall solver success. This qualifies the registered
within-checkpoint conditioning gate; it is not a learned camera head, an official
pose benchmark result or a strong absolute accuracy result. Three related
sequences and one training seed remain insufficient for broad generalization.

Fresh completion evaluation on 128 newly captured rooms / 512 targets gives
7.47% reference benefit, 7.33 dB feature signal/error and only 42.52% of teacher
spatial variation. The paired room interval for absolute MSE reduction is
[0.01325, 0.01690]. Co-visibility AUROC is 0.7315. Hidden-input isolation passes;
strict numerical reference-order invariance fails. RGB/camera/depth heads remain
untrained. The reviewed [page](../.data/publications/pilot12-main-reviewed/index.html)
and [PDF](../.data/publications/pilot12-main-reviewed/paper.pdf) show one run with
its own controls, annotated examples and every retained limitation. Synthetic
arm comparisons stay in the internal study. SOTA is not established.

The GPU study is closed, at 11.172 / 12 combined authorized GPU-command hours,
with 49.71 minutes unused. All 149 workspace tests pass; native scoring and
publication checks pass. The original strict startup forward-parity failure,
its preregistered numerical amendment, the failed optimized ETH3D export and
the corrected capture-wrapper invocation remain auditable. The canonical
export completes every pair and shares hard/local score arrays. Its correctness
checks do not relabel optimized parity as passed. Historical studies below
retain their original results and budget snapshots.

The completed [Pilot 11 main phase](studies/pilot-11-full-adaptation.md) performs
12,000 continuation updates over 8,192 rooms, with all scheduled stages and
teacher probes verified. Its fresh 128-room / 512-target evaluation gives 6.42%
reference benefit, 7.36 dB feature signal/error and only 43.30% of teacher spatial
variation. Hidden-input isolation passes; strict numerical reference permutation
fails. All matching transfer and local-precision gates pass, with 18.7348px /
20.78% PCK3 on HPatches viewpoint and 29.7577px / 4.37% PCK3 on ETH3D. Their pixel
frames differ. All 580 / 3,365 pairs remain in the reported populations.

Camera transfer remains unqualified. On the reused 186-pair TUM development
cohort, pose recall within 10 degrees is 14.61% at 256px and 25.97% at 512px.
Both resolutions fail the positive-gain-in-every-sequence tests against the
same-image and encoder controls. The secondary resolution probe is a readout
diagnostic of the same fixed checkpoint, not a checkpoint selection rule.
The reviewed [page](../.data/publications/pilot11-main-reviewed/index.html) and
[PDF](../.data/publications/pilot11-main-reviewed/paper.pdf) retain all failures,
learning curves and deterministic annotated examples. Neither improved latent
error nor matching gains establish sharp RGB output or a SOTA result.

The preceding [Pilot 10 qualification](studies/pilot-10-real-pose.md)
evaluates the frozen Pilot 09 weights on three locally unused TUM sequences.
All 186 eligible pairs are scored; solver failures remain in the denominators.
Fusion gives 20.03% pose recall at 10 degrees, versus 14.58% for the same-image
control and 13.57% for the encoder. Mean rotation and signed translation-direction
errors are 8.03 and 52.71 degrees. The strict all-sequence gate **fails** against
the same-image control. Known intrinsics and a native eight-point solver produce
these motion estimates; they do not establish learned camera prediction.
The [page](../.data/publications/pilot10-real-pose/index.html) and
[PDF](../.data/publications/pilot10-real-pose/paper.pdf) retain the negative gate.
Any reuse of this cohort is now development, including Pilot 11. Its completed
2,048-update screen cuts known-transform error from 7.4941 to 3.4275 pixels with
full adaptation, but completion MSE regresses by 4.23%, failing the registered
1% retention limit. Its fixed tail continuation is completed above. A
[matched longer recovery](studies/pilot-11-recovery.md) also completes 12,000
updates. It reduces known-transform error from 7.4265 to 2.2721px and raises
PCK8 from 73.64% to 98.99%, but completion MSE remains 4.99% higher than tail,
failing the unchanged 1% limit. Retain tail and skip the rejected arm's external
exports. Both completed decisions use synthetic validation only. The separately
registered [batching probe](studies/pilot-11-batch-probe.md) completes all three
legs and passes numerical replay, with 1.48% drift between batch-16 controls.
Batch 32 uses 1.965 times the process VRAM for only 14.64--16.33% more throughput,
failing its 20% performance threshold. Batch 16 and the selected weights stay
unchanged. Pilot 11 closes at **26,476.85 / 43,200 GPU-command seconds**
(7.355 / 12 hours), with all jobs finished and 4.645 hours unused. Shared-load
energy and trace limitations remain explicit in the internal studies.

The preceding [Pilot 09 study](studies/pilot-09-local-readout.md) completes 2,700
continuation updates and passes both registered local-precision gates and all
same-operator fusion-transfer gates. The fixed local centroid changes only the
readout within the selected checkpoint: matches within three pixels increase
from 13.26% to 19.69% on HPatches viewpoint and 2.33% to 4.19% on ETH3D. Mean
error falls from 20.0946 to 19.4655 and 33.1232 to 30.8876 pixels, respectively.
HPatches uses a 240 by 240 scoring frame; ETH3D uses original image coordinates.
The percentages are not comparable across those scales. All 580 / 3,365 pairs
are scored; positive paired precision intervals qualify the readout change.

Equally refined encoder and trained same-image controls are also beaten, with
positive paired error and precision intervals on both benchmarks. These remain
development datasets; the result isolates the inference operator, not the
continuation phase's effect or independent generalization. One training seed,
low absolute fine-match accuracy and unmatched public protocols prevent a SOTA
claim. The fresh 128-room / 512-target cohort has 7.30 dB feature signal/error
(not RGB PSNR), 5.99% lower latent MSE with references and only 42.27% of teacher
spatial variation. Strict numerical reference-order invariance still fails;
RGB, camera and depth heads remain untrained. The single-run
[page](../.data/publications/pilot09-local-readout/index.html) and
[23-page PDF](../.data/publications/pilot09-local-readout/paper.pdf) include a
metric guide, annotated matching and completion, uncertainty and efficiency.

The preceding [Pilot 08 study](studies/pilot-08-equivariance.md) completes 6,000
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
consumed in its immutable ledger. Pilot 09 draws only from its 2,882.58-second
remainder and consumes 2,345.76 seconds. Combined usage is **111.05 / 120 minutes**,
leaving 8.95 minutes at that closeout. Pilot 10 then consumes 29.49 seconds,
leaving **8.46 minutes unused** in the old allowance. Pilot 11 uses its own new
ledger and does not modify these completed receipts. The earlier
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
