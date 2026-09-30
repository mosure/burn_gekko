# Evaluation, metrics, and controlled experiments

Status: proposed protocol. Freeze each dataset/protocol version, metric reduction,
failure rule and primary comparison before running the confirmatory study.

## 1. Questions the evaluation must answer

1. Does the model use a reference view to improve features and reconstruction?
2. Does predicted RI identify geometric co-visibility, and where does it fail?
3. Does joint multi-view fusion outperform pairwise fusion at controlled cost?
4. Does sparse encoding reduce actual runtime while retaining useful geometry?
5. Do results transfer beyond the renderer and survive independent training seeds?

Training loss, attractive RGB completions, token counts and attention heatmaps
alone do not answer these questions. Pairwise matching, visibility, downstream
geometry and measured cost all have separate protocols.

## 2. Evaluation ladder

| Level | Data / supervision | Required result |
| --- | --- | --- |
| E0: mathematical correctness | Analytic images, cameras and surfaces | Loss/gradient, reprojection, visibility and permutation checks |
| E1: in-distribution synthetic | Scene-disjoint sealed Zeroverse split | RI, correspondence, failure strata and reconstruction diagnostics |
| E2: controlled synthetic shift | Independent geometry with held-out factor ranges | Layout/material/lighting/camera robustness and sparse degradation |
| E3: external synthetic | Candidate Hypersim split | Generator-domain transfer without target-domain fitting |
| E4: real indoor zero-shot | Fixed real-scene/pair lists, e.g. ScanNet++ and ETH3D protocol | Untuned geometry/correspondence and RI performance |
| E5: frozen supervised probes | Fixed training labels, held-out scenes | Pose, depth/pointmap or matching capability of each frozen representation |
| E6: adaptation | Explicit RGB-only or labeled fitting data | Additional performance, clearly separated from E4/E5 |

The [ETH3D overview](https://www.eth3d.net/overview) contains multiple benchmark
families. Specify the exact correspondence subset and pair-generation/evaluation
code; an arbitrary ETH3D stereo score is not automatically comparable with Gekko's
correspondence experiment. Follow official split conventions for
[ScanNet++](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation) and the
[Hypersim source](https://github.com/apple-aiml-research/ml-hypersim). Dataset access
and downloads belong to implementation, not this planning change.

“Zero-shot” means no fitting on the evaluation domain in this study. The upstream
V-JEPA pretraining population may have unknown overlap; disclose that limitation
rather than asserting every real scene is unseen by the foundation encoder.

## 3. Predeclared endpoints

Register one primary endpoint per claim so a favorable metric cannot be selected
after seeing results:

| Claim | Primary endpoint | Supporting/guardrail endpoints |
| --- | --- | --- |
| H1: RI improves fusion representation | Scene-macro AEPE of a fixed correspondence readout, full objective vs MAE+CroCo | CroCo control, pair RI AP, frozen pose probe, low-overlap failures |
| RI is a useful co-visibility capability | Scene-macro directional AP of predicted RI against geometric labels | Raw-error proxy and feature-confidence baselines, boundary/texture strata |
| H2: joint references improve fusion | AEPE at a fixed total reference-token budget vs best validation-selected pair-pooling control | Set-visibility AP, fixed-GPU-hour comparison, seen/unseen view counts |
| H3: sparse execution is useful | End-to-end latency at a preregistered quality constraint | AP loss, relative AEPE degradation, memory and energy if measured |
| H5: real-domain transfer | Same fixed AEPE readout on a frozen real-domain protocol | RI AP, pose AUC and coverage/failure rate |

Proposed sparse acceptance: at least 20% lower median end-to-end latency, no more
than 2 absolute AP percentage points lost, and no more than 5% relative AEPE
increase versus the same model's dense operating point. Register this before the
test sweep. Show the full frontier even if no operating point meets it.

Use confidence intervals for paired scene-level differences. An improvement
claim needs its interval to support the registered direction, not merely a
better point estimate. A practically negligible difference can be statistically
detectable; report both size and uncertainty.

## 4. Metric definitions

### Geometric co-visibility and reference utility

Compute labels using the [geometry protocol](data-pipeline.md#5-geometric-co-visibility-labels).
Use the predicted raw RI score for ranking metrics; it is not automatically a
probability. Use the unmasked inference path for the main co-visibility result.

| Metric | Definition / reporting rule |
| --- | --- |
| Directional AP / AUPRC | Rank valid pixels for each ordered target/reference edge; aggregate with scene weighting |
| AUROC | Supplemental; always report positive prevalence and AP because easy negatives can inflate it |
| IoU and F1 | Use one validation-fitted threshold, frozen before test; report threshold provenance |
| Precision/recall at selected operating points | Predeclare desired recall/precision; retain false-positive maps |
| Boundary quality | Boundary F-score with a stated pixel tolerance, plus interior/boundary AP |
| Set AP | Compare direct set score with OR visibility labels and explicit unknown handling |
| Calibration | Brier score, NLL and reliability/ECE for a separately fitted probability mapping |
| Utility fidelity | Correlation/rank correlation and weighted RI regression error against repeated-mask reconstruction improvement |
| Reference ranking | NDCG or top-k utility regret against measured reconstruction gain; geometry-overlap ranking is a separate diagnostic |

For AP, report both global pixel-micro and scene-macro results. A proposed macro
convention pools eligible directional pixels within each scene, computes AP, then
averages scenes equally; preserve per-edge metrics as well. Empty-class scenes
have undefined AP/AUROC: report counts and exclusion policy, and evaluate their
false-positive behavior separately. Do not quietly replace undefined values by
zero, one, or a favorable subset.

Fit a scalar monotonic calibration mapping on a dedicated validation calibration
split. Record its parameters and score distribution. This is labeled calibration;
raw ranking remains the uncalibrated capability. Reliability plots must identify
the binning rule, class prevalence and calibration domain. Keep calibration fixed
across test sparsity/view-count conditions to reveal drift; separately calibrated
variants are additional rows.

Diagnostic raw RI requires reconstruction errors and a sampled target mask.
Average it over a fixed number of masks, initially eight, and state that budget.
Compare it with learned RI on the same pairs/pixels; report the extra inference
cost. Low MAE-error pixels are explicitly stratified instead of silently removed
from headline geometric metrics.

### Correspondence

Use one documented descriptor readout across objective ablations. Initial
proposal: L2-normalized final per-view encoder features as the foundation
baseline, and final normalized decoder features `f(t|r)`/`f(r|t)` for fusion.
Use the same mutual nearest-neighbor or soft-argmax matching method, coordinate
convention, and subpixel conversion for every row. Fix the chosen readout before
confirmation; tune any thresholds only on validation.

For a predicted match `qhat(p)` and valid GT match `q(p)`:

```text
EPE(p) = Euclidean norm(qhat(p) - q(p)) in original-image pixels
AEPE   = mean EPE over the declared valid correspondence support
PCK@d  = fraction with EPE <= d
```

Report AEPE, median EPE, PCK at 1/3/5 pixels where resolution permits, and
normalized PCK at a fixed fraction of image diagonal. Report unmatched/rejected
fractions. A confidence filter must show coverage-risk curves and an all-query
result; it may not improve AEPE merely by discarding difficult queries.

Use correspondence support independent of the predicted score. Geometrically
non-visible points contribute to matchability/false-match metrics, not fabricated
pixel correspondences. Sparse descriptor outputs need declared interpolation or
query coverage. Include both queried-token accuracy and full-image accuracy
where a dense claim is made.

Frozen V-JEPA encoder features should be identical across decoder-only training
runs. Improvements in those runs belong to fused decoder features or heads;
never describe them as improvements in the frozen encoder itself.

### Pose and reconstruction probes

Use fixed calibrated matching plus a robust essential-matrix estimator for a
label-free geometric pose readout. Register RANSAC settings, correspondence cap,
random seed, cheirality handling and minimum baseline rules.

Rotation error is the angle of `R_pred * transpose(R_gt)`. Translation-direction
error is the angle between unit translation vectors after the documented pose
sign selection. A typical pose AUC uses `max(rotation_error, translation_error)`
at 5/10/20 degrees. Count solver failures as failures; report pure-rotation and
negligible-baseline cases separately rather than discarding them invisibly.

A metric translation or pointmap probe is supervised and depends on scale
information/priors. RGB-only epipolar pose estimates have no metric scale.
Report translation in metres only when a labeled scale-predicting probe or
calibrated scale source supplies it. Use fixed joint angle/distance thresholds
for that probe, with a separate protocol when reproducing a paper's thresholds.

Depth/pointmap metrics: AbsRel, RMSE, log-RMSE, delta accuracy, pointmap Euclidean
error, and optionally 3D accuracy/completeness/F-score. State coordinate frame,
units, valid depth range, clipping and alignment. Present metric predictions
without alignment as the primary metric claim; scale/Sim(3)-aligned variants are
separate. Use one global alignment per declared scene/set, never per pixel or a
different invisible fit for each view. Multi-view cycle/reprojection consistency
is a useful supplement, not proof of correct geometry by itself.

### RGB completion and representation health

Log masked MSE/MAE and cross-view improvement over MAE. Patch-normalized output
does not yield an unconditional absolute-color reconstruction: visualization
using GT patch means/variances must be labeled as diagnostic. If RGB PSNR/SSIM
or perceptual metrics are desired, add an explicitly trained raw-RGB prediction
path or disclose the target-statistics dependency and keep it out of the primary
geometry claim.

Also monitor descriptor variance/effective rank, cosine similarity of unrelated
images, layer norms, and reference-shuffle sensitivity. Use fixed real/synthetic
diagnostic inputs to distinguish representation collapse from domain mismatch.

### Efficiency

Report two workloads: one target with `K` references, and all requested target
outputs for a complete set. The second includes every additional decoder/pair
pass; do not report the first as the cost of the second.

Measure median/p90 end-to-end latency, images/groups per second, encoder and
decoder timings, training seconds/update, peak VRAM, host RSS, disk/decode
throughput, and model/checkpoint size. Include selection, packing, scatter,
transfer and calibration overhead. Specify batch size, view count, retained
tokens, resolution, precision, kernel/backend, warmup and synchronization.

Report dense keyframe or refresh costs if temporal state is later added. For a
sparse stateful system, give amortized sequence cost, worst refresh latency,
staleness and sequence length; a steady-state sparse frame alone is insufficient.

Optional power measurements can support energy per processed group. GPU-hours
alone do not establish energy or carbon without the corresponding measurement
and assumptions. Report training/generation cost even if inference is efficient.

## 5. Minimum experiment matrix

All models in the matched main table use the same imported pretrained checkpoint,
data regime, target masks, transforms, decoder capacity and readout unless the
row explicitly changes that factor.

| ID | Variant | Priority / purpose |
| --- | --- | --- |
| B0 | Frozen V-JEPA features, no learned fusion | Required foundation baseline; no fusion-training cost |
| B1 | Frozen V-JEPA + CroCo reconstruction decoder | Required cross-view objective control |
| B2 | B1 + MAE reconstruction | Required extra-task/compute control |
| B3 | B2 + RI loss and prediction | Required proposed two-view model |
| C0 | B3 with no reference at evaluation | Required reference-use diagnostic |
| C1 | B3 with scene-shuffled/unrelated reference | Required shortcut/false-match diagnostic |
| M0 | Continued pair training + uniform feature pooling | Required control for set continuation and extra training |
| M1 | Same pair model + RI-weighted pair pooling | Required low-complexity multi-view baseline |
| M2 | Joint set reconstruction/RI + sampled pair objective | Required multi-view candidate |
| SP0 | M2 dense encoding and fusion | Dense operating-point reference |
| SP1 | M2 dense encoding then gather | Required separation of fusion sparsity from encoder savings |
| SP2 | M2 sparse contextual encoding at same selected inputs | Required sparse-execution comparison |
| SP3 | Mixed-budget-trained M2 and dense continued-training control | Required only if selected as the sparse model claim |
| A0 | Paper Eq. 8 vs released epsilon formulation | Small numerical/behavior ablation |
| A1 | Native-image vs repeated-frame video mode | Optional encoder-mode ablation |
| A2 | Random-initialized matched encoder or small scratch Gekko | Optional attribution of pretrained knowledge; cost-limited |
| A3 | S0 RGB sampling vs S1 geometry-curated sampling | Prioritized if overlap distribution limits training |
| A4 | Last-two-block encoder adaptation | Optional after frozen-model diagnosis |
| A5 | Random/uniform/texture/learned selection, oracle bound | Optional routing study after simple sparsity works |
| A6 | Joint set without auxiliary pair term | Useful set-objective ablation if budget remains |

Objective rows without RI training have no trained RI head. Do not compare their
random extra channel with B3. Use a fixed descriptor-match confidence or applicable
reconstruction-error proxy for visibility baselines, and the identical descriptor
readout/frozen supervised probes for representation comparisons. A matched
supervised visibility probe on all backbones is a separate probe experiment.

Add released CroCo/Gekko, MuM/Muskie and another strong geometric model only when
artifact availability and compatible protocols are verified. They belong in an
**external context table** with pretraining data, trainable parameters, inference
resolution and supervision disclosed. They are not matched-causal controls.

## 6. What “fair comparison” means here

Run distinct analyses rather than claiming one budget matches everything:

- **Equal examples/updates:** same learning opportunity and paired random stream;
  report differing GPU-hours for extra objectives.
- **Equal measured GPU-hours:** learning curves compared at common compute
  budgets; report differing numbers of examples and updates.
- **Equal total reference tokens:** distribute the same observation budget across
  different numbers of reference views.
- **Equal per-view resolution:** allow more views to add information and compute;
  show both changes.
- **Equal probe capacity and labels:** train the same downstream head protocol on
  each backbone; distinguish frozen and end-to-end tuning.

Count all shared pretraining and continuations in model cost. A warm-started set
model cannot omit its pair-training cost while the pair baseline includes it.
Use the same number of hyperparameter-selection opportunities where practical,
and retain unsuccessful trials in the ledger.

## 7. Stratification and failure analysis

Report performance by directional overlap, baseline, parallax, view count,
reference-token budget, room layout, visible object class, texture, lighting,
glass/specularity, boundary distance, depth, and image resolution. Predeclare a
small primary stratum set so exploratory slicing does not become selective
reporting. Publish support counts and prevalence in each stratum.

Counterfactual suites include identical references, duplicate views, unrelated
rooms, adjacent views with tiny baseline, opposing cameras, occluded furniture,
large blank walls, repeated chairs, specular displays, glass partitions, dark
corners, extreme FOV, image corruption and missing references. Later dynamic
experiments add moving people and asynchronous timestamps.

Specific failure questions: Does RI confuse low reconstruction error with low
visibility? Does sparse texture selection abandon blank but co-visible walls?
Does set fusion become overconfident when several references are redundant?
Does a model exploit generator palette/layout regularities? Do calibration and
coverage fail before descriptor accuracy noticeably changes?

## 8. Statistics and selection discipline

Use three independent training seeds for selected confirmatory comparisons.
Report each seed plus mean/variation. Use paired resampling of held-out scenes
for differences; for aggregate uncertainty across runs, use a documented
hierarchical bootstrap over training seeds and scenes. Millions of pixels from
one room are not millions of independent observations.

A proposed 95% interval with 2,000 bootstrap replicates is adequate for the first
analysis tool; record RNG and aggregation choices. Multiple confirmatory
comparisons need a predeclared family-wise or false-discovery policy, or clearly
separated primary and exploratory tests. Three seeds give limited precision for
training variability, which should remain a stated limitation.

Select thresholds, calibration and model variants using validation only. Reserve
the test set for the locked analysis; do not silently modify the model after test
inspection and reuse the same result as confirmation. Bug fixes invalidate
affected comparisons and require rerunning all relevant rows.

## 9. Evaluation artifacts

Every report includes model/dataset/protocol hashes, prediction coordinate frame,
sampling/regime identity, numerical precision, sample/scene counts, invalid and
failure counts, per-scene metrics, aggregate metrics, seed results, uncertainty,
runtime metadata and calibration identity.

Generate PR curves, reliability plots, quality/latency frontiers, overlap-stratum
plots, learning curves and fixed-selection qualitative panels directly from those
artifacts. Store raw predictions for an audit subset and all reproducible summary
rows. Paper tables must cite the exact run/protocol IDs that generated them.
