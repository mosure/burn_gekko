# Pilot 17: real-view localization and pose consensus

Status: both CPU diagnostics complete. No new model training.

Pilot 16 found a synthetic camera benefit from actual-view geometry training,
while Pilot 14 found unstable real-view camera transfer. Before another training
continuation, this study checks whether coordinate refinement or camera fitting
obscures useful matching information. The old GPU allowance has 96.281 seconds
left; this diagnostic consumes no GPU time. A renewed training ceiling is pending.

## Fixed protocol

Use the unchanged Pilot 13 geometry checkpoint's cached RGB predictions for all
186 TUM development pairs, all three descriptor readouts, both temporal intervals,
and solver seeds 781 through 788. Retain the original eight-point RANSAC policy:
2,048 maximum / 64 minimum trials, confidence 0.999, 12 minimum inliers,
3 original-image-pixel Sampson threshold and 1 cm baseline eligibility.

Compare two coordinate readouts: the recorded 3×3 local probability centroid,
and the hard patch center already present in the same prediction. Keep the exact
same mutual-match population. Recompute every fit for both readouts and every
seed; verify that recorded coordinates reproduce the original seed's inliers,
trials, failure state and angles. Report all outcomes without selecting a seed,
coordinate rule or checkpoint. Failed fits contribute 180 degrees. Camera metrics
retain equal pairs within sequence, then equal sequences and equal solver seeds.
Also retain summaries for both frame intervals.

After fitting on every RGB mutual match, evaluate its residual to the ground-truth
essential matrix. Report the fraction consistent at the unchanged 3-pixel
threshold, and the corresponding fraction within the fitted RANSAC consensus.
These are post-inference diagnostic labels, never a filter for solver input.
Exclude low-baseline pairs from epipolar diagnostics and retain them in rotation
and solver-failure reporting. Pair-weighted epipolar statistics have a different,
explicit aggregation from the sequence-macro camera metrics.

Compute the probability of at least one eight-point sample containing only
epipolar-consistent matches at the maximum trial count, with sampling without
replacement inside each trial. This is a diagnostic of sampling opportunity,
not a probability of correct pose: epipolar consistency cannot detect errors
along the epipolar line, and the solver may terminate before the maximum budget.

This uses repeatedly inspected development sequences. Any improvement is a
coordinate/solver diagnostic on the same weights, not a learned-model gain or
independent real-data qualification. It does not change the published model.

## Reproduction

```sh
cargo run --locked --profile pilot -p burn_gekko_eval --bin gekko-eval -- \
  pose-localization --config configs/eval/pose-localization-pilot17.toml
```

Rust owns the analysis and TOML owns its configuration. Outputs, upstream hashes,
all 16 solver reports, per-match-population summaries and consensus diagnostics
are stored under `.data/pilot-17/eight-point/` and `.data/pilot-17/five-point/`.
The initial coordinate-only run remains in `.data/pilot-17/localization/`.
Existing exports remain unchanged. After scoring, the same command with
`--report-only` verifies the complete evidence closure and writes `head.json`
for the native page/paper builder, without fitting again.

## Coordinate diagnosis

The initial native run completed in 32.68 CPU-command seconds. The eight-seed
aggregate reproduces the existing pose sensitivity results; the original-seed replay
check passes. These results retain all 186 pairs / 185 pose-eligible pairs.

| Same weights and mutual matches | Recorded subpixel coordinates | Hard patch centers |
|---|---:|---:|
| Pair-conditioned mean pose AUC@10 | 7.99% | 4.28% |
| AUC@10 range across solver seeds | 6.29–10.04% | 4.07–4.65% |
| Mean rotation error, failures included | 7.94° | 57.62° |
| Mean signed translation-direction error | 51.71° | 96.19° |
| Matches consistent with ground-truth epipolar geometry | 40.07% | 26.45% |
| RANSAC consensus consistent with ground-truth epipolar geometry | 48.35% | 28.69% |

The same-image and encoder recorded-coordinate readouts have AUC@10 of 6.79%
and 7.27%, respectively. Their ground-truth epipolar-consistent fractions are
36.95% and 33.96%. Pair conditioning improves those necessary geometric checks,
but they remain weak. Subpixel refinement is helpful; deleting it is not a fix.

## Five-point follow-up registered before outcomes

The observed match population motivates a calibrated minimal-solver check.
Use the same fixed inputs, all three readouts, both coordinate rules, all eight
seeds, threshold, maximum/minimum trials, inlier requirement and signed cheirality.
Change only the minimal hypothesis generator from eight to five points, and
the corresponding RANSAC stopping exponent. Score every returned candidate
against all mutual matches. Retain the same optional final eight-point consensus
refit. No local optimization, threshold sweep, label-filtered matches or model
training is added. These are equal maximum trial budgets, not equal CPU time.

The native adapter pins the MIT-licensed
[vision-geometry 0.9.0 solver](https://docs.rs/vision-geometry/0.9.0/vision_geometry/epipolar/index.html).
It rejects nonfinite/rank-deficient samples and checks essential constraints and
sample residuals before accepting polynomial roots. The unchanged default remains
the eight-point solver, preserving historical reports. Both noisy nonplanar
fixtures with outliers and degenerate samples must pass before the real-data run.

`configs/eval/pose-five-point-pilot17.toml` writes to `.data/pilot-17/five-point/`.
The final binary also repeats the eight-point control in `.data/pilot-17/eight-point/`
to verify that adding the solver preserves existing outputs. The initial
`.data/pilot-17/localization/` study remains immutable. Every method, seed and
failure will be retained regardless of the result. A solver gain is not evidence
of new learned capability or a SotA foundation model.

## Five-point results

| Recorded subpixel readout, same model | Eight-point mean AUC@10 | Five-point mean AUC@10 |
|---|---:|---:|
| Pair-conditioned | 7.99% | **14.19%** |
| Same-image | 6.79% | 12.88% |
| Encoder | 7.27% | 12.33% |

The pair-conditioned five-point result ranges **11.71–17.46%** across the eight
seeds. Every seed improves over the eight-point result; the entire five-point
range is above the old range. Mean rotation error falls from **7.94° to 5.46°**,
and signed translation-direction error from **51.71° to 35.50°**. The improvement
holds in the mean of each sequence: long-office **5.67% → 14.86%**, structure-far
**8.35% → 12.24%**, structure-near **9.97% → 15.49%** AUC@10. Both registered
frame intervals improve: 15-frame **4.93% → 7.80%**, 60-frame **12.21% → 21.95%**.
These are sensitivity means on reused data, not statistical confidence intervals.

The five-point solver does not make hard patch centers competitive: their
pair-conditioned AUC@10 is **5.51%**, versus 4.28% with eight-point fitting.
The frozen subpixel matches' epipolar-consistent fraction remains exactly 40.07%.
Their consistency within the selected RANSAC consensus rises only to **49.88%**.
Better minimal hypotheses recover more usable poses, while inaccurate matches
and incorrect consensus remain substantial. The 60-frame result is consistent
with better translation observability, but this temporal split does not isolate
baseline from all other image changes.

Five-point pair conditioning passes the historical all-sequence/success gate
against same-image descriptors in **2/8** seeds, and against encoder descriptors
in **3/8**. Both hold together only for seed 786 (**1/8**). No seed is selected.
The original fusion weakness remains; this result must not be called resolved
transfer, improved intrinsic prediction, improved RGB reconstruction, or SotA.

The final eight-point command takes **33.15 seconds**, and the five-point command
**143.56 seconds**, for all six descriptor/coordinate combinations and eight
seeds. These are wall times of CPU-only commands, with other workspace checks
running during part of the study; they are not controlled solver speed benchmarks.
GPU command time for this entire study is **zero** and the prior allowance remains
**96.281 seconds**. No new training allowance has been inferred.

## Implication for the next training study

Keep subpixel localization. The proposed full-cohort geometry continuation remains
fixed, including its already registered synthetic retention gates. Do not alter
those gates using this real-data result. For a future independently registered
camera qualification, use the calibrated five-point solver as a declared protocol
and retain the eight-point diagnostic for continuity. Finer spatial localization
and match reliability are justified next training targets; a solver-only change
does not supply missing image detail or repair the learned calibration head's
generalization. Longer foundation or feature-export runs still require the pending
compute-ceiling answer.

The updated single-checkpoint [project page](../../.data/publications/pilot17-camera-diagnostics/index.html)
and [PDF](../../.data/publications/pilot17-camera-diagnostics/paper.pdf) retain the
actual RGB PSNR, annotated samples and learned-camera metrics from the fixed head
study. Separate calibrated-solver capability tables carry the new results and
their full hashed evidence closure. No differently trained private model is
substituted or compared in the publication.

## Verification

The workspace test run passes 174 tests; the final evaluator suite additionally
passes the new multi-axis polynomial-root and publication-tampering tests
(41 evaluator unit tests total). CPU workspace and CUDA trainer strict Clippy,
owned-crate formatting, encoder import provenance, and all eight report tests
pass. Geometry tests cover exact/noisy nonplanar scenes, outliers, deterministic
replay, signed pose, degenerate samples and extreme/nonfinite inputs. A deliberately
changed capability source checksum is rejected before creating a publication.

Native publication validation checks **120 file hashes, 103 decoded images and
48 local links**. All six sample choices pass at 390/768/1440-pixel browser widths,
without overflow, missing images or JavaScript errors. Chromium runs with GPU
disabled. The 35-page PDF's new camera tables were visually inspected. CI and
deployment results are recorded separately after source closeout.
