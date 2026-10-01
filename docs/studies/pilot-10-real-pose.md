# Pilot 10: independent real-image camera-motion qualification

Registered 2026-09-30 before model inference or scoring on this cohort. The user
requested continued experimentation. Until another allowance is explicitly set,
this study uses at most **536.8263035626151 GPU-command seconds**, the remainder
of the preserved Pilot 08/09 allowance. CPU implementation, dataset preparation,
scoring and publication do not consume GPU time. No automatic budget extension.

## Frozen candidate and question

Freeze the Pilot 09 final checkpoint
`a02277687d0f10cebd7bfe9e487facc6c24d461ea67a377e29b8f04b495d7185` and its 256-pixel
input, block-6 spatial route, reciprocal temperature 0.07 and local 3x3 centroid.
Does its pair-conditioned descriptor recover relative camera motion on previously
unused real RGB images, beyond equally refined same-image and encoder controls?
There is no new training, checkpoint selection or threshold sweep in this study.
The report uses this one training run and checkpoint; it is an additional
qualification of the same model, not a comparison of private model versions.

## Dataset and pair protocol

Use the TUM RGB-D Freiburg 3 `long_office_household`, `structure_texture_far` and
`structure_texture_near` sequences. These have not appeared in prior local
training, selection or evaluation. All three share one acquisition environment,
so this is a small independent diagnostic, not a broad real-world benchmark.
Download the official archives into `.data/benchmarks/tum-rgbd/` and record hashes.
TUM data is CC BY 4.0. Cite the dataset and retain its original filenames.

For each complete timestamp-ordered RGB sequence select 32 uniformly spaced
anchor indices from zero through `frame_count - 61`, inclusive. Pair each with
the images 15 and 60 frames later: 64 pairs per sequence, 192 intended pairs.
Do not filter pairs using images, geometry, overlap or model quality. Associate
each image with its nearest ground-truth camera pose within 20 ms. Record any
association exclusions without replacement. Pose is not supplied to inference.

Freiburg 3 RGB images are already undistorted. Use published RGB calibration
`fx=535.4, fy=539.2, cx=320.1, cy=247.6` in the original 640x480 frame. Decode RGB
in Rust, resize to 256x256 using the image crate's triangle filter, and cache HWC
float32 RGB in [0,1]. Bind both original and resized files by checksum. Keep
camera labels in a separate manifest rejected by the RGB-only inference schema.

## Calibrated geometric probe

All three descriptors use the identical predeclared local readout and mutual
hard-match mask. Supply the resulting sparse patch-center correspondences and
known intrinsics to a separate CPU pose solver. This is a **calibrated geometric
probe**, not a learned camera/intrinsics head, a metric-scale predictor or a SLAM
system. Keep the untrained camera-head capability explicitly separate.

Use a native normalized eight-point essential-matrix estimator, rank-two/equal
singular-value projection, deterministic RANSAC (2,048 maximum trials, 64 minimum,
0.999 confidence), a three-original-pixel Sampson threshold converted using mean
focal length, and an all-inlier refit. Require at least 12 inliers. Decompose the
essential matrix and choose its rotation/translation sign by positive-depth
triangulation. The solver sees no ground-truth pose or depth. Test exact geometry,
nonplanar noisy outliers, degeneracy and camera-frame conventions before scoring.
This eight-point implementation is not claimed to match published five-point
pose protocols.

Report signed rotation/translation-direction errors in degrees, pose AUC at
5/10/20 degrees, recall at those thresholds, solver success and correspondence /
inlier counts. Failures contribute 180 degrees to applicable error/recall/AUC
metrics; never report only successful pairs. Baselines below 1 cm are excluded
from translation/pose metrics and counted explicitly, while rotation remains.
Use equal pair weight within sequence then equal sequence weight. With only
three related sequences, intervals cannot establish broad generalization.

Predeclared transfer gate: pair-conditioned pose AUC@10 must exceed both controls
in each of the three sequences, with no lower overall solver success fraction.
Regardless of outcome, retain every pair and failure reason, deterministic
first/middle/last pair visualizations, exact preprocessing and solver parameters,
power/time receipts and a generated page/PDF. No tuning on this outcome. Any
subsequent use for model/readout development must explicitly relabel this cohort.

Primary references:
[TUM dataset and license](https://cvg.cit.tum.de/data/datasets/rgbd-dataset),
[file formats and camera calibration](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats),
[official downloads](https://webshare.cvg.cit.tum.de/g/rgbd/dataset/).

## Completed result

All 186 eligible pairs completed inference with the frozen checkpoint. Six of
the 192 proposed pairs lacked a pose association within 20 ms; no replacements
were chosen. One near-sequence pair has a baseline below 1 cm and is excluded
only from translation and pose metrics. Each method returns 184 poses; both
failures remain in the angular errors, recall and AUC denominators.

| Same-checkpoint readout | Mean rotation error | Mean translation-direction error | Pose AUC at 10 degrees | Poses within 10 degrees |
| --- | ---: | ---: | ---: | ---: |
| Pair-conditioned local | 8.03 degrees | 52.71 degrees | 7.88% | 20.03% |
| Same-image local | 8.99 degrees | 58.20 degrees | 5.72% | 14.58% |
| Encoder local | 8.75 degrees | 56.59 degrees | 4.93% | 13.57% |

The pair-conditioned readout improves AUC at 10 degrees over the encoder in
every sequence. It loses to the same-image control on structure-far: 6.15%
versus 9.53%. **The registered all-sequence transfer gate fails.** This result
supports useful camera-motion signal but does not establish dependable motion
recovery or SOTA. A returned pose is not necessarily an accurate pose.

GPU inference used 29.493 seconds, leaving 507.333 seconds of the old allowance;
the separately authorized Pilot 11 allowance is unaffected. Dataset preparation,
normalized eight-point RANSAC, scoring, validation and publication are native CPU
Rust. The report reuses the previous 512-target completion evidence unchanged.
No model, readout, solver threshold or pair selection was adjusted after scoring.

Reproduce preparation with `gekko-eval prepare-tum --config configs/data/prepare-pilot10-tum.toml`, score the pinned RGB-only predictions with
`gekko-eval pose --config configs/eval/pose-pilot10.toml`, and generate the local
page/PDF from `configs/publish/pilot10-real-pose.toml`. Artifacts live under
`.data/pilot-10/`; the review bundle is
`.data/publications/pilot10-real-pose/`. The 19-page PDF has annotated predicted
matches with RANSAC inlier labels, not claimed ground-truth correspondences.
Browser verification covers all six completion samples at 390, 768 and 1440 px,
with no broken images, JavaScript errors or horizontal overflow.

Pilot 11 is motivated partly by this failure. Any later reuse of these three
sequences is therefore **development evaluation**; their original frozen result
remains the first locally independent assessment. Broader independent datasets,
training-seed replication and protocol-matched public baselines remain open.
