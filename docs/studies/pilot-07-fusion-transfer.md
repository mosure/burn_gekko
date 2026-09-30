# Fusion transfer: positional audit and controlled training

Completed bounded study, 2026-09-29 local time. The
[54-page annotated final PDF](../../.data/pilot-07/fusion-audit/fusion-transfer-final-report.pdf)
contains controlled training, independent evaluation, latent maps, real-image
matches, and measured GPU efficiency. Transfer improves substantially, but
fusion still trails the stronger encoder baseline. The earlier reports remain
immutable.

## Independent outcome

The selected checkpoint is `route-balanced-continue`, SHA-256
`61ecd81f7d9c4094ff1fa120e761176301db14e4a22e056cc966bf8616a81bf6`.
The [selection record](../../configs/archive/pilot-07/pilot07-fusion-final-selection.toml) freezes
its weights and all 26 readouts at 2026-09-30 02:44:03 UTC, before either
independent set is evaluated. Its audited ancestry contains 17,200 optimizer
updates, 275,200 target exposures and 8,192 unique training rooms.

All 3,365 ETH3D interval pairs, 10 scenes and seven intervals are evaluated in
original image coordinates. Lower AEPE is better.

| Readout | Original fusion AEPE | Selected fusion AEPE | Strong own encoder AEPE | Encoder minus fusion, 95% scene interval |
| --- | ---: | ---: | ---: | ---: |
| Raw decoder | 53.765 | 39.161 | 38.056 | -1.105 [-1.815, -0.386] |
| Conditional decoder | 57.328 | 38.068 | 36.748 | -1.321 [-2.041, -0.596] |
| Conditional attention | 100.826 | 40.298 | 36.748 | -3.550 [-5.856, -1.292] |

The first encoder control is centered block-6 cosine; the other two apply the
same reciprocal conditional normalization to that feature level. Paired scene
intervals also show lower PCK3 than the corresponding strong encoder in all
three fusion readouts. The gains against the original fusion are real, but
neither weaker final-layer controls nor a better headline AEPE establish a
fusion benefit over the strongest preselected encoder. This independent test
does not establish SOTA or published refinement parity.

The fresh 128 four-view rooms provide 512 targets under the fixed 90% mask and
two-reference protocol. Latent MSE falls from 0.227313 to 0.207651 (8.65%). The
paired room-bootstrap improvement is 0.019662 [0.018863, 0.020469]. Learned RI
co-visibility AUROC rises from 0.6227 to 0.6755. Related references beat the
monocular branch, unrelated-room references and shuffled token positions, with
positive room-bootstrap intervals. RI uses a separate full-target branch and
is a ranking score, not a calibrated visibility probability.

On the fixed first 32 rooms, one/two/three references give selected-model MSE
0.214724/0.207617/0.204532. The architecture therefore operates beyond two views
without refitting, although one-reference benefit over the monocular branch is
small. Hidden-target RGB and monocular/reference isolation differences are
exactly zero. Reference-order differences remain nonzero under float32 CUDA
(candidate maximum 0.002661, RMS 0.000156); the strict 1e-5 bound does not pass.

The study ends with 37,240.67 of 43,200 GPU-command seconds charged, including
failures, data preparation, evaluation and profiling. No GPU job or owned
monitor remains active. Validation includes 96 Rust workspace/all-target tests,
strict CUDA Clippy, native execution parity, fixed-feature controls and
checksummed independent selection. CPU reporting is outside the GPU ceiling.
No accuracy tuning follows these independent outcomes in this study.

## What the trunk knows about position

The encoder retains its per-image spatial encoding. The six-block fusion decoder
uses 2D rotary encoding in self-attention and cross-attention; every reference
reuses its own image grid. References share a role embedding and form an unordered
set. The model receives no camera intrinsics, extrinsics, rays, depth or 3D points.
This is an RGB-only architecture with valid image coordinates, not a calibrated
multi-view coordinate system. Identical coordinates across views need not describe
the same scene point.

[DPPE](https://arxiv.org/html/2606.31585v1) analyzes camera-based transforms in
attention and value aggregation. Its rotation/translation coupling diagnosis
concerns a mechanism absent from this decoder: our values are not camera-rotated.
Adding a DPPE-style path would require supplied or predicted cameras and a separate
evaluation of that input contract. The present experiment isolates image RoPE
before adding such a path.

An auxiliary camera predictor is a reasonable subsequent experiment. It should
predict relative pose in a declared gauge, account for translation scale ambiguity,
and be evaluated with predicted cameras at inference. Intrinsics need meaningful
variation and comparison with a constant-prior baseline. A metadata-only audit of
the first 32 training rooms found vertical FOVs of 28.2–106.2 degrees, so this cache
does contain focal variation. That sample does not establish real-camera transfer.
The captured geometry is represented by a centered perspective camera with
vertical FOV and image size. The present square captures do not vary principal
point or aspect ratio; a broader intrinsics head would need corresponding
calibrated crop/aspect augmentation and tests outside that camera family.
Intermediate supervision would require an explicit gradient schedule and an
ablation against the same RGB-only model; low camera loss alone would not prove
better correspondence.

For a concrete follow-up, predict focal length relative to image size, a
continuous rotation representation, and relative translation with an explicit
scale convention. Fit heads first on detached features, then measure a ramped
auxiliary gradient into the trunk. Compare an auxiliary-loss-only variant with
one that actually conditions attention on predicted cameras. Any camera head
feeding masked completion must use the sparse target and references; letting it
see the full target would create an unintended route around the mask. Renderer
pose targets would also make this a separately declared geometry-supervised
variant, rather than the current RGB-only self-distillation study.

[ZipSplat](https://arxiv.org/html/2606.05102v1) uses optional camera conditioning and
selectively detaches a geometric loss for Gaussians already contributing to the
render. This motivates controlled use of geometry, but is not a direct recipe for
our proposed camera head. Its pretrained backbone and training regime are also
different. No weights or implementation from either paper enter this study.

## Frozen checkpoint diagnosis

Checkpoint: `c83bc1fc4d6fdd7b57f63eee24d85cdd1e41e93ca377c489e122b0e65685b64b`.
All 59 HPatches viewpoint sequences / 295 pairs; 256-pixel model input, 240-pixel
metric coordinates. Hard patch matches with bilinear displacement upsampling.

| Readout / intervention | Viewpoint AEPE, lower is better |
| --- | ---: |
| Centered student encoder | 31.010 |
| Student encoder | 32.545 |
| Fixed teacher | 34.079 |
| Centered fusion decoder | 35.732 |
| Original fusion decoder | 36.651 |
| Centered decoder, cross-view RoPE disabled | 38.901 |
| Original reciprocal probability attention | 43.322 |
| Reciprocal mean attention logits | 43.634 |
| Unrotated, centered content logits | 62.235 |

The full 68-readout diagnostic family is retained. Switching probabilities to
logits, selecting individual layers, or removing cross-view RoPE at inference
does not recover encoder-level correspondence. The content scores are themselves
poor. This supports a learned fusion/objective mismatch rather than a readout-only
fix. It does not prove that cross-view RoPE should be removed during training:
that intervention changes the distribution seen by an already trained model.

The native trace reproduces the ordinary decoder and attention maps on CPU.
CUDA descriptor argmaxes can differ slightly near ties when extra trace kernels
change numeric dispatch: the traced raw decoder AEPE is 36.653 rather than 36.651.
The reciprocal probability readout is unchanged at reported precision.

A label-only resolution diagnostic projects the true homography at the same
16x16 patch centres, quantizes to the nearest reference cell and applies the
same displacement interpolation. It gives 4.549 px viewpoint AEPE, substantially
below the model errors. This uses ground truth and is unavailable to inference;
it is not a rigorous lower bound because interpolation can couple neighboring
discrete choices. It shows that the present regression cannot be explained by
the patch grid alone. The saved artifact is
`.data/pilot-07/fusion-audit/hpatches-grid-diagnostic.json`.

## Training intervention

[Registered protocol](../../configs/archive/pilot-07/pilot07-fusion-protocol.toml): four matched arms,
600 updates each, batch 16, 1,024 rooms, same ancestry, examples, masks and schedule.
Both AdamW optimizers restart; the encoder stays frozen during this factorial
screen. The factors are cross-view RoPE on/off and dense semantic guidance on/off.
Self-attention and monocular-branch RoPE remain enabled in all arms.

Masked V-JEPA latent prediction remains the primary loss. The guided arms add:

- Mean layer KL from the head-mean cross-attention scores to centered cosine
  affinities from a frozen copy of the audited warm-start student encoder.
- Dense latent preservation through the existing prediction head on an unmasked
  target/reference pair.

Each auxiliary term has weight 0.1. Teacher affinity temperature is 0.07. The
auxiliary teacher stays fixed even across exact optimizer resume. Its features
are semantic soft targets, not ground-truth correspondences or co-visibility.
Full target features are accepted only in the separate dense auxiliary branch;
the sparse completion path cannot read them. No new model parameters, camera
inputs, renderer labels or noncommercial pretrained weights are introduced.

This also addresses an information-set mismatch: completion is trained with a
sparse target and two references, while matching uses two full images. The dense
pair objective exposes the fusion trunk to its matching-time input pattern.

The completed factorial screen favors retaining image RoPE:

| Arm | Validation latent MSE | Decoder AEPE | Centered decoder AEPE | Reciprocal conditional attention AEPE |
| --- | ---: | ---: | ---: | ---: |
| Unchanged objective / RoPE | 0.21612 | 36.506 | 35.780 | 43.229 |
| Unchanged objective / no cross-view RoPE | 0.22144 | 39.242 | 37.067 | 70.206 |
| Semantic guidance / RoPE | 0.21800 | 37.845 | 37.485 | **30.598** |
| Semantic guidance / no cross-view RoPE | 0.22258 | 37.188 | 36.804 | 35.999 |

The frozen centered student encoder scores 31.010 in every arm. Guidance with
RoPE reduces attention error by 29.2% against its matched control, with 0.87%
relative latent-error regression. However, decoder matching remains worse than
the encoder. This is a partial repair, not a completed quality qualification.
The guided arm's learned RI AUROC also drops from 0.675 to 0.645, while the actual
latent error-improvement proxy is essentially unchanged (0.6222 versus 0.6229).

The reciprocal conditional readout averages log-softmax scores across layers
and the two directions. Unlike raw reciprocal logits, it is invariant to an
arbitrary row offset, which row-wise KL training cannot identify. Both the old
probability readout and all alternative standard readouts remain reported.

The v14 assessment also applies this same reciprocal normalization to raw and
centered encoder and decoder cosine scores. Its feature temperature is fixed at
0.07, matching the already registered auxiliary affinity temperature. This
control was registered before scoring the refinement endpoints or either new
holdout: attention-versus-plain-cosine gains alone do not distinguish a better
representation from a better readout. All twelve earlier readouts remain, with
six additional feature controls. Training and the sealed v13 continuation
binary are unchanged by this evaluation extension.

## Direct descriptor alignment

A second matched screen starts from the guided checkpoint, uses all 8,192 cached
training rooms and 64 validation rooms, and trains 600 updates in both arms.
Both arms use bidirectional dense pairs and keep the encoder frozen. The only
difference is a weight-0.1 symmetric KL loss on cosine similarities between the
two fused decoder feature sets. Its stopped targets are the same frozen ancestor
encoder affinities; it does not introduce geometric supervision.

The prediction head can preserve information in a transformed metric without
preserving nearest-neighbor similarity in its input descriptors. This experiment
therefore supervises the descriptor metric directly, rather than assuming dense
prediction MSE will do so. A centered prediction-head readout is also recorded to
distinguish feature loss from a metric mismatch. The objective is checked for
gradient flow to both descriptor branches and detachment of teacher targets.

The screen uses a weights-only warm start and fresh optimizers. All 600 updates
completed in each arm. A checksummed sample audit confirms identical 9,600 target
exposures, covering 6,333 of the 8,192 rooms, with identical learning rates. The
descriptor weight is the sole configuration difference.

| Arm | Validation latent MSE | Decoder AEPE | Centered decoder AEPE | Conditional attention AEPE | Learned RI AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bidirectional guidance control | 0.21677 | 36.590 | 36.255 | 29.761 | 0.691 |
| Direct descriptor alignment | 0.21789 | **33.947** | 34.237 | **29.363** | 0.642 |

The descriptor treatment reduces decoder AEPE by 7.2% and attention AEPE by 1.3%
against the matched control. Its 0.52% relative latent-error increase meets the
registered 5% limit. However, decoder matching still trails the unchanged
centered encoder's 31.010, and the learned RI ranking regresses. The underlying
latent error-gain AUROC changes only from 0.635 to 0.631. Centering the prediction
head does not repair the remaining decoder gap (34.625 AEPE for the treatment).

A subsequent matched phase starts from the descriptor checkpoint. Both arms
completed 2,000 updates on the full room pool, with a 6,000-update learning-rate
horizon, peak LR 0.0001, and RI active from the first update. One keeps the encoder
frozen; the other opens the last two blocks and then all image blocks/stem after
at least 400 updates and 0.2% measured latent improvement per stage. The encoder
LR ratio remains 0.05. This measures whether more training and progressive
adaptation close the decoder gap and recover RI; it does not assume they will.

The adaptive arm passed both gates, with 400 frozen, 400 partially unfrozen and
1,200 fully unfrozen updates. The first and last encoder attention-weight changes
were 0.000425 and 0.002643 respectively; the teacher stayed unchanged. Both arms
visited all 8,192 rooms and all 24,576 room/target pairs, with exactly the same
32,000 target exposures. Each endpoint's complete ancestry contains 6,200 updates
and 99,200 exposures. The matched control review checks both optimizer records,
sample order, learning rates, stage gradients and actual parameter updates.

| Refinement arm | Validation latent MSE | Monocular MSE | Spatial variance ratio | Learned RI AUROC |
| --- | ---: | ---: | ---: | ---: |
| Frozen encoder | 0.20964 | 0.22830 | 0.3461 | 0.6564 |
| Progressive encoder adaptation | 0.20898 | 0.22748 | 0.3484 | 0.6540 |

Adaptation reduces latent MSE by only 0.31% against the matched control, with a
small RI decrease. Median measured throughput is 20.7 targets/s for the frozen
run and 15.4 targets/s during the adaptive run's fully unfrozen stage. CPU
engineering/build checks ran concurrently with parts of that stage; these are
observed study timings, not an isolated backend benchmark. Stable process GPU
memory is about 20.6 / 22.9 / 34.8 GiB for the frozen / last-two / full stages.

The completed common v14 export scores all 116 HPatches sequences and all 18
readouts. Every immutable teacher match and every frozen-encoder match is
identical across the compared checkpoints; the adaptive encoder is allowed to
change. Primary viewpoint results are:

| Model | Centered encoder | Conditional centered encoder | Raw decoder | Conditional decoder | Conditional attention |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original continuation | 31.010 | 30.232 | 36.653 | 36.005 | 43.371 |
| Descriptor screen | 31.010 | 30.232 | 33.947 | 33.073 | 29.363 |
| Refine frozen | 31.010 | 30.232 | 33.312 | 32.342 | 29.019 |
| Refine adaptive | 30.927 | 30.071 | 33.108 | 32.233 | 28.946 |

Adaptive versus frozen raw decoder improvement is 0.204 px, with a descriptive
59-sequence interval [0.054, 0.352]. The attention difference is only 0.073 px,
with interval [-0.030, 0.189]. Adaptive attention is better than its equally
normalized encoder by 1.124 px in the mean, but that interval includes zero
[-0.240, 2.322]. Thus the large repair versus the old fusion is supported, while
encoder-beating attention remains uncertain and decoder regression remains.

A separate diagnostic samples the 16x16 patch centres directly, using a 15-pixel
correctness radius in HP-240 coordinates. Relative to the conditional encoder,
the refined conditional decoder corrects 3.81% and damages 7.05% of valid queries
on average across pairs. Its PCK15 is 53.37%, versus the encoder's 56.62%.
Attention reduces mean distance but also has lower PCK15 (54.05%). These are
patch-centre diagnostics, not the interpolated dense benchmark metric. The
analysis retains both accuracy and outlier effects rather than treating a
lower mean error as sufficient evidence of better precision.

## Descriptor-strength screen

A subsequent registered matched experiment starts from the adaptive refinement
checkpoint. Both arms run 1,000 updates on the complete room pool, with the same
fresh optimizers, 4,000-update learning-rate horizon, peak LR 0.0001 and RI active
immediately. The sole treatment is descriptor affinity KL weight 0.1 versus 1.0;
attention and dense preservation remain 0.1. All screen updates freeze the
encoder. A gate at the end can allow later exact continuation to unfreeze the
last blocks and then the full encoder, after 1,000 updates per stage and 0.2%
latent improvement. This tests whether descriptor preservation is underweighted
without changing positional encoding or introducing camera supervision.

Selection requires improvement in raw and equally normalized decoder viewpoint
AEPE, at most 2% relative attention regression and at most 5% latent regression
against the matched control. PCK, RI, variance and encoder comparisons remain
explicit. The new independent sets are still reserved.

Both arms completed all 1,000 updates in 19.1 and 19.2 minutes respectively.
They used identical inputs, masks, learning rates and actual encoder stages.
The sole configuration difference was the descriptor weight. Each endpoint's
audited ancestry contains 7,200 updates and 115,200 target exposures, with all
8,192 training rooms represented and no validation/test optimizer inputs.

| Descriptor weight | Latent MSE | Raw decoder AEPE | Conditional decoder AEPE | Conditional attention AEPE | RI AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.1 control | 0.20756 | 32.888 | 31.976 | 28.764 | 0.675 |
| 1.0 treatment | 0.20893 | **31.907** | **30.844** | 28.808 | 0.677 |

The treatment improves raw decoder AEPE by 2.98% and conditional decoder AEPE
by 3.54%. Paired sequence intervals for the absolute improvements are
0.982 [0.681, 1.279] and 1.132 [0.862, 1.430] pixels. Attention changes by
0.15% and latent MSE by 0.66%, satisfying the registered limits. Conditional
decoder PCK3 rises from 8.14% to 8.45%, versus the encoder's unchanged 8.76%.

The encoder remains better: centered encoder AEPE is 30.927, and conditional
centered encoder AEPE is 30.071. The corresponding residual decoder gaps are
0.980 and 0.773 pixels, with paired intervals excluding zero. Attention is
better in mean, but its encoder-minus-attention interval is still inconclusive:
1.263 [-0.017, 2.391] pixels. These are development intervals, not independent
confirmation or training-seed uncertainty.

The qualifying treatment completed its exact continuation from update 1,000 to
4,000 in 3,463.7 seconds, restoring both optimizers and the gate. It executed
250 frozen, 1,000 final-two-block and 1,750 full-encoder updates. The teacher
remained unchanged; both the first and last student blocks changed. The audited
lineage now contains 10,200 updates and 163,200 target exposures.

Validation latent MSE reaches 0.202598, learned RI AUROC 0.6930, raw decoder AEPE
31.491, conditional decoder AEPE 30.521 and conditional attention AEPE 28.796.
The respective own final-layer encoder controls score 30.893 and 30.052.
Continuation improves the decoder, but its remaining gaps are 0.598
[0.078, 1.106] and 0.469 [0.066, 0.894] pixels. Conditional attention improves
on that encoder by 1.256 [0.068, 2.284] pixels. These remain development results.

The completed read-only hierarchy probe gives a stronger encoder baseline:

| Conditional centered features | Fixed teacher AEPE | Own refinement encoder AEPE | Own encoder PCK3 |
| --- | ---: | ---: | ---: |
| Block 3 | 39.472 | 38.197 | 8.68% |
| Block 6 | **25.919** | **25.219** | **11.76%** |
| Block 9 | 27.263 | 26.077 | 10.79% |
| Block 12 | 31.662 | 30.090 | 8.76% |
| Uniform cosine-score mean | 26.264 | 25.371 | 11.68% |

The own encoder's block-6 improvement over block 12 is 4.852 [3.883, 5.838]
pixels. It holds in the largest true-displacement stratum as well. Teacher
final-layer controls match exactly; capturing extra student layers changes
0.053% of conditional centered final-layer argmaxes, consistent with the
previously measured CUDA dispatch sensitivity. This small numerical difference
does not account for the multi-pixel hierarchy gap.

The next registered screen changes only the affinity teacher from the final
layer to trained block 6, starting both arms from the same continuation endpoint.
Primary and dense latent targets stay final-layer. All future external comparisons
retain block-6 encoder baselines: beating the weaker final-layer control alone
does not establish a fusion advantage. No hierarchy-guided training result has
been observed yet.

## Spatial-affinity result and direct input experiment

The matched affinity-level screen completes 1,000 updates per arm from the same
4000-step metric continuation. Both encoders remain frozen. The sole objective
difference is an empty `teacher_layers` list versus `[6]`; primary and dense
latent targets remain final-layer features. Samples, masks, learning rates and
stages match exactly. All 40 declared teacher/frozen-encoder readout controls are
identical in the common v16 export.

| Readout | Final-layer affinity control | Block-6 affinity | Block-6 PCK3 |
| --- | ---: | ---: | ---: |
| Raw decoder | 31.434 | **29.294** | 9.11% |
| Conditional decoder | 30.433 | **28.014** | 9.88% |
| Conditional attention | 28.637 | **26.946** | 9.27% |
| Unchanged block-6 conditional encoder | 25.201 | 25.201 | 11.75% |

Control-minus-treatment viewpoint AEPE is 2.140 px with a 59-sequence bootstrap
interval [1.414, 2.877] for raw descriptors; 2.419 [1.731, 3.136] for conditional
descriptors; and 1.691 [1.084, 2.312] for attention. PCK3 improves in all three
readouts. Validation latent MSE is 0.204149 versus 0.204862, a 0.349% regression
within the registered 5% limit. Learned RI AUROC is 0.66985 versus 0.67163;
spatial variance ratios are 0.36139 versus 0.36251.

This closes the regression against the weaker final-layer encoder in this
development evaluation, but **fusion still trails the stronger block-6 encoder**.
Its conditional decoder error is 2.813 px higher, with interval [2.350, 3.285].
Guidance improves the trunk without making the intermediate representation an
input. The separately registered next experiment therefore appends block-6
student features to the final features with a zero-initialized projection
extension. Its matched control keeps the original input, and both keep the
successful block-6 auxiliary targets. Initialization, gradient flow, hidden-RGB
isolation and exact checkpoint continuation are tested before native runs.

The selected guidance checkpoint is
`e9d9090f8de0652dab9b7dc643c3a481564f1f0f7ddf689a55a4da70e28799d5`.
Its lineage contains 11,200 actual updates, 179,200 target exposures and all
8,192 training rooms, with no nontraining optimizer inputs or noncommercial
dependencies. Both independent qualification sets remain unobserved at this
decision. The direct input path is an experiment, not an established improvement.

## GPU efficiency investigation

The user's stall concern is supported by a bounded Nsight replay of 32 fully
unfrozen updates. All replayed total losses equal the original run. A warmed
24-update window contains 523,272 kernels, with kernel intervals occupying 40.34%
of elapsed time. RGB transfers consume 7.668 seconds, or 0.3195 seconds/update.
This explains why the device's 97% median activity counter was insufficient
evidence of efficient computation. Trace gaps can also include host work and
untraced desktop activity; kernel time is not SM occupancy.

CubeCL 0.10's rank-4 NHWC upload becomes a CUDA 2D copy with 12-byte rows. The
new path uploads one flat contiguous buffer, then reshapes and permutes on the
device. In an alternating GPU microbenchmark, normalized RGB is bitwise equal
for all three tested layouts. Batch-16 upload plus normalization falls from
61.119 to 7.186 ms; batch 1 falls from 3.771 to 0.957 ms. CPU planar packing is
also correct but slower than flat upload.

The subsequent matched 64-update training check passes: every logged loss and
final validation MSE is identical, with maximum gradient-norm difference
1.73e-8. Excluding the first eight updates, median update time falls from 0.7911
to 0.4860 seconds (1.63x throughput). Complete command time falls from 152.54 to
128.05 seconds; observed board energy falls from 7.311 to 6.665 Wh (8.84%).
This includes startup, validation and desktop activity, and measures a frozen
encoder configuration. It is not an energy claim for every unfreezing stage.
The qualified flat upload is used in the spatial-affinity screen. See the
[GPU efficiency record](../gpu-efficiency.md) for reproduction and limitations.

The two completed 1,000-update spatial arms sustain median update times of
0.4834 and 0.4815 seconds. Their complete commands consume 73.34 and 74.32 Wh,
or 16.50 and 16.72 board joules per trained target. Both telemetry records have
complete supported coverage. These longer-run measurements describe the actual
quality workload; the isolated before/after energy claim remains the matched
64-update check.

`tools/study/gpu_efficiency.py` integrates measured board power with explicit gap
accounting. The unprofiled continuation consumed approximately 293.59 Wh,
or 22.02 gross board joules per trained target including preparation/evaluation
and desktop activity. The experiment keeps the existing 12-hour ledger.

## Evaluation boundaries

HPatches is now development data because its previous failures motivated this
work. Selecting layers, losses or positional settings on it is tuning, even if
its pixels never appear in a training batch. Previously inspected synthetic test
rooms are also development evidence for this phase.

ETH3D is prepared independently: all 3,365 official interval pairs, 2,448 distinct
images, 10 scenes and seven temporal intervals. The candidate/readout selection
record must be sealed before native export. RGB inputs and sparse point labels
are separate artifacts. Original-resolution scoring preserves the official
distinction between image/scene means and point-weighted PCK; tests cover resize
coordinates, duplicate point rasterization and unequal scene sizes.

The current local matching procedure does not reproduce ZeroCo's refinement.
Its published scores are therefore context, not a directly comparable leaderboard
claim. A single continuation seed and a passed holdout are insufficient to claim
state of the art. RGB blur is not declared resolved by a latent-only experiment.

## Provenance and compute

All data, weights, telemetry and reports remain under `.data/`. User-facing
configuration is TOML. The factorial training uses sealed `latent_pilot-v9` and
its completed HPatches evaluation uses `hpatches_export-v12`. The descriptor
screen uses `latent_pilot-v13`; the matching v13 exporters add the centered latent
readout. Source archives and SHA-256 receipts accompany each version under
`.data/pilot-07/fusion-audit/`.

The [objective appendix](../fusion-objective.md) records the distinct main and
auxiliary teachers, all gradient paths and the RI adaptation. The descriptive
`student_initialization` string in older exact-resume provenance could describe
a cold start despite the structured `resume` path and correctly restored states.
The source now distinguishes these cases. Immutable artifacts are retained with
`resume-description-annotation.json`; their model/optimizer restoration checks,
logged starting steps and lineage audits remain authoritative. The sealed v13
binary is retained for exact continuation of its original source identity.

The existing cumulative 43,200-second Pilot07 ledger is retained. No budget reset
or noncommercial initialization is permitted. Native GPU work, including failed
commands, counts against this ceiling; CPU engineering and data downloads do not.

## Direct-input initialization preflight

The v17 narrow-versus-wide input experiment failed its registered native
initialization gate: maximum initial per-view cross-MSE difference was
1.28204e-5 against a 1e-5 bound. Its 1000-update quality runs were not started.
The default 64-update path retained exact component losses and validation MSE.

A separately registered v18 experiment uses the same widened projection and
block-6 capture in both arms, with only `spatial_input_scale` set to zero or one.
This isolates the extra information while matching tensor layout and parameter
count. Its native preflight retains the 1e-5 bound; no rejected threshold is
relaxed. Both initial cross-view and monocular per-view MSE differences are
exactly zero. Frozen encoder/teacher checks and hidden-RGB isolation pass.

The completed 1,000-update arms give the following common HPatches viewpoint
development results. The encoder remains frozen in both arms.

| Readout | Scale-zero control AEPE | Block-6 input AEPE | Paired improvement, 95% sequence interval |
| --- | ---: | ---: | ---: |
| Raw decoder | 28.781 | 28.282 | 0.499 [0.229, 0.764] |
| Conditional decoder | 27.594 | 27.142 | 0.452 [0.196, 0.703] |
| Conditional attention | 26.584 | 26.296 | 0.288 [0.080, 0.500] |

Decoder PCK3 improves in both readouts. Validation masked MSE rises 0.072%
relative, and learned RI AUROC rises from 0.6613 to 0.6758. All registered
development screening gates pass. Fusion still trails the stronger block-6
encoder (plain/conditional AEPE 26.403/25.201), so this is evidence that spatial
input helps, not a resolved transfer claim. Fixed teacher and separately
captured block-6 readouts are identical. Small final-layer argmax changes
against the narrow parent remain within the registered numerical-control bounds.

The selected step-1000 model is exactly resumed through its original 6000-step
horizon, with both optimizers, the sample position, fixed affinity teacher and
validation-gated unfreezing state restored. Its endpoint is retained only if all
three AEPE means improve, decoder PCK3 loses at most 1% relative, masked MSE
rises at most 5% relative, and RI AUROC loses at most 0.02 absolute. Otherwise
the qualifying step-1000 parent is retained. This rule is recorded before the
continuation outcomes and independent evaluation.

The completed continuation passes every gate. Raw/conditional decoder AEPE is
27.548/26.422; conditional attention is 25.696. Validation masked MSE falls from
0.204486 to 0.195858, and learned RI AUROC rises from 0.6758 to 0.7356. It executes
250 frozen, 1,000 last-block-stage, and 3,750 fully unfrozen updates, with median
times 0.490/0.549/0.761 seconds. The fixed teacher remains unchanged and both
early and late student blocks update. Its 4,185-second command consumes 460.12
board Wh under the recorded shared-GPU conditions.

The stronger own encoder still has plain/conditional AEPE 26.539/25.487. Since
the continuation finishes below its 9,000-second allowance, the remaining
development budget supports one final matched descriptor-preservation screen:
1,500 frozen-encoder updates per arm, descriptor weights 1 versus 4, identical
own endpoint initialization and fresh optimizers. Both use the new phase's
frozen own ancestor as the block-6 affinity teacher. Its
[registration](../../configs/archive/pilot-07/pilot07-fusion-preservation-screen.toml) fixes the
selection rule before training and retains 4,000 seconds for independent
qualification. Increasing an auxiliary weight is a tested hypothesis, not an
assumed repair; the prior endpoint remains a fallback.

The two matched commands consume 92.15 and 92.91 board Wh, with median update
times approximately 0.958 and 0.944 seconds. Concurrent desktop GPU activity
makes these later timings different execution conditions from the earlier
0.48-second runs; the same sealed old binary reproduces the slowdown. The
[shared-GPU record](../gpu-efficiency.md) separates that observation from the
qualified upload repair. Both training commands and their complete evaluations
remain in the original cumulative ledger.

## Final preservation screen and selection

Both 1,500-update preservation arms complete with the encoder frozen. Weight 4
improves all three AEPE means over its matched weight-1 control, but attention
AEPE is 25.7477 versus the qualifying 6000-step parent's 25.6962. The control
also fails the all-three-improvement rule against that parent. The registered
rule therefore retains `route-balanced-continue`; neither later arm is chosen
after inspecting independent results.

The v18 binary initially rejects weight 4 before training because its parameter
validator only accepts weights through 1. Version 20 changes that finite range
to `[0, 4]`, with boundary tests and no training-arithmetic change. Its default
weight-1 replay matches all component losses for 64 updates; maximum gradient
norm difference is 3.20e-8, and initial cross/monocular MSE differences are zero.
An undersized native preparation allowance stops the replay after update 52;
its intact model and both optimizers are exactly resumed for the remaining 12
updates. The zero-update rejection and interrupted replay remain in the ledger.

The preservation arms consume 107.10 and 109.34 board Wh. These and the earlier
phases run under recorded shared desktop activity. Their gross energy values
are descriptive; the controlled energy improvement remains the separate upload
qualification. The final report includes the unsuccessful accuracy experiment
and its recovery, rather than selecting the best metric from different arms.
