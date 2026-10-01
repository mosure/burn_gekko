# Pilot 14: information limits and camera-solver stability

This study diagnoses the fixed Pilot 13 geometry checkpoint before spending on
another training continuation. Its inherited GPU allowance is 361.215639 seconds;
no additional allowance is assumed. There is no new training, checkpoint
selection, coefficient search or dataset generation in this study.

The registered residual-budget protocol, solver seed panel, binary/source pins
and immutable inputs live in `.data/pilot-14/`. CPU scoring is native Rust.
TOML inputs remain in `configs/eval/` and `configs/experiments/`. Old artifacts
and the completed Pilot 13 report remain unchanged.

## Why variance alone is a poor target

`burn_gekko_eval::metrics::detail` separates hidden-token error into each
channel's spatial mean bias and centered spatial error. It also measures
neighbor differences where both endpoints are hidden. Per-view teacher-assisted
gain and variance-matching calculations are diagnostic oracles, not legal
inference transforms. They do not count as improved predictions.

The initial analysis uses all 512 targets from the already reused Pilot 12
development cohort, assessed with the Pilot 13 checkpoint:

| Two-reference completion measure | Result |
|---|---:|
| Hidden-token MSE | 0.187221 |
| Channel-mean bias MSE | 0.006122 |
| Centered spatial MSE | 0.181099 |
| Centered spatial correlation | 0.6443 |
| Adjacent-difference power relative to teacher | 15.92% |
| Adjacent-difference correlation | 0.3835 |
| Mean teacher-assisted best scalar gain | 0.9926 |
| MSE after per-view oracle gain | 0.187008 |
| MSE after matching teacher variance | 0.227198 |

Almost all error remains in spatial structure; a scalar amplitude fix barely
helps. Teacher-variance matching makes MSE about 21.4% worse. That calculation
even has access to the teacher, so a simple contrast boost is not a demonstrated
solution. This does not prove every detail-aware training objective must fail;
it rejects treating variance retention alone as quality.

References improve centered structure error by 0.013721, with paired room-bootstrap
95% interval [0.012032, 0.015381]. Mean-bias error improves by only 0.001119.
Reference information is useful, but most teacher neighbor variation is missing.
The diagnostic preserves undefined zero-power cases and explicitly counts them.
Analytic tests separate biased/scaled correct structure from orthogonal wrong
structure, and reject invalid shapes, duplicate masks and nonfinite values.

## Camera results depend on solver randomness

The fixed 186-pair TUM development export, its three readouts and the original
solver policy are rescored with seeds 781 through 788. No image inference is
repeated. Every seed retains solver failures and the one tiny-baseline exclusion.
The original seed replays within 1e-12 in angle and with exact inlier/trial
identities. There is no best-seed selection.

At the original maximum of 2,048 RANSAC trials, pair-conditioned AUC@10 has
mean 7.99%, range 6.29--10.04%, and population standard deviation 1.12 percentage
points. The same-image transfer gate passes only 1 of 8 seeds; the encoder-control
gate passes 2 of 8. Mean macro AUC gain against the same-image control is positive
for every seed, but the all-sequence/success gate is not stable. Even the original
encoder-control pass is insufficient to establish robust camera transfer.

A second CPU-only diagnostic was registered after this finding, before its
outcomes: increase only the maximum trial count to 8,192, retain the same eight
seeds, point population, confidence, threshold and controls. Both trial policies
and every seed remain in the report. This is a solver-budget sensitivity test,
not a new model result.

| Pair-conditioned camera measure across 8 seeds | Maximum 2,048 trials | Maximum 8,192 trials |
|---|---:|---:|
| Mean AUC@10 | 7.99% | 9.42% |
| AUC@10 range | 6.29--10.04% | 7.63--11.86% |
| AUC@10 population standard deviation | 1.12 pp | 1.18 pp |
| Mean pose recall within 10 degrees | 19.96% | 23.13% |
| Same-image all-sequence gate passes | 1/8 | 2/8 |
| Encoder-control all-sequence gate passes | 2/8 | 2/8 |

More solver effort improves the mean but does not remove variability or make
the all-sequence gate robust. The original seed at 8,192 trials passes both
gates; reporting that seed alone would hide the other outcomes. Neither changing
the solver nor using extra references is a learned model improvement. The
original primary camera report and its failures remain unchanged. Future camera
selection must not overfit one solver seed on these three development sequences.
The two control gates need to hold together: none of the eight seeds passes both
at 2,048 trials, and only seed 781 passes both at 8,192. The 2/8 counts above
refer to each control separately; they are not two joint passes.

## Third-reference information probe

The new CUDA exporter first replays the original two-reference assessment on
eight rooms / 32 targets. Target, cross-view and monocular latent arrays and all
declared scalar metrics agree exactly: maximum differences are zero. This also
qualifies the provenance-only correction described below.

The fixed checkpoint is then assessed on all 128 rooms / 512 targets with three
references, using the exact original mask seed 829 and 90% random mask. The
previously held-out Pilot 13 cohort is now reused development for this probe.
The original two-reference export remains the fixed information baseline.
No result feeds training or checkpoint selection.

| Same-checkpoint information set | Two references | Three references |
|---|---:|---:|
| Hidden-token MSE | 0.184311 | 0.181256 |
| Monocular MSE | 0.198570 | 0.198570 |
| Feature cosine | 0.90256 | 0.90425 |
| Feature signal/error, not RGB PSNR | 7.3819 dB | 7.4540 dB |
| Teacher spatial variation retained | 43.56% | 43.81% |

The third reference reduces error by about 1.66% relative to two references.
Native monocular independence passes the existing 1e-7 threshold. The paired
room-mean MSE reduction is 0.003055, with 95% interval [0.002545, 0.003599] over
128 rooms. Reference benefit relative to monocular rises from 7.18% to 8.72%.
On this same cohort, centered spatial correlation rises from 0.6526 to 0.6597,
and adjacent-difference correlation from 0.3990 to 0.4084. Neighbor-difference
power remains only 17.18% / 17.33% of the teacher. Two-reference oracle amplitude
adjustment changes MSE only from 0.184311 to 0.184118, while variance matching
worsens it to 0.222533. This independently repeats the initial development
cohort's diagnostic within another already captured cohort; it is not a new
held-out evaluation.
The small variance change does not establish resolution of the smoothing issue.
This model supports more than two input views; the result is an information-set
probe, not a comparison between separately trained model versions.

## Correct provenance and process measurement

The old standalone assessment exporter wrote the constant
`geometry_training_supervision = false`. That name incorrectly implied no
training geometry, even though Pilot 13's training configuration explicitly
records the renderer auxiliary. New exports separate
`evaluation_geometry_used_for_training = false` from
`selected_training_phase_view_geometry`, pin the original training config and
state that earlier phases remain in the checkpoint ancestry. A CPU integration
test runs assessment with a deliberately corrupted training-label cache to
verify that inference never reads it. Old artifacts are preserved; the correction
is documented in `.data/pilot-14/legacy-provenance-correction.json`.

The process observer now uses a supported 10-second interval, verifies that it
is alive and producing rows before inference, and stops after the probe. Native
Rust parses its counters without treating unavailable values as zero. The full
three-reference command reports eight numeric model SM observations averaging
49.38%; this is a short inference/startup window, not occupancy or training
utilization. Desktop processes remain present and untouched. Board energy
remains shared-load-inclusive, with no process-attributed power claim.

Replay and full assessment consume 42.881433 and 100.598125 GPU-command seconds.
Combined authorized usage is 42,982.263919 / 43,200 seconds, leaving
217.736081 seconds (3.63 minutes). No renewed budget is inferred from this study.

## Next training decision

The next training study must preserve the real matching gains while directly
testing recoverable spatial structure. Increasing feature variance alone is not
a justified fix. A matched continuation over the existing 8,192 training rooms
can test sustained actual-view supervision, but requires a new explicit GPU
ceiling. Camera retention must account for solver variability, and should first
use separate synthetic development cameras before any new real-data qualification.
An independent real cohort and matched public model protocols remain required
before a SOTA claim. No further training has been launched while the budget
question is pending.

## Verification and review artifacts

All **157 workspace tests**, owned-crate formatting and CUDA strict Clippy over
owned crates/all targets pass. Coverage includes amplitude-versus-structure
identities, masked-neighbor boundaries, incomplete/tampered exports, replay
numeric failures, room-cluster weighting and geometry provenance without any
inference-time training cache. The first Clippy attempt raced a source edit and
failed to find the newly added module; that log is retained, followed by clean
checks on the completed source. No production retries or relaxed metric
tolerances hide failed attempts.

The [32-page PDF](../../.data/publications/pilot14-information-diagnostics-reviewed/paper.pdf)
and [project page](../../.data/publications/pilot14-information-diagnostics-reviewed/index.html)
retain the one Pilot 13 training run, checkpoint and original two-reference
gallery. Four native capability artifacts add the structure diagnostic,
reference-count uncertainty and both solver-seed panels. The primary original
pose protocol is not replaced by a favorable seed or larger solver budget.
Native validation checks 114 bundle hashes, 97 images and 42 local links.
The page passes all six sample selections at 390/768/1440px, with no overflow
or JavaScript errors. No commit, push, registry publication or deployment occurred.
The reviewed bundle improves display labels without changing metric IDs, units,
values or input evidence. Its seven report tests and strict Clippy pass after
that presentation change. The original draft is retained. Visual inspection
confirms readable diagnostic tables and annotated samples. All 521 files pinned
by the Pilot 13 closeout remain unchanged.
