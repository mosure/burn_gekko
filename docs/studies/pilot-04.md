# Pilot 04: reconstruction quality recovery

**Accepted for the bounded procedural-room task.** The pervasive blur and
incoherent room-level collapse in pilot 03 are corrected. The hybrid reduces
hidden RGB error by **61.0%** on 64 fresh rooms and has a substantial advantage
over its trained monocular control. Fine texture, thin objects, ghosting and
patch brightness seams remain imperfect. This is not exact reconstruction or
real-image transfer qualification.

[Review the PDF with matched before/after images, crops, errors and visibility annotations](../../.data/pilot-04/burn_gekko_pilot_04_report.pdf).
[Machine-readable results](../../.data/pilot-04/quality-summary.json).
[Architecture, inputs and reproduction guide](../hybrid-pipeline.md).

## Fresh-test results

The checkpoint was frozen before generating the fresh cache. All 192 target views
from 64 rooms were evaluated at 256×256 with 75% of target patches hidden and two
reference views. The previous pilot-03 candidate was evaluated on the identical
pixels and masks. Every hybrid target was exported and independently scored in
NumPy; no hidden target statistics or image enhancement is used.

| Readout | Previous model | Selected hybrid |
| --- | ---: | ---: |
| Hidden RGB MSE, all 192 targets | 0.00351063 | **0.00136866** |
| PSNR from pooled MSE | 24.55 dB | **28.64 dB** |
| Edge cosine, 16 exactly matched target-zero exports | 0.0929 | **0.5178** |
| Edge energy / target, same 16 exports | 0.0730 | **0.8497** |
| Dense co-visibility AUROC, all targets | 0.5209 | **0.7476** |
| Dense co-visibility AP | 0.8375 | **0.9235** |

The full-test hybrid edge cosine is **0.5633** (95% room-bootstrap CI
0.5409–0.5851), and its edge-energy ratio is **0.8595** (0.8229–0.8980).
Excluding all 16px patch boundaries, cosine is 0.5739 and energy ratio is 0.8421:
the improvement is not confined to block seams. At patch boundaries, the energy
ratio is 1.3285, so excess seams remain a measured limitation.

Correct-reference RGB MSE is 0.001369, compared with **0.002503 monocular** and
**0.004635 using unrelated rooms**. Cross-view completion reduces error by 45.3%
relative to monocular completion. Room-paired 95% confidence intervals for
baseline, monocular and unrelated-reference advantages all exclude zero.
Separate CUDA evaluations differ in monocular MSE by at most 2.74e-6; this is
small relative to the measured 0.001134 monocular advantage. The monocular API
accepts no reference input.

A full-pipeline hidden-pixel intervention on a fresh target changes RGB output by
exactly **zero**; reference permutation also changes it by zero. Model RGB inputs
exclude depth, pose, geometric labels and hidden target patch statistics.
The pretrained RI channel is evaluated using a separate dense-target
forward. In this adaptation study it is inherited, not directly retrained.

Geometry labels identify a mean 82.4% of labeled hidden pixels as visible in at
least one reference. Conditional RGB MSE is 0.001111 there, versus 0.003274 on
pixels absent from both references. Unknown labels are excluded. Constant-score
AP equals the visibility prevalence, approximately 0.825; high AP alone is not
proof of matching competence.

## What resolved the failure

Earlier weight, patch-layout, sparse-mask, branch-isolation, CPU numerical-parity,
tiny-set-overfit and RGB/geometry checks passed. A small scratch decoder still
failed to learn adequate held-out correspondence and appearance reconstruction.
There is no evidence that an additional image-layout repair explains this gain.
The individual effects of representation, capacity and pretraining are not isolated.

The user explicitly authorized **both encoders, prioritizing reconstruction
quality**. The resulting model combines frozen V-JEPA 2.1 Base and the released
Gekko-L appearance encoder. A residual semantic/observed-RGB adapter feeds the
pretrained cross-attention decoder. A learned head predicts RGB patch mean and
scale, avoiding the original model's oracle target-statistic visualization.
Shared pairwise RGB predictions are averaged across references.

The strict Burn importer consumes all 712 F32 released checkpoint tensors exactly
once, with exact shape checks. CPU relative RMS against official PyTorch is at
most 2.96e-6; CUDA at most 0.00221. Dense/sparse encoder, decoder features, cross
RGB/RI, MAE and dense RGB/RI are checked. Hidden-pixel encoder interventions are
exactly zero on CPU and CUDA. The original imported `burn_jepa` files remain
unchanged and checksum-verified.

| Training stage | Rooms | Updates / batch | Main change | Wall time | Warm throughput | Peak process VRAM |
| --- | ---: | --- | --- | ---: | ---: | ---: |
| Calibration/adapter | 128 | 3,000 / 8 | Frozen encoders and entire decoder | 10.3 min | 43.1 images/s | 14.4 GiB |
| Decoder adaptation | 512 | 3,000 / 8 | Last two decoder blocks, shared monocular calibration, gradient error | 16.8 min | 29.0 images/s | 23.3 GiB |
| Contrast preservation | 512 | 1,500 / 8 | Small training-only gradient-energy term | 8.7 min | 29.4 images/s | 23.3 GiB |

Each stage starts a new AdamW optimizer from previous weights. This is **not
optimizer resumption**. Both encoders, the first ten decoder blocks and packed
RGB/RI output heads stay frozen. Weight-delta assertions verify unchanged frozen
parameters and updated final blocks. There are 19,667,074 trainable parameters in
the final two stages. Warm timing excludes the first 100 updates; end-to-end
throughput of the final stage is 23.0 images/s. Device utilization includes the
workstation desktop; process VRAM is tracked separately.

Full validation selected contrast preservation over the minimum-MSE checkpoint:
0.001341 versus 0.001214 MSE; 0.845 versus 0.351 edge-energy ratio. This is an
explicit quality tradeoff, frozen before fresh-test capture. The adapted model's
validation RI AUROC is 0.734, below the unadapted released model's 0.778. The report
preserves that regression rather than presenting reconstruction adaptation as an
RI improvement over the released model.

## Rejected controls and boundaries

Explicit learned RGB transport was sharp but misplaced objects: about 0.025 edge
cosine on the initial four-room screen, despite substantial gradient energy.
Visible-only local translation and homography registration also failed. These
are retained diagnostics, not accepted pipeline defaults.

A released Gekko positive control recovered stronger structure, but used hidden
patch statistics for display. It is labeled **ORACLE** and excluded from final
standalone RGB acceptance. The selected hybrid predicts every required statistic.

The appearance weights are from
[Gekko ViT-L, 500k steps](https://huggingface.co/thibautloiseau/gekko-vitl-500k),
revision `79fba28dd59ec54fffa0134fae681d88ed084513`, SHA256
`ce415f674dfcbf9d66ba91ebf213abb0010bd115cba72586829e7f5205df6e08`,
with **CC-BY-NC-SA-4.0** terms. Its large external pretraining is not included in
this workstation budget. This comparison does not establish a unique V-JEPA
benefit, a new learning method, joint set-attention gains, sparse-reference speedup,
calibrated RI probabilities or real-data transfer.

## Data, verification and reproduction

Published `bevy_zeroverse=0.23.0` and `bevy_zeroverse_burn=0.6.0` were rechecked as
the latest stable releases on 2026-09-28 at 16:58 UTC. Immutable RGB/annotation
shards are cached under `.data/datasets/`; frozen feature caches are GPU-resident
per run. User input configurations and command plans are TOML. Measured results
remain JSON/JSONL. The fresh 81-room capture took 26.3 seconds; its extra 1/16
train/validation rooms were unused.

- Training cache: `4287ae3d171d81e05febe5ad057d2ca448a03d006f0da30e018c5350e90ec149`.
- Fresh cache: `5fad342973901c9338699b842b476aad52294c729f0694a545bfb8a26ebb8b17`.
- Test seeds: 2026110017–2026110080, disjoint from all seven previous caches.
- Selected checkpoint: `.data/runs/pilot-04-hybrid-detail/final.mpk`.
- Checkpoint SHA256: `492198b1d4b1348a704890668928c1db510941394096a5d97caa593e29ab68f8`.
- Frozen selection: `.data/pilot-04/selection.toml`; initial gates: `protocol.toml`.
- Exact commands, logs and GPU telemetry: `.data/pilot-04/plans/` and stage folders.
- Stage source/binary snapshots, checkpoint receipts and raw exports: `.data/pilot-04/`.

The metered experiment ledger totals **3,704.7 seconds (61.7 minutes)** against the
7,200-second ceiling, including rejected GPU trials, CPU/CUDA parity commands,
capture and final evaluations. Compilation, downloads and Python analysis/report
rendering are outside this ledger. No further training or selection used the
fresh-test outcomes. Historical pilot-03 artifacts are unchanged; the previous
model was copied into a separate evaluation directory.

**52 workspace tests, strict CUDA-feature Clippy, encoder-import checksums and
independent exported-float metrics pass.** Tests include loss mask-boundary
isolation, sampler gradients, checkpoint continuation for the original trainer,
cache validation and rejection of overlapping generalization datasets. Final
model corruption/permutation checks are recorded in `final-input-audit.json`.

```sh
cargo test --workspace --locked
cargo clippy --workspace --all-targets --features cuda --locked -- -D warnings
python3 tools/interop/verify_encoder_import.py

# Regenerate this report from the preserved local artifacts.
OPENBLAS_NUM_THREADS=1 .data/analysis-venv/bin/python tools/legacy/hybrid_report.py \
  --config configs/train/report-pilot04.toml
```

The [hybrid guide](../hybrid-pipeline.md) describes new-run commands and information
flow. The [paper plan](../paper-plan.md) remains a research program: multiple seeds,
controlled component ablations, real data, sparse quality/cost curves and a
claim-to-evidence audit are still required for broader scientific claims.
