# End-to-end reconstruction without noncommercial weights

The primary trainer is `e2e_pilot`. It uses the pinned Burn V-JEPA image encoder,
fresh set-attention fusion, separate cross-view/monocular RGB heads, and the
self-supervised relative-improvement head. All training inputs are RGB. Renderer
geometry is reserved for evaluation. Released Gekko weights are a historical
quality/parity reference and never initialize or supervise this trainer.

## Initialization and licenses

`initialization.kind = "scratch"` creates the entire model randomly. Encoder
width, depth and heads are configurable. `kind = "vjepa21"` loads the existing
hash-pinned Base image encoder, whose tensors were checked against the official
EMA weights in pilot 03. Fusion and all heads are still new. The official
[V-JEPA repository](https://github.com/facebookresearch/vjepa2) provides the 2.1
checkpoints under its [MIT license](https://github.com/facebookresearch/vjepa2/blob/main/LICENSE).
The audited local package identity is
`c408f68dd18a38824d0fa1d615e6f9f9f04f111d71a6f7c846f41dc187a8795f`.
License source snapshots and hashes are under `.data/pilot-05/licenses/`.

DINOv3 has its own [license agreement](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md),
including redistribution and use conditions; it is not MIT or a blanket
noncommercial license. This study prioritizes the already audited V-JEPA import.
A DINOv3 loader and checkpoint parity audit remain future work.

Arbitrary pretrained package hashes are rejected until audited. End-to-end
checkpoint metadata records the full provenance and rejects listed NC weight
dependencies. This is a reproducibility guard, not a substitute for verifying
the origin of new weights. MIT notices must accompany redistributed upstream
weights or derivatives. Neither the pretrained V-JEPA arm nor its learned decoder
is described as an entirely from-scratch model; only the random arm is.

## Forward and training contracts

The target supplies its observed patches before encoder attention. Each reference
is independently encoded; batching reference images does not mix their attention.
The fusion decoder attends to the reference token set with shared image-grid
rotary coordinates. Optional RGB appearance features contain only patches each
branch is allowed to observe. Reference order does not carry a role index.

RGB completion has no dense-target-feature argument. `rgb_head = "direct"`
predicts unconstrained RGB in fixed ImageNet channel space. The historical
`"calibrated"` head predicts normalized patch content and its own mean/log standard
deviation. Hidden ground-truth statistics never enter inference.
The monocular branch uses the shared trunk without reference inputs. A separate
dense-target forward predicts RI; error-derived RI targets and weights detach
from both RGB heads. The loss contains RGB/statistics, normalized content,
hidden-pixel gradient terms and detached RI supervision. These are self-supervised
training targets, not inference inputs or renderer supervision.

The historical RGB-calibration recipe is not an exact replication of the
released Gekko training loss: cross-view RGB MSE has weight 10, monocular RGB MSE
5, and RI 0.1. The edge term acts on cross-view predictions. These unequal branch
objectives can confound relative-improvement interpretation. Predicted content
is normalized before calibration in that historical mode.
[Pilot 05](studies/pilot-05.md) reports both underfitting and held-out blur, near-chance
RI ranking, and a CUDA reference-order residual above the strict tolerance.
The training path is verified; the recipe is not a qualified quality solution.

[Pilot 06](studies/pilot-06.md) adds an unconstrained direct head with matched objectives
for both RGB branches: 10×hidden RGB MSE + hidden RGB L1 + aligned derivative L1
at pixel offsets 1, 2, 4 and 8. Derivative pairs require both endpoints hidden.
There is no gradient-energy reward. `ri_start_step` delays dense RI training until
the configured absolute update; `ri_weight` controls its contribution. These
changes are being evaluated, not assumed to resolve blur.

The random encoder trains jointly from step zero. The pretrained schedule is:

1. Optimize the new fusion trunk and heads with the image encoder frozen.
2. Unfreeze the final two encoder blocks and output norms after at least 500
   updates, at least 2% validation improvement since stage entry, and no more than
   10% regression from the best probe in the stage.
3. Apply the same measured gate before training all image blocks, the image patch
   stem, and image modality embedding. Unused video parameters stay frozen.

Gates use `probe_rooms` validation rooms (default four) and all target views. Training
probes use the same room limit. Each probe exports view zero from every requested
room, so the detailed image metrics cover that room set; full final evaluations
export every view. Gates cannot inspect
test data. Logs record actual transitions, per-stage steps, gradient tensor counts,
and first/last/stem parameter changes. A stage may remain closed if it fails its
gate. Separate AdamW instances apply a true encoder learning-rate ratio;
rescaling Adam gradients would not provide that ratio. Global norm clipping is
applied before splitting the gradients by parameter identity.

## Data, checkpoints, and execution

Published Zeroverse captures remain immutable, hashed disk caches under `.data/`.
`rgb_cache = "host"` retains bounded raw RGB in RAM and uploads each batch;
`"device"` preserves the earlier resident-GPU mode. `rgb_cache_max_mib` is checked
before allocation. No encoder feature cache exists in this
trainer, so unfreezing cannot reuse stale representations. Masked and full view
features are recomputed each update.

`configs/pilot05-{vjepa,scratch}.toml` define the 256px study and
`configs/archive/pilot-05/pilot05-cuda-smoke.toml` defines the small verification run. Configs are
TOML; measured logs and manifests are JSON/JSONL. The command runner also records
GPU throughput, power, and process VRAM. Pilot 05 had a cumulative 7,200-second
cap. The user expanded pilot 06 to 43,200 seconds. The runner's `--budget-ledger`
option shares that ceiling across sequential legs. The initial fit/capture
overlap is separately reserved and accounted in full; its timing is not an
uncontended throughput measurement.

The initial V-JEPA recipe has a small gradient-energy penalty. Early validation
showed substantial repeated, unaligned patch detail, so the recorded pilot-05
amendment removed that penalty from the subsequent scratch arm and added
`pilot05-vjepa-no-energy.toml`. The optional frozen control uses
`pilot05-vjepa-frozen.toml`. These are distinct experiments; an energy ratio near
one is insufficient evidence of accurate reconstruction. Only the no-energy
arms support comparisons at the same reconstruction objective.

The completed study, resolved configs and annotated PDF are linked from
[pilot 05](studies/pilot-05.md). All four main arms completed 4,000 updates. The fixed
one-room and varying-mask 16-room diagnostics stopped at 1,129 and 895 updates
under their wall caps; these are not completed 2,000/1,000-update schedules.
The optional frozen control was not run.

`pilot05-vjepa-qknorm.toml` additionally enables learned LayerNorm on the query
and key of each attention head. This is initialized from scratch. Released Gekko
uses this normalization for self-attention; extending it to cross-attention here
is an explicit ablation, not a claim of architectural parity. The option defaults
off. The v1 source and executables are retained under `.data/pilot-05/` for exact
reproduction of the first experiments; code identities intentionally prevent
silently resuming an old run under a changed training implementation.

```sh
cargo build --profile pilot --features cuda --locked --bin e2e_pilot -j 4
target/pilot/e2e_pilot --config configs/archive/pilot-05/pilot05-vjepa-no-energy.toml \
  --run .data/runs/my-vjepa-run
# For continuation, keep the decay horizon fixed and set steps to a larger total:
target/pilot/e2e_pilot --config path/to/continuation.toml \
  --checkpoint .data/runs/my-vjepa-run/final \
  --run .data/runs/my-vjepa-continuation
```

Final and `checkpoint_every` periodic directories contain the encoder and decoder model, both AdamW
states, absolute step, unfreeze state, configuration/code identity, and hashes.
Continuation restores these, the sample schedule, and mask sequence; changing
initialization, data, optimizer settings or model/loss code is rejected. CPU
continuation is tested against uninterrupted training. `--evaluate validation`
or `--evaluate test` with `--checkpoint` performs inference only; `--unrelated`
replaces references with views from another room. Test output is exported for all
room/view targets, including raw RGB predictions and monocular controls.

Snapshots use a staging directory and atomic rename. A `STOP` file in the active
run directory asks the trainer to stop between updates, save its final checkpoint
and evaluate. The absolute decay horizon must remain fixed for exact continuation.

`--warm-start PATH` explicitly loads only model weights from an audited own
checkpoint. It permits a new dataset or loss schedule, resets both optimizers
and the absolute step, and retains the source encoder's unfreeze stage. The
parent model and metadata hashes, source dataset, source step and optimizer reset
are recorded in the new checkpoint. Incompatible reconstruction architectures or
NC dependencies are rejected. This operation is a new fine-tuning experiment,
not exact continuation; it cannot be combined with `--checkpoint` or evaluation.
MIT-derived snapshots automatically include the audited upstream license and notice.

Lower reconstruction MSE alone does not qualify quality. Reports also inspect
hidden edge alignment/energy, within-patch detail and seams, reference ablations,
and annotated samples. The large released Gekko baseline's prior cannot be equated
with a short reconstruction run from random weights. Synthetic-room results do
not establish real-image generalization or a new paper contribution.

The archived `e2e_pilot-v1` binary evaluates or continues the first V-JEPA and
scratch checkpoints under their original code identity. The optional-normalization
v2 loader can read v1 model records, with a measured CUDA RGB maximum difference
of 6.85e-6 on the smoke checkpoint; this is approximate compatibility, not a
bit-exact equivalence claim. New experiments use the separately archived v2
executable and source snapshot.

## Optional RGB appearance transport

`appearance_transport = true` enables a shared sampling head inside the decoder
optimizer; it requires `rgb_head = "direct"`. The head predicts per-reference
backward dx/dy and mixture logits on a four-samples-per-patch grid. Dense bilinear
sampling mixes reference RGB with the existing generator. The defaults are
`transport_max_displacement = 64.0` pixels per axis and
`transport_loss_weight = 5.0` for the RGB-derived multiscale photometric auxiliary.
No renderer geometry or hidden target RGB enters completion. RI supervision uses
the actual blended reconstruction, not the generator before mixing.

This is an explicit extension of the Gekko reconstruction recipe. Primary hidden
RGB/derivative losses still match between CVC and MAE, but CVC also receives the
transport auxiliary. Do not describe the total branch training signals as equal.
The transport head can be added by an explicit own-checkpoint warm start; parent
hashes and optimizer resets are recorded. Loading old records with transport off
preserves output, and exact CPU continuation with the new head is tested.

Raw exports additionally contain `generated.f32`, `transported.f32`, `mixture.f32`
(generator followed by references), and `flow-N.f32` (HWC, two pixel-displacement
channels). Analysis checks the mixture identity and measures component error and
source usage. Checkpoint reports assert that the appearance-head weights update.
`transport_pyramid_loss = true` uses actual reference-image pyramids for the
auxiliary: reduce reference and target images, flow and reference-mixture weights
at scales 2/4/8/16/32; divide flow by scale; then warp each reduced reference.
The default `false` retains pooled warped-output supervision for reproducing
earlier experiments. A translated-stripe contract demonstrates the missing
coarse displacement gradient that motivated this correction. Changing the loss
requires an explicit own-checkpoint transfer; exact continuation retains the
original loss and optimizer state.

Pilot 06's 66-test suite, strict CUDA Clippy, and native batch-16 preflights
passed. CPU evaluation can compare saved flows to captured geometry with
`tools/legacy/appearance_geometry_audit.py`; it checks target RGB, reference order and
self reprojection, and never supplies labels to training. Improvement in
held-out reconstruction quality remains under investigation.

The optional `transport_coarse_smoothness_weight` adds edge-aware absolute
second derivatives of the native quarter-resolution flow controls. Zero keeps
the prior objective. RGB is pooled by four, detached and used only to weight
curvature across image edges. There are no geometry supervision inputs. The
control field uses quarter-resolution pixel units; it applies tanh before
upsampling, whereas the unchanged inference field applies tanh after
upsampling. Enabled runs export `coarse-flow-N.f32` as H/4 by W/4 by 2 HWC
float32, alongside full-resolution sampling flows. Tests check affine fields,
nonzero curvature gradients, edge attenuation, reference ordering, hidden-input
independence and checkpoint/optimizer replay. Version 9 passed 68 workspace
tests, strict CUDA Clippy and the native build; its matched GPU screen is
recorded separately in pilot 06.
