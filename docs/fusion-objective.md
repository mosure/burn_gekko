# Fusion-transfer objective and information flow

This describes the implemented Pilot07 RGB-only latent adaptation. It is not an
exact reproduction of Gekko's RGB training, V-JEPA pretraining, or a camera-based
fusion architecture. Concrete coefficients, ancestry and schedules are in the
registered TOML configs; completed results are in
[the fusion-transfer study](studies/pilot-07-fusion-transfer.md).

## Inputs and fixed teachers

The student encodes visible target patches before transformer attention, and
encodes each reference image independently. A shared fusion decoder predicts
all target tokens from the sparse target and a reference set. The matched
monocular branch has the same visible target and no reference images. Both
branches use the same latent projection. A separate full-target branch predicts
relative improvement (RI); its dense features never enter sparse completion.

The primary teacher `T0` is the audited MIT V-JEPA 2.1 Base encoder and never
updates. The auxiliary teacher `Ta` can instead be the audited own student
encoder at a named warm-start checkpoint. In the reported guided phases, `Ta`
remains frozen at that ancestor, including across exact optimizer resumes.
Its normalization and feature coordinate system are therefore distinguished
from `T0`. Neither teacher has noncommercial weight ancestry.

All teacher forwards use the non-autodiff backend. Default student forwards
request only their consumed final feature output. Optional feature routes
request their additional consumed level, avoiding unused hierarchical autodiff
branches. The separate hierarchy audit is inference-only.
In the affinity-level screen, block-6 features supply stopped auxiliary targets;
they are not an additional student input to fusion. The separately registered
`spatial_input_layer` experiment tests direct access to intermediate features.

With this optional input route enabled, the student concatenates the selected
trained feature level after its final-layer tokens. The decoder's shared input
projection gains rows initialized to zero, preserving the previous model's
mathematical initial prediction. Those rows can then learn from the existing
objectives. Both feature levels are extracted after the branch's sparse mask;
the route cannot read hidden target RGB. The route adds 294,912 weights for the
Base encoder and width-384 decoder, without another pretrained package.

Projection extension is permitted only for a new phase. Exact resume and
assessment require the saved weight dimensions to agree with the configured
input route. The primary teacher and prediction dimension remain unchanged.
Historical encoder readouts explicitly use the final-layer portion; separate
block-6 controls retain the stronger spatial baseline. This option is an
architecture experiment and is not a qualified accuracy result merely because
its initialization, gradient and checkpoint tests pass.

## Main latent and RI losses

For each patch, `N(z)` subtracts the channel mean and divides by the square root
of channel variance plus `1e-6`. Teacher normalization is parameter-free;
predictions are not normalized. For a common hidden set `H`, with cross-view
prediction `p_x` and monocular prediction `p_m`,

```text
e_x[i] = mean_channels((p_x[i] - N(stopgrad(T0(image)))[i])²)
e_m[i] = mean_channels((p_m[i] - N(stopgrad(T0(image)))[i])²)
L_mask = 0.5 * (mean_H(e_x) + mean_H(e_m))
L_visible = 0.5 * (mean_visible(e_x) + mean_visible(e_m))
gain[i] = stopgrad(clamp((e_m[i] - e_x[i]) / max(e_m[i], 1e-6), 0, 1))
L_RI = mean_H((predicted_RI[i] - gain[i])²)
```

The current RI adaptation is unweighted regression to clipped latent gain.
[Gekko equation 8](https://arxiv.org/html/2609.01530v1) instead scales the
relative-gain residual by detached monocular error. That difference is explicit
and has not been changed during the fusion-transfer experiments. RI is a
utility proxy and ranking score, not a calibrated visibility probability.

## Dense pair guidance

An additional full-target/full-reference pair trains the same fusion trunk in
the information pattern used for correspondence. In bidirectional phases, the
decoder is run in both directions, giving `D_t` and `D_r`. This branch uses one
reference at a time, alternating references by update. The primary masked
completion branch still sees its declared reference set.

For each image independently, `U(z)` removes the mean across patch positions,
then L2-normalizes each patch feature. With fixed temperature `tau = 0.07`,

```text
A_tr = stopgrad(row_softmax(U(Ta(target)) @ U(Ta(reference))ᵀ / tau))
A_rt = stopgrad(row_softmax(U(Ta(reference)) @ U(Ta(target))ᵀ / tau))
```

These are semantic affinity targets, not true correspondence or visibility
labels. Three optional losses operate on the dense pair:

- **Attention guidance:** mean row KL from `A_tr` to row-softmax of each decoder
  layer's head-mean cross-attention logits, averaged over layers. Bidirectional
  phases average the two directions. Individual heads retain their own value
  routing; the objective does not equate each head with a correspondence map.
- **Dense preservation:** MSE between the shared projection of `D_t` and
  `N(Ta(target))`, averaged with the reverse direction when enabled. This uses
  the named auxiliary ancestor, whereas the main masked target uses `T0`.
- **Descriptor alignment:** symmetric mean row KL from `A_tr` and `A_rt` to
  the forward and transposed cosine-score matrices of `D_t` and `D_r`, divided
  by `tau`. These decoder descriptors are L2-normalized without patch-mean
  centering. Both descriptor branches receive gradients; teachers do not.

The total objective is

```text
L = L_mask + lambda_visible * L_visible + lambda_RI * L_RI
    + lambda_attention * L_attention + lambda_dense * L_dense
    + lambda_descriptor * L_descriptor
```

The strength screen uses `lambda_visible = lambda_RI = lambda_attention =
lambda_dense = 0.1` and changes only `lambda_descriptor` from `0.1` to `1.0`.
The earlier factorial and descriptor screens have their separately recorded
coefficients and branch choices. Weights-only phases restart optimizers and
the stage gate. Exact continuations restore model, both AdamW states, gate,
sample/mask position and the original learning-rate horizon.

An optional `fusion_auxiliary.teacher_layers` selects one-based, trained
hierarchical encoder outputs for affinity targets. Empty preserves the final
layer. Multiple selected levels contribute the uniform mean of centered unit
cosine score matrices **before** temperature and softmax; gradients stop at all
teacher levels. This changes neither the primary teacher nor the dense latent
target. The block-6 versus final-layer comparison is a separately registered
matched experiment, motivated by the development-only hierarchy probe.

## Position and inference readouts

The fusion trunk retains image-grid 2D RoPE. Every reference repeats its own
grid; a shared reference-role embedding preserves reference-set semantics.
Intrinsics, extrinsics, camera rays and depth are absent from model inputs.
Image coordinates do not establish a common 3D coordinate frame.

The original probability and raw-logit attention readouts remain controls.
The conditional readout averages per-layer forward and transposed reverse
log-softmax scores before nearest-neighbor selection. It is invariant to
arbitrary row offsets that row-wise KL cannot identify. To separate learned
fusion from normalization effects, the same fixed-temperature operator is
also applied to raw and centered encoder and decoder cosine scores. No method
is selected separately for individual test examples.

Full-image matching and masked completion are different information sets and
are reported separately. Teacher features, homographies and renderer geometry
are unavailable to learned matching scores. Geometry enters CPU evaluation and
development model selection only, never the optimizer's losses. Changing
hidden target RGB leaves sparse predictions unchanged in the saved input audit.

## What this does not establish

The auxiliary losses can preserve an encoder's matching geometry without
creating new geometric information. An encoder-level probe, matched loss
controls, fixed readouts and independent real-view qualification are needed
to establish a fusion benefit. Lower latent MSE alone does not establish sharp
RGB synthesis, calibrated co-visibility, camera recovery or state of the art.

The experimental `spatial_input_scale` defaults to one and requires an enabled
`spatial_input_layer` if changed. Zero provides a matched-layout control: both
arms capture the same middle features and widen the projection, but the control
zeros only the added feature channels. This keeps the projection shape and
parameter count equal. The first narrow-versus-wide native preflight exceeded
its registered initial-MSE tolerance and was rejected before quality training;
mathematical neutrality alone is not a claim of bitwise CUDA parity.

The descriptor KL coefficient is a loss multiplier, not a probability. The
registered final preservation screen tests weights 1 and 4. Version 20 expands
its accepted finite range from `[0, 1]` to `[0, 4]`; attention and dense-latent
weight limits and all experiment selection gates stay unchanged. Boundary tests
reject negative, excessive and nonfinite values. The earlier v18 weight-4
startup is retained as a zero-update validation rejection. A native weight-1
trajectory comparison qualifies reuse of the completed control across this
validation-only Rust change.
