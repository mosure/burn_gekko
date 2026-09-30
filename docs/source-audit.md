# Source audit and reference ledger

The original 2026-09-27 sections are planning snapshots. Dated implementation
updates below record later registry, numerical, and runtime verification.

## Pilot 06 registry, training and evaluation (2026-09-29 UTC)

Capture now pins published `bevy_zeroverse=0.25.0` and
`bevy_zeroverse_burn=0.8.0`; the September 29 registry recheck confirms these
remain latest. The original capture receipts and the new read-only recheck are
retained separately in `.data/pilot-06/registry-versions.json` and
`.data/pilot-06/registry-recheck.json`.

Primary package records: [generator](https://crates.io/api/v1/crates/bevy_zeroverse/0.25.0)
and [Burn adapter](https://crates.io/api/v1/crates/bevy_zeroverse_burn/0.8.0).

The v9 end-to-end trainer adapts the audited MIT V-JEPA encoder and trains the
fusion, RGB, relative-improvement and appearance-sampling heads. Its selected
ancestry contains no released Gekko weights or noncommercial teacher. Recursive
checkpoint and sampler auditing covers 37,230 updates and 531,680 training
examples; exact-resume prefixes are counted once. This is an adapted recipe,
not a reproduction of the original Gekko training protocol.

The [Pilot 06 report](studies/pilot-06.md) preserves the quality failures: low held-out
gradient energy, missing local structure, and a native reference-permutation
maximum residual above the original strict threshold. The v9 source passed 68
workspace tests and strict CUDA Clippy; 53 core source/dependency files match
the archived implementation. Passing implementation checks is separate from
reconstruction qualification. Earlier sections below describe historical states.

## Pilot 03 registry and encoder refresh (2026-09-28 UTC)

The current capture workspace pins published `bevy_zeroverse=0.23.0` and
`bevy_zeroverse_burn=0.6.0`. Both downloaded archives match their crates.io
checksums. Records and original archives are in `.data/pilot-03/`.

| Crate | Version | Registry archive SHA-256 |
| --- | --- | --- |
| `bevy_zeroverse` | `0.23.0` | `79f5b7a0be927b461d6365f8b4dac33b31becadc3a2f6a0753d7916ce98df6a7` |
| `bevy_zeroverse_burn` | `0.6.0` | `5c499b4ae97f3378866d2319154338170b1304dd64d7c57077cd72ee4af5bf6a` |

Primary records: [generator](https://crates.io/api/v1/crates/bevy_zeroverse/0.23.0)
and [Burn adapter](https://crates.io/api/v1/crates/bevy_zeroverse_burn/0.6.0).
The cache identity advances to `gekko-capture=3`; previous versions remain
readable with their original fingerprints. Historical audit sections below are
retained as snapshots, not current package advice.

Independent native-image encoder qualification now uses official V-JEPA source
revision `204698b45b3712590f06245fbfba32d3be539812` and the official
[Base checkpoint](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt)
(SHA-256 `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d`).
All 158 imported encoder tensors exactly match its `ema_encoder` after F16
conversion. Using the same quantized weights expanded to F32, the CPU
implementation agrees with official PyTorch dense and sparse image outputs at
32 and 256 pixels, with relative RMS error below 0.000004. Against the original
F32 checkpoint, CPU output error is at most 0.000773 relative RMS in these cases.
CUDA comparisons at 32, 256 and 384 pixels have relative RMS error
up to 0.0029 and cosine above 0.999995: they **fail** the original 0.001 strict
RMS gate. This residual is consistent with accelerated arithmetic; CubeK can
use TF32 stages for F32 matrix inputs. It is not silently relabeled exact parity.
The source implementation/weight mapping and the GPU numerical residual are
separate findings. See `tools/interop/check_encoder_parity.py`, `examples/encoder_audit.rs`
and `.data/pilot-03/encoder-audit{,-cpu}/parity.json` for reproducible evidence.

Decoder auditing uses pinned Gekko revision
`63f0ec9957885ea82cc2f4637d502003fbf9afcb`:
[decoder call sites](https://github.com/thibautloiseau/gekko/blob/63f0ec9957885ea82cc2f4637d502003fbf9afcb/models/gekko.py)
and [attention blocks](https://github.com/thibautloiseau/gekko/blob/63f0ec9957885ea82cc2f4637d502003fbf9afcb/layers/blocks.py).
The MAE path passes the pre-block target to its cross-attention slot. The initial
Burn implementation instead used the target after self-attention. Pilot 03 adds
an explicit compatibility flag and an independent branch-equivalence test; both
arms of the larger comparison enable the corrected behavior. Spatial rotary
attention is independently implemented and tested as an optional position mode.

This is still an experimental multi-reference decoder, not an exact Gekko
reproduction. It freezes V-JEPA features, optionally appends observed RGB,
concatenates reference banks, and optionally predicts patch statistics. The
published default's self-attention query/key normalization, initialization,
dimensions and original encoder training are not all reproduced. The recipe
comparison cannot isolate the MAE correction or any individual architectural
change as the sole cause of improved reconstruction.

The final decoder CPU/CUDA prediction and input-gradient comparison passes a
0.01 relative RMS gate, with worst residual 0.002183. It covers width 384,
two blocks, batch two, a 4×4 token grid, rotary attention, corrected MAE context
and learned patch calibration. All three exported head weight arrays match
exactly. The first attempt was an invalid comparison: a saved lazy CPU clone
and independently initialized original held different weights. The corrected
audit reloads the same file in both backends. The pilot trainer already
materializes its record before cloning. Both attempts and their command time
are retained; see `.data/pilot-03/decoder-audit-correction.json` and
`decoder-backend-corrected-audit/comparison.json`.

## Subsequent published-package upgrade

The implementation later checked the registry again and pinned the latest
unyanked releases below (2026-09-28 UTC, September 27 America/Chicago). The rest
of this source audit retains its original planning snapshot. Runtime evidence,
including the newly published multi-view rig and `position=2` metadata, is in
[pilot 01](studies/pilot-01.md).

| Crate | Pinned version | Registry archive SHA-256 |
| --- | --- | --- |
| `bevy_zeroverse` | `0.22.0` | `7585402d61335cb8a024d0bc915c2cce3de1d2cca1e83fcafdf3faeabf49e069` |
| `bevy_zeroverse_burn` | `0.5.0` | `964348f4c093fcf33c51081d2eb9b5f0a6e2746e7fcb868852e87033231dfaaa` |

Primary records: [generator](https://crates.io/api/v1/crates/bevy_zeroverse/0.22.0)
and [Burn adapter](https://crates.io/api/v1/crates/bevy_zeroverse_burn/0.5.0).
Responses are archived in `.data/pilot-01/*-registry.json`; only these two package
entries changed in the capture lockfile during this upgrade.

## Local state

| Source | Observed revision | Relevant state |
| --- | --- | --- |
| `/home/mosure/repos/burn_gekko` | `8ded054`, initial commit | Initially a clean repository containing a one-line README |
| `/home/mosure/repos/burn_jepa` | `939abcea4648fd2ad0e12cb6d7bf4874f6bdf871` | Root package `burn_jepa 0.21.0`; multiple uncommitted viewer/AnyUp/high-resolution changes |
| `/media/mosure/hyper1/repos/bevy_zeroverse` | `bf0af1f83f3e34888c3ad4ccf5b246a06d585d8c` | Multiple uncommitted camera, O-voxel, generation, and documentation changes |

No applicable `AGENTS.md` was found in the inspected repositories or their
ancestor locations. No modifications were made to either source repository.
The inspected encoder files `config.rs`, `model.rs`, `tokens.rs`, `positional.rs`,
`sparse_patchify.rs`, `safetensors_io.rs`, `pipeline.rs`, `feature_memory.rs`,
`model_package.rs`, and `tests/numerical_parity.rs` had no worktree diff at this
snapshot. The surrounding manifest and exported library surface were dirty.

## Published Zeroverse artifacts

The crates.io API reported the following unyanked releases. Their downloaded
archive hashes matched the registry checksums. Both archives report the Zeroverse
revision above in `.cargo_vcs_info.json`.

| Package | Version | SHA-256 |
| --- | --- | --- |
| `bevy_zeroverse` | `0.21.0` | `9e5a5654df8c953470d391b85b97359e0276ad5fae29d90c1b0563b3055193ae` |
| `bevy_zeroverse_burn` | `0.4.0` | `6a0d85289d7a1c929d3adbdf8cd3299cbff156b422dfcb501311c8187030f846` |

Primary references: [generator registry record](https://crates.io/api/v1/crates/bevy_zeroverse/0.21.0),
[Burn integration registry record](https://crates.io/api/v1/crates/bevy_zeroverse_burn/0.4.0),
[generator source archive](https://static.crates.io/crates/bevy_zeroverse/bevy_zeroverse-0.21.0.crate),
[Burn integration archive](https://static.crates.io/crates/bevy_zeroverse_burn/bevy_zeroverse_burn-0.4.0.crate).

The normalized published manifests have no root `[patch.crates-io]` overrides.
The local generator manifest contains WGPU overrides, so performance measured
from that checkout would not automatically characterize a registry build.
Published source includes procedural indoor generation and camera path controls;
it does not contain `cameras/multiview.rs` or the local `CameraSettings.multiview`
policy. Never put that option in the initial published-generator recipe.

The archive's capture identity is:

```text
capture-v20;bevy=0.19.1;burn=0.21.0;burn_human=0.5.1;bevy_burn_human=0.6.1;burn_human_motion=0.1.1;burn_ardy=0.1.4;burn_llama=0.1.2;burn_human_inference=0.1.4;ardy_motion=5;surface_flow=1;indoor=13
```

The current local dataset guide still contains older language saying indoor
flow is rejected. Published source now configures temporal geometry attachments
for requested flow and validates exported flow buffers. Treat this as a
documentation/source discrepancy: qualify the exact recipe before temporal use.
Pairwise simultaneous-view labels will be derived from depth and calibration,
independently of temporal flow.

Useful local implementation anchors:

- `src/scene/procedural_indoor/{mod,layout,cameras,materials,validation}.rs`
- `src/sample.rs`, `src/io.rs`, `src/render/ground_truth/`
- `crates/burn/src/{generator,dataset,flow}.rs`
- `crates/burn/src/bin/zeroverse_gen.rs`
- `crates/burn/tests/{indoor_roundtrip,flow_roundtrip}.rs`
- `docs/procedural_indoor_dataset.md`, `docs/optical_flow.md`
- Local-only camera-policy discussion: `docs/multiview_cameras.md`

## Encoder facts used in the design

The root package declares `MIT OR Apache-2.0`; root license-text files were absent
from the inspected revision, so the import must resolve the canonical notices.
The only located license files belonged to the separate `bevy_burn` crate.

The root `burn_jepa` crate uses Burn/burn-store 0.21 and Rust edition 2024, with
`rust-version = 1.92`. Its encoder exposes `forward_image`, `forward_video`,
mask-batched image encoding, sparse encoder plans, and sparse patchify paths.
`VJepaEncoderOutput` carries tokens, original token indices, the grid, and optional
hierarchical features. The native image path has a distinct patch embedding and
modality embedding; it does not require inventing a video from camera views.

The Base config has width 768, 12 layers, 12 attention heads, and patch size 16.
Image token count at 384 square is 576. The config's default 64 video frames and
tubelet size 2 describe the video path, not the number of camera views.

`VJepaLoadOptions` defaults to partial loading, and the upstream loader can try
multiple encoder keys. A Gekko import must tighten the required encoder contract.
The existing real-weight parity test is ignored by default and fixture-dependent;
its existence does not prove a new encoder package is correct. Existing official
micro fixtures emphasize video inputs, so native image and sparse-image parity
are required additions.

The [official V-JEPA model constructors](https://github.com/facebookresearch/vjepa2/blob/main/src/hub/backbones.py)
select `ema_encoder` for the distilled Base/Large models and separate image/video
configuration. Use the selected constructor's exact checkpoint semantics rather
than guessing from a filename. The inspected moving source also contained a
temporary localhost download base, reinforcing the need to pin artifact URLs,
repository revisions, and hashes explicitly.

## Gekko paper and implementation

[Gekko, arXiv:2609.01530v1](https://arxiv.org/html/2609.01530v1) supplies the
three-path objective: masked cross-view reconstruction, masked monocular
reconstruction, and prediction of their relative error improvement from an
unmasked pair. The two reconstruction paths share a target mask. The paper uses
a detached, error-weighted RI loss and discusses the ambiguity of predictable
regions and the extension from pairs to reference sets. This project treats
multi-view set utility and sparse-budget utility as proposed extensions.

The inspected [official implementation](https://github.com/thibautloiseau/gekko/tree/63f0ec9957885ea82cc2f4637d502003fbf9afcb)
was pinned to `63f0ec9957885ea82cc2f4637d502003fbf9afcb`. Its
[criterion](https://github.com/thibautloiseau/gekko/blob/63f0ec9957885ea82cc2f4637d502003fbf9afcb/models/criterion.py)
has several objective modes; `MAECroCoLoss` with `ratio_style=linear` clamps the
detached MAE coefficient by epsilon. This differs at small errors from the
unclamped paper equation. The plan names both modes and tests them explicitly.

Its [model](https://github.com/thibautloiseau/gekko/blob/63f0ec9957885ea82cc2f4637d502003fbf9afcb/models/gekko.py)
shares decoder blocks but has separate MAE and cross-view output heads/mask
tokens. The MAE path supplies its own target sequence to the decoder's attention.
The generic config is not a complete training recipe: the selected experiment
must enable MAE and the intended loss. The code also includes external frozen
encoders, so a frozen-backbone substitution alone should not be sold as novelty.

Gekko/CroCo source and released Gekko weights carry CC BY-NC-SA terms, per the
[Gekko license](https://github.com/thibautloiseau/gekko/blob/63f0ec9957885ea82cc2f4637d502003fbf9afcb/LICENSE).
Plan an independently authored Burn implementation of the method, retain
attribution, and keep third-party source/weight provenance explicit. Do not assign
the imported encoder's MIT/Apache terms to unrelated borrowed code or weights.
This is an artifact-provenance requirement, not a legal conclusion about a future
distribution.

## Research and benchmark references

| Source | Role in this project |
| --- | --- |
| [V-JEPA 2.1 paper](https://arxiv.org/abs/2603.14482) and [official code](https://github.com/facebookresearch/vjepa2) | Dense visual backbone and reference inference |
| [Gekko project](https://thibautloiseau.github.io/projects/gekko/) | Original method, attribution and qualitative context |
| [CroCo official code](https://github.com/naver/croco) | Cross-view completion baseline and downstream protocol pointers |
| [MuM](https://arxiv.org/abs/2511.17309) | Multi-view masked modeling prior work |
| [Muskie](https://arxiv.org/abs/2511.18115) | Native multi-view representation prior work |
| [ETH3D overview](https://www.eth3d.net/overview) | Correspondence/geometry benchmark families; exact subset must be frozen |
| [ScanNet++ documentation](https://scannetpp.mlsg.cit.tum.de/scannetpp/documentation) | Real indoor calibration and scene splits |
| [Hypersim official repository](https://github.com/apple-aiml-research/ml-hypersim) | Independent synthetic-domain evaluation candidate |

These references establish what to investigate, not results for `burn_gekko`.
Freeze benchmark versions, pair lists, access terms, preprocessing, and evaluation
code before a confirmatory run. The paper plan includes a further related-work
review before making a novelty claim.
