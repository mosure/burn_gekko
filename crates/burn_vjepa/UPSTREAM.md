This is a deliberately small import of the root `burn_jepa` crate from
`mosure/burn_jepa`, revision `939abcea4648fd2ad0e12cb6d7bf4874f6bdf871`.
`UPSTREAM.json` preserves the SHA-256 of the original six imported source files.
Five remain verbatim. `LOCAL_MODIFICATIONS.toml` records the adapted model file;
do not interpret its upstream hash as the current local hash.
`tools/interop/verify_encoder_import.py` checks both records: each adaptation must bind
to the original revision and file hash, and its current contents must match
the separately recorded local hash. Further unrecorded changes are rejected.

The upstream Cargo manifest declares `MIT OR Apache-2.0`. This distribution
includes both license texts from the same author's `crates/bevy_burn` directory
in `mosure/burn_jepa`. The original six-file attribution and hashes remain intact.
The standalone package is named `burn_vjepa` to distinguish this focused fork
from the independently published `burn_jepa` package.

Local integration adds a minimal manifest and export surface. Viewer, AnyUp,
RAC, temporal caches, dataset code, and upstream training tools are excluded.
The image encoder, video encoder, predictor, positional encodings, sparse mask
plans, and checkpoint conversion code are retained together to preserve their
internal contracts. The predictor is never optimized by the Gekko trainer.

`src/package.rs` adapts the pinned upstream `model_package.rs` F16 load adapter
and multipart application logic, adding required-tensor and duplicate checks.
`tests/checkpoint.rs` adapts the native Safetensors round-trip test from upstream
`tests/numerical_parity.rs` to exercise dense and sparse image forwards. These
local adaptations are separate from the original import in `UPSTREAM.json`.
The imported model already exposes staged parameter unfreezing. The local model
adaptation adds an explicit image capture-layer policy, as verified against the
pinned upstream Git object. The latent trainer requests only final features, avoiding
unused differentiable hierarchical branches. Its regression test checks dense
and sparse output/gradient parity against the default forward. The default
hierarchical encoder interface is preserved.
Two module-scoped allowances silence newer Clippy lints in the imported files;
all local code is still checked with warnings denied.

Default builds use generic patch projection followed by token selection before
transformer attention. Sparse patch projection kernels are optional and are not
claimed to be exercised by the default preflight. Native image encoding uses
`image_patch_embed`; separate camera views are never passed as video frames.

## Pretrained V-JEPA 2.1 weight notice

`VJEPA21_WEIGHTS_LICENSE` is the exact MIT license from the audited Meta source
commit `204698b45b3712590f06245fbfba32d3be539812`. It is copied into checkpoints
that derive from the audited V-JEPA 2.1 Base weights. Its source URL and the
official weight checksum are retained in `.data/pilot-06/licenses/`. Random
encoder training does not load those pretrained weights.
