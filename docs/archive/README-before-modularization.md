# burn_gekko

Burn training pipeline for independently encoded V-JEPA 2.1 views, multi-view
latent prediction and relative-improvement objectives, and published Zeroverse rooms.

**Primary task: V-JEPA 2.1 latent prediction.** The `latent_pilot`
trains random multi-view fusion and a shared cross-view/monocular latent head
against a fixed MIT V-JEPA teacher, with gated student unfreezing and geometric
co-visibility diagnostics. See the [latent pipeline](../latent-pipeline.md).
RGB reconstruction remains a separate diagnostic; its previous blur is unresolved.

[Pilot 07 continuation and annotated PDF](../studies/pilot-07-latent-continuation.md)
compares random and compact masks, then continues the selected model with exact
optimizer state. Under identical validation inputs, latent MSE falls 8.9% versus
the parent and patch-grid encoder EPE improves from 30.34 to 29.41px. It includes
a new 256-room holdout, RGB-only HPatches transfer, frozen-encoder controls and
an experimentally verified fix for growing partial-unfreeze GPU memory.
Fresh-room latent error improves 8.8%, and adapted encoder error improves 4.5%
on the declared HPatches viewpoint readout. At that checkpoint, decoder and
attention matching regressed against the frozen encoder on that test.
This is a research prototype; [SOTA qualification](../sota-evidence.md) remains
open. The [initial latent screen](../studies/pilot-07-latent.md) is preserved separately.

The [fusion-transfer study](../studies/pilot-07-fusion-transfer.md) tests positional
encoding, dense affinity guidance, decoder alignment and encoder adaptation
under matched controls. Its hierarchy probe identifies a stronger block-6
encoder baseline, which subsequent fusion comparisons retain. Block-6 affinity
guidance, direct spatial input and gated encoder adaptation improve transfer.
The [annotated final PDF](../../.data/pilot-07/fusion-audit/fusion-transfer-final-report.pdf)
includes the sealed independent evaluation: ETH3D decoder AEPE falls from
53.76 to 39.16 pixels and attention AEPE from 100.83 to 40.30; fresh-room latent
MSE falls 8.65%. All three fusion readouts still trail their stronger block-6
encoder controls. The fusion weakness is reduced, not resolved.

[GPU profiling and repair](../gpu-efficiency.md) found slow pitched RGB
uploads despite high activity counters. Contiguous uploads pass a matched
64-update check with identical logged losses and validation MSE, **1.63x faster
steady updates**, and **8.84% less complete-command board energy**. Full study
runs retain power telemetry and the existing cumulative compute ceiling.
The completed study uses 10.345 of its 12 GPU-command hours. A final exact replay
still measures substantial dispatch/synchronization gaps; high GPU activity
alone is not treated as efficient execution.

**Project requirement: end-to-end reconstruction training without noncommercial
pretrained weights.** The `e2e_pilot` trainer initializes fusion and RGB/RI heads
from scratch, with either a random image encoder or the audited MIT V-JEPA 2.1
Base package. It trains encoder parameters, supports gated progressive unfreezing,
and saves both optimizers for continuation. See the [end-to-end guide](../e2e-pipeline.md).

[Pilot 05 results and annotated PDF](../studies/pilot-05.md) cover four 4,000-update
experiments, two fit diagnostics, and 64 reserved test rooms in 114.7 minutes of
capture/experiment time. Encoder training works; **held-out blur and useful
co-visibility remain unresolved**. No candidate passed the quality gates.

[Pilot 06](../studies/pilot-06.md) regenerated 8,192 training / 128 validation / 128 test
rooms at 256px with published Zeroverse 0.25.0 / Burn adapter 0.8.0. The selected
lineage contains 37,230 updates and 531,680 target exposures without noncommercial
weights. Reserved-test PSNR is 27.06 dB and cross-view MSE improves 10.7% over the
monocular branch, but **blur remains unresolved**: gradient energy is 0.222
against the retained 0.5 minimum. The study stays inside its 12-hour cumulative
command ceiling; the [annotated PDF](../../.data/pilot-06/burn_gekko_pilot_06_report.pdf)
includes unsuccessful experiments, raw samples and a measured process slowdown.

```sh
cargo build --profile pilot --features cuda --locked --bin e2e_pilot -j 4
# Use a fresh output directory; this recipe is an explicitly bounded GPU study.
target/pilot/e2e_pilot --config configs/pilot05-vjepa-no-energy.toml \
  --run .data/runs/my-vjepa-run
```

`configs/pilot05-scratch.toml` trains the image encoder and fusion model jointly
from random initialization. The V-JEPA recipes start with the encoder frozen,
then train its final blocks and finally all image blocks and the patch stem
after validation gates pass. The new trainer caches RGB, recomputes features
every update, and never imports released Gekko weights or uses them as a teacher.
Explicit warm starts from this project's own audited checkpoints record parent
hashes and optimizer resets; exact continuation restores both optimizers.

Pilot 04 is a **noncommercial reference baseline only**: its released Gekko
encoder and decoder improved synthetic-room reconstruction, but those weights and
derived checkpoints do not meet the project's initialization requirement. They
are excluded from the new trainer, including as teachers or decoder warm starts.
See its historical [report and before/after PDF](../studies/pilot-04.md)
and [hybrid pipeline](../hybrid-pipeline.md). The earlier trainer implements
the pinned encoder import, offline capture/cache, RGB-only training, optimizer
checkpoints, batching, a frozen feature cache, and held-out diagnostics. See the
[pilot 03 diagnosis, controlled experiments and PDF](../studies/pilot-03.md),
[pilot 02 report and annotated evaluation](../studies/pilot-02.md), earlier
[pilot 01 measurements](../studies/pilot-01.md), and historical
[initial implementation record](../implementation-status.md).

Capture uses published **`bevy_zeroverse=0.25.0`** and
**`bevy_zeroverse_burn=0.8.0`**, pinned with their lockfile in the isolated tool.

The hybrid entry point is `hybrid_pilot`; its TOML recipes are
`configs/pilot04-hybrid-{smoke,screen,main,detail}.toml`. It uses the pinned local
Gekko weights documented in [the hybrid guide](../hybrid-pipeline.md), which
carry CC-BY-NC-SA-4.0 terms independently of this workspace's code license.
`configs/pilot04-hybrid-test.toml` evaluates the selected checkpoint; choose a
fresh `.data/` output directory. These larger recipes do not run by default.

The default config performs **two steps with a random tiny encoder** on a cached
dataset. The V-JEPA-base config instead loads a real local weight package and
keeps its encoder frozen. No automatic model downloads or full training runs occur.

User configs in `configs/` and study command plans use **TOML**. Resolved runs
save `config.toml` and `pilot-config.toml`; captures save `capture.toml`.
Metrics, manifests, and reports remain JSON/JSONL. Historical JSON config
snapshots remain readable for evaluation and capture recovery; new CLI config
inputs require TOML. Upstream encoder package/checkpoint formats are unchanged.

```sh
cargo test --workspace --locked
cargo build --manifest-path tools/zeroverse_capture/Cargo.toml --locked -j 4
cargo run --locked --bin gekko -- capture --dry-run
cargo run --locked --bin gekko -- capture
```

Capture prints `.data/datasets/<id>`. The default is four independent rooms,
two static cameras per room, one instant, 64×64 pixels, RGB/depth/position, with
train/validation/test room counts 2/1/1. The capture fingerprint includes its
binary hash and configuration. Repeating it validates and reuses existing data.
Partial captures are preserved for inspection and are never accepted as datasets.
After an adapter repair, `recover-capture --staging .data/datasets/<partial>`
can explicitly revalidate an existing complete capture, with the original
generator binary. It verifies identity, hashes, seeds, and geometry before publishing
the cache entry. It does not render or bypass validation.

```sh
# Replace <id> with the capture output; use a fresh run name for each invocation.
cargo run --locked --bin gekko -- verify-dataset --dataset .data/datasets/<id>
cargo run --locked --bin gekko -- audit-geometry --dataset .data/datasets/<id>
cargo run --locked --bin gekko -- train-preflight \
  --dataset .data/datasets/<id> --run cpu-check
cargo run --locked --bin gekko -- eval-preflight \
  --dataset .data/datasets/<id> --run .data/runs/cpu-check
```

For the local pretrained package, stage `manifest.json`, `jepa.bpk.parts.json`,
and its listed shards under `.data/models/vjepa2_1_base/`. This workspace already
has a copy of the existing local package; it is ignored by Git. The loader checks
all shard hashes and required tensors before training. Its F16-stored weights are
expanded to F32. Pilot 03 independently verifies all 158 tensors against the
official EMA checkpoint after F16 conversion, and verifies dense/sparse CPU
image-forward parity. CUDA has a small recorded numerical residual and fails
the original strict RMS threshold; see the [source audit](../source-audit.md).

```sh
cargo run --locked --features cuda --bin gekko -- train-preflight \
  --dataset .data/datasets/<id> --run cuda-check --backend cuda \
  --config configs/train-vjepa-base-preflight.toml
cargo run --locked --features cuda --bin gekko -- eval-preflight \
  --dataset .data/datasets/<id> --run .data/runs/cuda-check --backend cuda
```

Data, weights, logs, checkpoints, evaluation reports, and test fixtures live in
`./.data/`. Cargo build artifacts live in the respective ignored `target/` folders.
Capture is capped at 32,768 total rooms, four cameras, and 512×512 pixels.
`e2e_pilot` supports up to 500,000 updates, batch size 32 and a 12-hour wall limit,
with explicit RGB-cache memory bounds. Default configs remain tiny.
`train-preflight` remains capped at 16 steps, batch size one, 128×128 pixels,
and a 64-wide/two-layer decoder. `train-pilot` permits up to 100,000 steps,
batch size 16, 512×512 pixels, and a 512-wide/eight-layer decoder, with an
explicit wall limit of at most two hours. The default configs remain tiny.
Resume uses a new run
directory plus `--resume .data/runs/<previous>/checkpoint-000001`, with `steps`
set to the desired **total**. Model/optimizer checksums, dataset, encoder,
configuration, code identity, and backend must match. The test suite checks exact
CPU continuation; the pilot also checks CUDA continuation against an uninterrupted run.

For the bounded three-view pilot (12 train / 2 validation / 2 test rooms):

```sh
cargo build --profile pilot --features cuda --locked -j 4
target/pilot/gekko capture --config configs/capture-pilot.toml
target/pilot/gekko train-pilot --dataset .data/datasets/<id> \
  --config configs/pilot-smallset.toml --run smallset-01 --backend cuda
target/pilot/gekko eval-preflight --dataset .data/datasets/<id> \
  --run .data/runs/smallset-01 --backend cuda --step 0
target/pilot/gekko eval-preflight --dataset .data/datasets/<id> \
  --run .data/runs/smallset-01 --backend cuda --step 128
```

`configs/pilot-*.toml` also define online batch-one/batch-four timing screens,
the cached batch-four control, and fixed-example overfit. Each run saves periodic
model/optimizer checkpoints, fixed train/validation probes, per-step losses,
sampling identities, and timing reports. Training visits every room/target pair.
The optional full-view feature cache is resident per run; masked targets are
always re-encoded. Test rooms are excluded from training and monitoring.
Pilots can enable reproducible shuffled room/target epochs and a linear-warmup,
cosine-decay learning-rate schedule. Its absolute step horizon is retained on
resume. Evaluation accepts an optional TOML `--options` file for unrelated-room
reference controls and predefined annotated sample exports under `.data/`.

Reconstruction studies can set `image_features = "semantic_rgb"` to append
observed RGB patches to frozen semantic features; hidden target patches are
excluded. `decoder_position = "rope2d"` enables spatial rotary attention.
`normalize_targets = true` with `predict_patch_stats = true` learns patch
brightness/contrast alongside normalized content, allowing standalone RGB
without hidden target statistics. Its additional calibration loss is included
in total loss; the reported content and RI losses retain their original units.
`mae_context_before_self = true` selects the audited Gekko pre-block context
semantics. Defaults preserve historical checkpoint behavior.

Evaluation records unclipped hidden-pixel RGB MSE and labels normalized-head
displays that use target statistics as oracle visualizations. Annotated
completion panels copy only visible input patches. Geometry is reserved for
evaluation. `tools/reconstruction_diagnostics.py` independently checks exported
float arrays and measures edge fidelity; `tools/geometry_rgb_audit.py` provides
a separate, explicitly labeled ground-truth geometry warp diagnostic.

`tools/run_study.py` runs a TOML command plan with a cumulative wall ceiling,
per-command logs, and GPU telemetry. It never retries a failed command or extends
the budget. The recorded plan/protocol and all measurements are in
`.data/pilot-01/`; `tools/analyze_pilot.py` regenerates the summary and figures.
Study plans use a root `max_command_seconds` and repeated `[[commands]]` tables
with `name`, `max_seconds`, and an `argv` string array (Python 3.11+):

```toml
max_command_seconds = 60

[[commands]]
name = "capture-config-check"
max_seconds = 30
argv = ["target/debug/gekko", "capture", "--dry-run", "--config", "configs/capture-pilot.toml"]
```

Run a plan with `python3 tools/run_study.py --plan <plan.toml> --output .data/<study>`.
The original pilot's JSON plans are historical receipts; translate their config
paths to `.toml` when preparing a new study.

The original small decoder accepts one to three reference views using order-invariant
joint attention. The qualified hybrid instead averages shared pairwise RGB outputs;
its reference order is also invariant. These are separate extensions of pairwise
Gekko. Target masking
happens before encoder attention. Reference encoders currently use dense tokens;
sparse reference selection and specialized sparse patch kernels remain future work.

The [roadmap](../../ROADMAP.md) retains the wider research plan:

- [Architecture, losses, and sparse execution](../architecture.md)
- [Published generator, dataset contracts, and co-visibility](../data-pipeline.md)
- [Repository organization and encoder import](../repository-plan.md)
- [Training stages and resource planning](../training-plan.md)
- [Evaluation, metrics, and experiment matrix](../evaluation-plan.md)
- [Paper development and evidence requirements](../paper-plan.md)
- [Source audit and references](../source-audit.md)

Run `gekko --help` for the implemented CLI. Commands in the older planning
documents are specifications unless also documented here.
