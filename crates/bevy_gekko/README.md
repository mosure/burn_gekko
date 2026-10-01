# bevy_gekko

Native and WASM live demo using the published Zeroverse viewer, its editor camera,
and the same `burn_gekko_inference` model/metrics library. The Bevy renderer has a
separate workspace to keep renderer dependencies out of the training workspace.

Inference is requested explicitly and runs on a native worker thread or a Web
Worker. Scene and upload revisions invalidate old results. Number keys select
target views; the editor remains available for inspection and camera placement.

## Run

[Open the WebGPU demo](https://mosure.github.io/burn_gekko/demo/), or from the repo root:

```sh
cargo run --locked --manifest-path crates/bevy_gekko/Cargo.toml
```

The scene is available before model loading. Click **Load trained model** (393 MiB)
and then **Run inference**. Native Linux builds need the usual Bevy development
packages (`libasound2-dev`, `libudev-dev`, `pkg-config`). Browser inference requires
WebGPU on HTTPS or localhost. The first inference also compiles GPU kernels.

| Control | Action |
| --- | --- |
| `1`–`3` | Select the target camera and visit its position |
| `0` | Return to the initial interior editor view |
| Drag / right drag / wheel | Zeroverse editor orbit / pan / zoom |
| `C` | Move the selected capture camera to the current editor pose |
| `Shift` + `1`–`3` | Place that capture camera at the current editor pose |
| `Space` or `I` | Capture the current views and rerun inference |
| `N` | Generate another seeded procedural room |
| `U` | Upload two to four PNG/JPEG photographs of one scene |
| `L` | Load the trained model |

Uploads use a centered square crop resized to 256 × 256; images stay on the device.
In image mode, number keys `1`–`4` select the target. Choose **Zeroverse scene** to
return to the editor. Native startup also accepts repeated `--image path.png`.
Editor motion alone changes your inspection view; `C` explicitly places a capture
camera. Changes to capture cameras, target selection, room or uploads invalidate
annotations, including predictions still in flight.

## What the annotations mean

The demo runs the selected V-JEPA/fusion checkpoint and attached head-stability15
heads, without noncommercial weights. RGB completion receives only the visible
10% of target patches and the reference images. The full target is used afterward
for hidden-pixel PSNR. A separate dense RGB-pair route predicts rotation, signed
translation direction and normalized focal lengths; renderer truth is scored only
after inference and is absent for uploads. Translation has no metric scale.

Matching lines show mutual nearest 16-pixel descriptors, not geometric truth.
The RI heatmap is an unconstrained relative-improvement score, not a probability.
RGB remains blurry and camera calibration overfits. Live/new-renderer examples
are demonstrations, separate from the fixed offline evaluation protocol.

## Model export and browser build

```sh
cargo run --locked --release -p burn_gekko_inference --features ndarray \
  --bin gekko-demo-export -- --config configs/demo/model.toml --output .data/demo-model
cargo run --locked --manifest-path crates/bevy_gekko/Cargo.toml -- --model .data/demo-model

rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.129 --locked
tools/demo/build_web.sh .data/web .data/demo-model
python3 -m http.server 9917 --directory .data/web
```

The exporter requires the local experiment checkpoint artifacts named by the TOML
recipe. Otherwise download `burn-gekko-demo-heads15.tar` from the repository's
`demo-weights-heads15-v1` release; the deployment workflow pins its checksum.
Weights are copied exactly as float32 MessagePack records, split into 32 MiB parts.
There is no random fallback, quantization, Python inference or prediction replay.

`burn_gekko_inference` is compiled into a separate WASM module in a Web Worker;
the Bevy renderer and editor remain on the main thread. Rust metrics are shared
with the offline evaluation crates. Rendering uses reactive updates and inference
is on demand. The static site includes the project page, PDF, demo, model licenses
and provenance; no inference server or cross-origin weight host is required.
The browser explicitly uses Burn's unfused GPU backend: the fused WebGPU path
failed completion parity for this checkpoint. Native GPU, unfused browser GPU
and an independent NdArray CPU reference agree on the fixed uploaded inputs.
The scalar RI readout uses an equivalent explicit reduction to avoid a second
browser matmul discrepancy; parity checks cover every displayed prediction head.

## Verification

```sh
cargo test -p burn_gekko_inference --features ndarray --locked
cargo test --manifest-path crates/bevy_gekko/Cargo.toml --locked
cargo run --locked --manifest-path crates/bevy_gekko/Cargo.toml -- \
  --model .data/demo-model --auto-infer --smoke-output .data/live-demo/native
npm ci --prefix tools/demo
DISPLAY=:0 node tools/demo/smoke.mjs
```

Native smoke writes captured inputs, predictions, raw metrics and a window PNG,
then exits. Browser verification uses a headed Chrome with a real WebGPU adapter,
exercises camera switching/placement, reruns, stale-result rejection, the file
picker, resizing and regeneration, and compares uploaded-input outputs to native
inference. Set `DEMO_URL`, `DEMO_OUTPUT`, `DEMO_FIXTURES` or `CHROME` as needed.
Build/CI success alone does not establish rendered output or GPU numerical parity.

An optional CPU reference uses the same bundle and image decoder:

```sh
cargo run --release -p burn_gekko_inference --features ndarray --example check_bundle -- \
  .data/demo-model .data/cpu-reference.json view1.png view2.png view3.png
```
