# Live demo qualification

The demo uses `pilot13-head-stability15-f32`, with exact float32 foundation hash
`0639afa2e19d0e7ac5c9c1cc8caadc75d0dcf10f105ee2870607856b7a1ed5a7`
and the same attached camera/RGB heads as the current project page. This is an
inference and viewer qualification, not a new training study or benchmark claim.

Local verification on 2026-10-01 used Linux, Rust 1.98, Burn 0.21, Bevy 0.19.1,
published Zeroverse 0.26.0, Chrome 153.0.8010.52 and an NVIDIA RTX PRO 6000
Blackwell. The final Chrome run used normal GPU settings, without unsafe-WebGPU
or graphics-backend overrides. The desktop shared the GPU. No training was performed.

The native test captures three room cameras, runs real inference, saves source
images/predictions/raw metrics and a window screenshot, and exits. Headed browser
verification exercises target switching, editor orbit/placement, inference reruns,
local file upload, stale-result rejection, resize and regeneration. Rendering and
annotations were inspected; successful compilation alone is not the evidence.

For the same three uploaded native captures, target 1 and the same 26 visible
patches, the independent numerical checks were:

| Backend | Hidden-pixel RGB PSNR |
| --- | ---: |
| Native NdArray CPU reference | 20.67103713 dB |
| Native fused WGPU | 20.67103925 dB |
| Browser unfused WebGPU | 20.67104987 dB |

The browser/native maximum absolute difference across the 11 raw camera outputs
was 0.00000155. RI scores differed by at most 0.00000826; the 24 coarse match
coordinates were identical, with maximum descriptor-cosine difference 0.00000334.
The monocular control differed by 0.00000775 dB. These checks qualify this fixture and environment,
not all devices or inputs. The CPU example took 4.10 seconds for one forward call;
that is a diagnostic observation, not a throughput or efficiency benchmark.

Two integration regressions are covered explicitly: float render targets need
Zeroverse's linear-to-sRGB transfer exactly once, and PanOrbit's elevation has
the opposite sign from its X rotation. Readback and camera-focus tests protect
those contracts. Editor home uses Zeroverse's original interior pose, keeping the
room visible instead of looking down at its opaque roof. Changes to scene inputs
invalidate already-running predictions.

**The browser must currently use unfused WebGPU.** Burn 0.21's fused browser path
matched dense camera outputs but changed completion by 1.7738 dB and also changed
RI scores on this fixture. Direct `CubeBackend` restored completion parity.
A second discrepancy remained in the scalar RI matrix projection (maximum score
difference 0.56892). An explicit multiply/reduce with the exact trained weight and
bias restored RI parity, and a CPU regression checks equivalence to `Linear`.
No weights or activations were changed.

The browser smoke test rejects completion or monocular PSNR differences above
0.05 dB, raw camera/RI/cosine differences above 0.005, or changed match coordinates.
Do not re-enable fusion or the scalar matmul path without passing all these gates.

The native input PNG SHA-256 values are:

```text
input-0.png 2c6ae77b28b192e534e568d7a5b081d4dd941ee1af505b4df35e9cb6750e164c
input-1.png 9058c2a172dcaea6deb5c43ccfdd4bfef7d3ae97572eb1d4d9d475715fa15af7
input-2.png 5e190d19b56cd488a052d59d8a2a5c801ea823d983633bcb52074fde9c2cb9ca
```

Raw local evidence is under `.data/live-demo/`: `native/receipt.json`, the native
input/window PNGs, `cpu-reference.json`, and browser trace, result and screenshots.
The final native scene and uploaded-input runs are in `native-final/` and
`native-uploads/`; the latter correctly omits camera ground truth.
Use the [demo README](../crates/bevy_gekko/README.md) to reproduce the checks.
Model distribution is a checksummed release asset with licenses/provenance;
the Pages workflow compiles the two WASM modules and requires green source CI.
RGB remains blurry and the calibration head remains a research prototype.
