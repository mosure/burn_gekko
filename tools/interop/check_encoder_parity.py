#!/usr/bin/env python3
"""Compare exported Burn parameters/outputs with pinned official V-JEPA 2.1 code.

Reference checkout, checkpoint and outputs live in .data; no automatic downloads.
The official checkpoint is loaded with weights_only=True. This is an image-path
test of both dense and genuinely sparse inputs, not a video smoke test.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--export", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, str(args.reference.resolve()))
    from app.vjepa_2_1.models.vision_transformer import vit_base

    model = vit_base(img_size=(384,384), patch_size=16, num_frames=16,
                     tubelet_size=2, use_sdpa=True, use_rope=True,
                     img_temporal_dim_size=1, interpolate_rope=True)
    model.eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state = {k.replace("module.", "").replace("backbone.", ""): v
             for k,v in checkpoint["ema_encoder"].items()}
    model.load_state_dict(state, strict=True)
    manifest = json.loads((args.export / "manifest.json").read_text())

    def read(name):
        return np.fromfile(args.export / (name+".f32"), dtype="<f4").reshape(manifest["tensors"][name])

    weight_rows = []
    imported = {}
    for name in manifest["weights"]:
        actual = read(name)
        expected = state[name].float().numpy()
        quantized = state[name].half().float().numpy()
        assert actual.shape == expected.shape, (name, actual.shape, expected.shape)
        weight_rows.append(dict(name=name, max_abs=float(np.abs(actual-expected).max()),
                                quantized_max_abs=float(np.abs(actual-quantized).max())))
        imported[name] = torch.from_numpy(actual.copy())
    assert set(imported) == set(state), (set(imported)-set(state), set(state)-set(imported))
    outputs = []
    for source, weights in [("official_f32", state), ("imported_f16_as_f32", imported)]:
        model.load_state_dict(weights, strict=True)
        for size in sorted(int(k.split("-")[1]) for k in manifest["tensors"] if k.startswith("input-")):
            rgb = torch.from_numpy(read(f"input-{size}").copy()).unsqueeze(2)
            mask = torch.tensor([json.loads((args.export/f"mask-{size}.json").read_text())])
            for mode, indices in [("dense", None), ("sparse", mask)]:
                with torch.inference_mode():
                    expected = model(rgb, masks=indices).numpy()
                actual = read(f"{mode}-{size}")
                delta = actual-expected
                outputs.append(dict(source=source, size=size, mode=mode,
                    max_abs=float(np.abs(delta).max()), rmse=float(np.sqrt(np.mean(delta**2))),
                    relative_rmse=float(np.sqrt(np.mean(delta**2)/np.mean(expected**2))),
                    cosine=float(np.sum(actual*expected)/np.sqrt(np.sum(actual**2)*np.sum(expected**2)))))
    report = dict(weights=weight_rows, outputs=outputs,
                  max_quantized_weight_error=max(r["quantized_max_abs"] for r in weight_rows))
    (args.export/"parity.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="weights"}, indent=2))
    # GPU matmul rounding may accumulate; this is deliberately much tighter than feature scale.
    assert report["max_quantized_weight_error"] == 0, "package differs from official EMA checkpoint"
    assert all(r["relative_rmse"] < 1e-3 and r["cosine"] > 0.99999
               for r in outputs if r["source"]=="imported_f16_as_f32"), "encoder implementation parity failed"


if __name__ == "__main__":
    main()
