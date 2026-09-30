#!/usr/bin/env python3
"""Audit exported RGB predictions and render completion panels without enhancement.

Metrics use unclipped predictions; displayed RGB is clipped to [0,1]. Completion
composites copy only the visible input patches. Normalized-head inversions are
explicitly marked as oracle displays throughout.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pilot_report_data import sample_arrays


def inspect_sample(directory):
    a = sample_arrays(directory)
    m = a["meta"]
    h,w,p = m["height"],m["width"],m["patch_size"]
    target = a["target"]
    patches = target.reshape(h//p,p,w//p,p,3).transpose(0,2,1,3,4).reshape(-1,p*p*3)
    oracle=m["normalize_targets"] and not m.get("predicted_patch_statistics",False)
    def rgb_prediction(head):
        pred = np.fromfile(Path(directory)/f"{head}.f32",dtype="<f4").reshape(patches.shape)
        if m.get("predicted_patch_statistics",False):
            statistics=np.fromfile(Path(directory)/f"{head}-statistics.f32",dtype="<f4").reshape(-1,2)
            pred=pred*np.exp(statistics[:,1:2].clip(-10,1))+statistics[:,0:1]
        elif m["normalize_targets"]:
            pred=pred*np.sqrt(patches.var(axis=1,ddof=1,keepdims=True)+1e-6)+patches.mean(axis=1,keepdims=True)
        return pred.reshape(h//p,w//p,p,p,3).transpose(0,2,1,3,4).reshape(h,w,3)
    pred=rgb_prediction('cross')
    hidden = a["hidden"]
    mse = float(np.mean((pred[hidden]-target[hidden])**2))
    mae_rgb_mse=float(np.mean((rgb_prediction('mae')[hidden]-target[hidden])**2))
    if "masked_rgb_mse" in m["metrics"]:
        np.testing.assert_allclose(mse, m["metrics"]["masked_rgb_mse"],rtol=2e-5,atol=2e-8)
    residuals, truth_edges, pred_edges = [], [], []
    for axis in [0,1]:
        # Restrict to edges whose two pixels are both hidden, excluding observed boundaries.
        valid = np.take(hidden,range(hidden.shape[axis]-1),axis=axis) & np.take(hidden,range(1,hidden.shape[axis]),axis=axis)
        gt = np.diff(target,axis=axis)[valid]
        pr = np.diff(pred,axis=axis)[valid]
        residuals.append(np.abs(gt-pr).reshape(-1))
        truth_edges.append(gt.reshape(-1));pred_edges.append(pr.reshape(-1))
    gt,pr = np.concatenate(truth_edges),np.concatenate(pred_edges)
    composite = a["cross"].copy();composite[~hidden] = target[~hidden]
    a["completion"] = composite
    return a,dict(room_seed=m["room_seed"], oracle_statistics=oracle,
        hidden_rgb_mse=mse,hidden_rgb_psnr=float(-10*np.log10(max(mse,1e-12))),
        mae_hidden_rgb_mse=mae_rgb_mse,
        hidden_edge_mae=float(np.concatenate(residuals).mean()),
        edge_cosine=float(np.dot(gt,pr)/max(np.linalg.norm(gt)*np.linalg.norm(pr),1e-12)),
        predicted_to_true_edge_energy=float(np.mean(pr**2)/max(np.mean(gt**2),1e-12)),
        out_of_range_fraction=float(np.mean((pred[hidden]<0)|(pred[hidden]>1))))


def panel(directories, output):
    arrays = [inspect_sample(p) for p in directories]
    fig,axes=plt.subplots(len(arrays),5,figsize=(13,2.9*len(arrays)),squeeze=False)
    for row,(a,metrics) in zip(axes,arrays):
        images=[a["target"],a["masked"],a["references"][0],a["references"][1],a["completion"]]
        labels=["Target", "Input (75% hidden)", "Reference 1", "Reference 2",
                "Completion (oracle statistics)" if metrics["oracle_statistics"] else "Completion (standalone RGB)"]
        for ax,im,label in zip(row,images,labels):
            ax.imshow(im,interpolation="nearest");ax.set_title(label,fontsize=9);ax.axis("off")
        row[0].set_ylabel(str(metrics["room_seed"]))
        row[-1].text(.5,-.05,f"Hidden PSNR {metrics['hidden_rgb_psnr']:.2f} dB; edge cosine {metrics['edge_cosine']:.3f}",
                    transform=row[-1].transAxes,ha="center",fontsize=8)
    fig.tight_layout()
    fig.savefig(output,dpi=150,bbox_inches="tight");plt.close(fig)
    return [r for _,r in arrays]


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--samples",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    directories=sorted(args.samples.glob("room-*-view-*"))
    rows=[inspect_sample(d)[1] for d in directories]
    indices=sorted(set(np.linspace(0,len(directories),5,endpoint=True,dtype=int)[:-1]))
    panel([directories[i] for i in indices],args.output)
    args.output.with_suffix(".json").write_text(json.dumps(rows,indent=2)+"\n")


if __name__ == "__main__":
    main()
