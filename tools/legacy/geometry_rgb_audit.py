#!/usr/bin/env python3
"""Validation-only RGB/geometry alignment check. Uses ground-truth 3D: NOT a model result."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors.numpy import load
import zstandard


def main():
    p=argparse.ArgumentParser();p.add_argument("--dataset",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.dataset/"manifest.json").read_text())
    rooms=[r for r in manifest["scenes"] if r["split"]=="validation"]
    rows=[];panels=[]
    for i,entry in enumerate(rooms):
        with (args.dataset/"raw"/entry["file"]).open("rb") as f:
            with zstandard.ZstdDecompressor().stream_reader(f) as stream:
                blob=stream.read(128*1024*1024+1)
        assert len(blob)<=128*1024*1024
        t=load(blob);rgb=t["color"][0,0];depth=t["depth"][0,0,...,0]
        v,h,w,_=rgb.shape;aabb=t["aabb"][0];pos=t["position"][0,0]*(aabb[1]-aabb[0])+aabb[0]
        camera=t["world_from_view"][0,0];fovy=t["fovy"].reshape(v)
        world=pos[0].reshape(-1,3)
        yy,xx=np.mgrid[:h,:w];covered=np.zeros((h,w),bool)
        warp=np.full((h,w,3),.25,dtype=np.float32);identity=warp.copy()
        max_self=0.
        for ref in range(v):
            local=(world-camera[ref,3,:3]) @ camera[ref,:3,:3].T
            z=-local[:,2];fy=h/(2*np.tan(fovy[ref]/2))
            u=(fy*local[:,0]/np.maximum(z,1e-12)+w/2).reshape(h,w)
            y=(-fy*local[:,1]/np.maximum(z,1e-12)+h/2).reshape(h,w);z=z.reshape(h,w)
            valid=(depth[0]>0)&(z>0)&(u>=.5)&(u<w-.5)&(y>=.5)&(y<h-.5)
            ix=np.floor(u).astype(int).clip(0,w-1);iy=np.floor(y).astype(int).clip(0,h-1)
            if ref==0:
                max_self=float(np.sqrt((u-(xx+.5))**2+(y-(yy+.5))**2)[depth[0]>0].max())
                assert max_self<.1
                continue
            valid &= (depth[ref,iy,ix]>0)&(np.abs(z-depth[ref,iy,ix])<=np.maximum(.02,.01*z))
            x0=np.floor(u-.5).astype(int).clip(0,w-2);y0=np.floor(y-.5).astype(int).clip(0,h-2)
            wx=(u-.5-x0).clip(0,1)[...,None];wy=(y-.5-y0).clip(0,1)[...,None]
            color=(1-wy)*((1-wx)*rgb[ref,y0,x0]+wx*rgb[ref,y0,x0+1])+wy*((1-wx)*rgb[ref,y0+1,x0]+wx*rgb[ref,y0+1,x0+1])
            use=valid&~covered;warp[use]=color[use];identity[use]=rgb[ref][use];covered|=valid
        assert covered.any()
        mse=float(np.mean((warp[covered]-rgb[0][covered])**2));unwarped=float(np.mean((identity[covered]-rgb[0][covered])**2))
        rows.append(dict(seed=entry["seed"],target=0,covered_fraction=float(covered.mean()),
            geometry_oracle_rgb_mse=mse,geometry_oracle_psnr=float(-10*np.log10(max(mse,1e-12))),
            same_pixel_reference_mse=unwarped,max_self_reprojection_pixels=max_self))
        if i in [0,len(rooms)//4,len(rooms)//2,3*len(rooms)//4]:panels.append((entry["seed"],rgb[0],rgb[1],warp,covered))
    report=dict(dataset_id=manifest["dataset_id"],artifact_kind="ground_truth_geometry_diagnostic_not_model_output",
                interpolation="bilinear at pixel centers; first depth-consistent reference; grey uncovered pixels",
                rooms=rows,mean_oracle_mse=float(np.mean([r["geometry_oracle_rgb_mse"] for r in rows])),
                mean_unwarped_mse=float(np.mean([r["same_pixel_reference_mse"] for r in rows])))
    (args.output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    fig,axes=plt.subplots(len(panels),3,figsize=(10,3*len(panels)),squeeze=False)
    for row,(seed,target,ref,warp,_) in zip(axes,panels):
        for ax,im,title in zip(row,[target,ref,warp],[f"Target {seed}","Reference 1","GT geometry warp (ORACLE)"]):
            ax.imshow(im,interpolation="nearest");ax.set_title(title,fontsize=9);ax.axis("off")
    fig.tight_layout();fig.savefig(args.output/"geometry-oracle.png",dpi=140,bbox_inches="tight");plt.close(fig)
    print(json.dumps({k:v for k,v in report.items() if k!='rooms'},indent=2))


if __name__=="__main__":main()
