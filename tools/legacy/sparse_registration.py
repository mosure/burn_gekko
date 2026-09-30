#!/usr/bin/env python3
"""Diagnostic visible-only RGB registration baseline, not learned Gekko output.

A local translation cost volume is evaluated using observed pixels only. No
hidden target pixel, camera, depth, normal or co-visibility label is an input.
"""
import argparse,json,shutil,time
from pathlib import Path
import cv2
import numpy as np
from transport_diagnostics import inspect
cv2.setNumThreads(1)


def align(observed,visible,reference):
    h,w=visible.shape
    y,x=np.mgrid[:h,:w].astype(np.float32)
    flow=np.zeros((h,w,2),np.float32)
    mask=visible.astype(np.float32)
    for radius,stride,window in [(48,4,96),(8,2,48),(3,1,24),(2,1,16)]:
        denom=cv2.boxFilter(mask,-1,(window,window),normalize=False,borderType=cv2.BORDER_REFLECT)+1e-6
        best=np.full((h,w),np.inf,np.float32);offset=np.zeros_like(flow)
        for dy in range(-radius,radius+1,stride):
            for dx in range(-radius,radius+1,stride):
                xx=x+flow[:,:,0]+dx;yy=y+flow[:,:,1]+dy
                warp=cv2.remap(reference,xx,yy,cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
                err=np.mean(np.square(warp-observed),2)
                err+=((xx<0)|(xx>w-1)|(yy<0)|(yy>h-1)).astype(np.float32)*.01
                cost=cv2.boxFilter(err*mask,-1,(window,window),normalize=False,borderType=cv2.BORDER_REFLECT)/denom
                cost+=1e-7*(dx*dx+dy*dy)
                use=cost<best;best[use]=cost[use];offset[use]=[dx,dy]
        for k in range(2):
            offset[:,:,k]=cv2.medianBlur(offset[:,:,k],5)
        flow+=offset
    warp=cv2.remap(reference,x+flow[:,:,0],y+flow[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
    cost=cv2.boxFilter(np.mean(np.square(warp-observed),2)*mask,-1,(32,32),normalize=False)/(cv2.boxFilter(mask,-1,(32,32),normalize=False)+1e-6)
    return warp,flow,cost


def register(observed,visible,references):
    # Sanitization makes independence from hidden values an executable contract.
    observed=np.where(visible[:,:,None],observed,0).astype(np.float32)
    results=[align(observed,visible,r) for r in references]
    costs=np.stack([r[2] for r in results],2)
    chosen=np.argmin(costs,2)
    warped=np.stack([r[0] for r in results],2)
    prediction=np.take_along_axis(warped,chosen[:,:,None,None],2)[:,:,0,:]
    return prediction,results


def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=4);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    for directory in sorted(args.samples.glob('room-*-view-*'))[:args.limit]:
        m=json.loads((directory/'sample.json').read_text());h,w=m['height'],m['width']
        target=np.fromfile(directory/'target.f32',dtype='<f4').reshape(h,w,3)
        n=m.get('reference_count',len(m.get('reference_views',[])))
        references=[]
        for i in range(n):
            candidates=[directory/f'reference-{i}.f32',directory/f'reference-{i+1}.f32']
            references.append(np.fromfile(candidates[0] if candidates[0].exists() else candidates[1],dtype='<f4').reshape(h,w,3))
        visible=np.zeros((h//16,w//16),bool);visible.flat[m['visible_patch_ids']]=True
        visible=np.repeat(np.repeat(visible,16,0),16,1)
        prediction,results=register(target,visible,references)
        out=args.output/directory.name;out.mkdir()
        target.tofile(out/'target.f32');prediction.tofile(out/'prediction.f32')
        for i,r in enumerate(references):r.tofile(out/f'reference-{i}.f32')
        mse=float(np.mean(np.square(prediction[~visible]-target[~visible])))
        meta=dict(m,reference_count=n,hidden_rgb_mse=mse,method='visible-only local translation registration diagnostic')
        (out/'sample.json').write_text(json.dumps(meta,indent=2)+'\n')
        print(directory.name,inspect(out)[0],flush=True)
    print('seconds',time.monotonic()-start)
if __name__=='__main__':main()
