#!/usr/bin/env python3
"""A fixed no-learning/no-reference baseline using only observed target patches.

Inverse squared distance weights interpolate observed per-channel patch means
at pixel centers. This deliberately smooth baseline diagnoses how little RGB
MSE alone establishes. It never uses hidden target statistics or geometry.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from reconstruction_diagnostics import inspect_sample


def predict(target, visible_patch_ids, patch=16):
    h,w,_=target.shape
    ids=np.asarray(visible_patch_ids,dtype=int)
    patches=target.reshape(h//patch,patch,w//patch,patch,3).transpose(0,2,1,3,4).reshape(-1,patch*patch,3)
    observed_means=patches[ids].mean(axis=1)
    centers=np.stack([ids//(w//patch)+.5,ids%(w//patch)+.5],axis=1)
    yy,xx=np.meshgrid((np.arange(h)+.5)/patch,(np.arange(w)+.5)/patch,indexing='ij')
    positions=np.stack([yy,xx],axis=-1).reshape(-1,2)
    distances=((positions[:,None,:]-centers[None,:,:])**2).sum(axis=-1)
    weights=1/(distances+1e-6)
    weights/=weights.sum(axis=1,keepdims=True)
    return (weights@observed_means).reshape(h,w,3).astype(np.float32)


def evaluate(directory):
    a,model=inspect_sample(directory)
    pred=predict(a['target'],a['meta']['visible_patch_ids'],a['meta']['patch_size'])
    mask=a['hidden']
    mse=float(np.mean((pred[mask]-a['target'][mask])**2))
    changed=a['target'].copy();changed[mask]=7.0
    # A hidden-pixel intervention must not change any baseline prediction.
    np.testing.assert_array_equal(pred,predict(changed,a['meta']['visible_patch_ids'],a['meta']['patch_size']))
    return dict(room_seed=model['room_seed'],target_view=a['meta']['target_view'],
                baseline_hidden_rgb_mse=mse,model_hidden_rgb_mse=model['hidden_rgb_mse'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads(args.evaluation.read_text())
    rows=[evaluate(p) for p in report['samples']]
    result=dict(kind='fixed_visible_only_smooth_baseline',scope='exported target views only, not all evaluated targets',
                formula='inverse squared distance between pixel centers and observed patch centers, stabilizer 1e-6, weighted observed per-channel patch means',
                hidden_pixel_intervention_passed=True,rooms=rows,
                baseline_hidden_rgb_mse=float(np.mean([r['baseline_hidden_rgb_mse'] for r in rows])),
                model_hidden_rgb_mse=float(np.mean([r['model_hidden_rgb_mse'] for r in rows])))
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rooms'},indent=2))


if __name__=='__main__':main()
