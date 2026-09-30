#!/usr/bin/env python3
"""Score frozen RGB-only exports using held-out HPatches homographies.

Full 116-sequence / 580-pair HP-240 assessment of a declared local readout.
This is not claimed to reproduce ZeroCo's refinement or its published numbers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tomllib
import cv2
import numpy as np


def truth_map(h_ref_to_target, ref_hw, target_hw, size=240):
    # Match the HP-240 homography scaling convention in the official evaluator.
    scale = lambda hw: np.diag([size/hw[1],size/hw[0],1.])
    matrix = np.linalg.inv(scale(target_hw) @ h_ref_to_target @ np.linalg.inv(scale(ref_hw)))
    y,x = np.mgrid[:size,:size]
    homogeneous = np.stack([x,y,np.ones_like(x)],-1) @ matrix.T
    valid_den = np.abs(homogeneous[...,2])>1e-12
    xy = homogeneous[...,:2]/np.where(valid_den,homogeneous[...,2],1.)[...,None]
    valid = valid_den & np.isfinite(xy).all(-1) & (xy>=0).all(-1) & (xy<=size-1).all(-1)
    return xy,valid


def dense_map(indices, grid=16, size=240):
    ids=np.asarray(indices).reshape(grid,grid)
    assert ids.min()>=0 and ids.max()<grid*grid
    y,x=np.mgrid[:grid,:grid]
    flow=np.stack([ids%grid-x,ids//grid-y],-1).astype(np.float32)*(size/grid)
    flow=cv2.resize(flow,(size,size),interpolation=cv2.INTER_LINEAR)
    y,x=np.mgrid[:size,:size]
    return flow+np.stack([x,y],-1)


def paired_ci(rows, first, second, subset='all', metric='aepe'):
    """First minus second, resampling complete sequences for a declared metric."""
    assert metric in ('aepe', 'pck1', 'pck3', 'pck5')
    rows=[r for r in rows if subset=='all' or r['subset']==subset]
    lookup={(r['sequence'],r['target'],r['method']):r[metric] for r in rows}
    seq=sorted({r['sequence'] for r in rows})
    delta=np.array([np.mean([lookup[(s,t,first)]-lookup[(s,t,second)] for t in range(2,7)]) for s in seq])
    rng=np.random.default_rng(719)
    b=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
    return dict(mean=float(delta.mean()),low=float(np.quantile(b,.025)),high=float(np.quantile(b,.975)),sequences=len(seq))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    a=p.parse_args(); c=tomllib.loads(a.config.read_text())
    root,export,output=map(Path,[c['dataset'],c['export'],c['output']])
    assert output.resolve().is_relative_to(Path('.data').resolve()) and not output.exists()
    for key,filename in [('images_sha256','images.json'),('geometry_sha256','homographies.npz')]:
        assert hashlib.sha256((root/filename).read_bytes()).hexdigest()==c[key], f'{filename} checksum mismatch'
    manifest=json.loads((root/'images.json').read_text())
    geometry=np.load(root/'homographies.npz')
    by_name={s['name']:s for s in manifest['sequences']}
    evaluation_use=c.get('evaluation_use','development')
    assert evaluation_use in ('development','held_out')
    report=dict(status=f'external_{evaluation_use}',model_input_size=256,metric_image_size=240,
        protocol='Primary: 59 viewpoint sequences / 295 pairs matching the official ZeroCo CSV. Supplementary: 57 illumination sequences and all 116 combined. Hard patch matches, bilinear displacement upsampling, HP-240 homography scaling, pair-mean then sequence-mean aggregation.',
        comparability='Different model size and readout from ZeroCo; no published-number parity or SOTA claim.',
        external_data_used_for_training=c.get('external_data_used_for_training',False),
        external_data_used_for_checkpoint_selection=evaluation_use=='development',
        evaluation_use=evaluation_use,
        geometry_sha256=hashlib.sha256((root/'homographies.npz').read_bytes()).hexdigest(),models={})
    for model in c['models']:
        file=export/f'{model}.jsonl'
        rows=[]; seen=set(); gts={}
        for line in file.read_text().splitlines():
            r=json.loads(line); seq=by_name[r['sequence']]; target=r['target']
            key=(r['sequence'],target,r['method']); assert key not in seen;seen.add(key)
            assert target in range(2,7) and r['reference']==1 and r['grid']==[16,16]
            pair=key[:2]
            if pair not in gts:
                gts[pair]=truth_map(geometry[f'{pair[0]}_{target}'],seq['views'][0]['original_hw'],seq['views'][target-1]['original_hw'])
            gt,valid=gts[pair]
            error=np.linalg.norm(dense_map(r['indices'])-gt,axis=-1)[valid]
            assert len(error)>0 and np.isfinite(error).all()
            rows.append(dict(sequence=pair[0],target=target,method=r['method'],subset='illumination' if pair[0].startswith('i_') else 'viewpoint',
                aepe=float(error.mean()),pck1=float((error<=1).mean()),pck3=float((error<=3).mean()),pck5=float((error<=5).mean()),valid_pixels=int(valid.sum())))
        methods=sorted({r['method'] for r in rows})
        assert len(rows)==116*5*len(methods) and len(gts)==580
        summary={}
        for method in methods:
            summary[method]={}
            for subset in ['all','illumination','viewpoint']:
                part=[r for r in rows if r['method']==method and (subset=='all' or r['subset']==subset)]
                summary[method][subset]={k:float(np.mean([r[k] for r in part])) for k in ['aepe','pck1','pck3','pck5']}
                summary[method][subset]['pairs']=len(part)
        contrasts={m:paired_ci(rows,'fixed_teacher',m,'viewpoint') for m in methods if m!='fixed_teacher'}
        report['models'][model]=dict(summary=summary,teacher_minus_candidate_aepe=contrasts,rows=rows,
            contrast_subset='viewpoint; 59 sequences, 295 pairs',
            centered_teacher_minus_centered_student=paired_ci(rows,'centered_teacher','centered_student','viewpoint'),
            prediction_sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({m:v['summary'] for m,v in report['models'].items()},indent=2))


if __name__=='__main__':
    main()
