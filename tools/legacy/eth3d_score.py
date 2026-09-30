#!/usr/bin/env python3
"""Score sealed RGB-only predictions on all official ETH3D interval pairs.

Original-resolution sparse flow; local hard patch matches, not ZeroCo refinement.
AEPE and per-image PCK average pairs, then scenes, then intervals. Point-weighted
PCK is also reported per interval, matching the official aggregation distinction.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import tomllib
import numpy as np


def sparse_truth(points,hw):
    """Official labels place float displacements at rounded integer locations.

    Repeated locations use the last point, as in the published array assignment.
    """
    h,w=hw
    xy=np.rint(points[:,2:4]).astype(np.int64)
    assert (xy>=0).all() and (xy[:,0]<w).all() and (xy[:,1]<h).all()
    key=xy[:,1]*w+xy[:,0]
    _,reverse=np.unique(key[::-1],return_index=True)
    ids=len(key)-1-reverse
    return xy[ids],(points[:,:2]-points[:,2:4])[ids]


def sparse_flow(indices,xy,hw,grid=16):
    """Bilinear resize with OpenCV half-pixel coordinates, sampled sparsely."""
    h,w=hw;ids=np.asarray(indices).reshape(grid,grid)
    assert ids.min()>=0 and ids.max()<grid*grid
    yy,xx=np.mgrid[:grid,:grid]
    flow=np.stack([(ids%grid-xx)*(w/grid),(ids//grid-yy)*(h/grid)],-1).astype(np.float32)
    uv=(xy.astype(np.float32)+.5)*np.array([grid/w,grid/h],np.float32)-.5
    uv=np.clip(uv,0,grid-1);lower=np.floor(uv).astype(np.int64);upper=np.minimum(lower+1,grid-1)
    blend=uv-lower
    x,y=lower.T;xx,yy=upper.T;wx,wy=blend.T
    return ((1-wx)*(1-wy))[:,None]*flow[y,x]+(wx*(1-wy))[:,None]*flow[y,xx]+((1-wx)*wy)[:,None]*flow[yy,x]+(wx*wy)[:,None]*flow[yy,xx]


def aggregate(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['interval'],r['scene']].append(r)
    assert len(groups)==70
    per_scene={}
    for (interval,scene),part in groups.items():
        per_scene[f'{interval}/{scene}']={k:float(np.mean([r[k] for r in part])) for k in ['aepe','pck1','pck3','pck5']}
    intervals={}
    for interval in [3,5,7,9,11,13,15]:
        scenes=[v for k,v in per_scene.items() if k.startswith(f'{interval}/')];assert len(scenes)==10
        intervals[str(interval)]={k:float(np.mean([r[k] for r in scenes])) for k in scenes[0]}
        part=[r for r in rows if r['interval']==interval]
        n=sum(r['points'] for r in part)
        intervals[str(interval)].update({f'point_weighted_pck{t}':sum(r[f'correct{t}'] for r in part)/n for t in [1,3,5]})
    summary={k:float(np.mean([r[k] for r in intervals.values()])) for k in intervals['3']}
    return dict(summary=summary,intervals=intervals,scenes=per_scene)


def paired_scene_ci(first,second,metric='aepe'):
    """First minus second, keeping all intervals of each scene together."""
    assert metric in ('aepe','pck1','pck3','pck5')
    scenes=sorted({k.split('/',1)[1] for k in first})
    delta=np.array([np.mean([first[f'{r}/{s}'][metric]-second[f'{r}/{s}'][metric] for r in [3,5,7,9,11,13,15]]) for s in scenes])
    rng=np.random.default_rng(719);sample=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
    return dict(mean=float(delta.mean()),low=float(np.quantile(sample,.025)),high=float(np.quantile(sample,.975)),scenes=len(scenes),unit='scene bootstrap, not training seeds')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();c=tomllib.loads(a.config.read_text());root=Path(c['dataset']);out=Path(c['output'])
    assert out.resolve().is_relative_to(Path('.data').resolve()) and not out.exists()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert sha(root/'images.json')==c['images_sha256']
    assert sha(root/'correspondences.npz')==c['geometry_sha256']
    assert sha(c['selection_record'])==c['selection_sha256']
    selection=tomllib.loads(Path(c['selection_record']).read_text())
    export_config=tomllib.loads((Path(c['export'])/'config.toml').read_text())
    selected={m['name']:m for m in selection['models']}
    exported={m['name']:m for m in export_config['models']}
    assert selected.keys()==exported.keys()==set(c['models']), 'export differs from sealed model selection'
    assert export_config['selection_sha256']==c['selection_sha256']
    assert export_config.get('spatial_layer')==selection.get('spatial_layer'), 'spatial encoder control differs from sealed selection'
    assert export_config['images_sha256']==c['images_sha256'], 'export used a different image manifest'
    for name,model in exported.items():
        assert Path(model['checkpoint']).resolve()==Path(selected[name]['checkpoint']).resolve()
        assert model['model_sha256']==selected[name]['model_sha256']
        provenance=json.loads((Path(c['export'])/f'{name}-provenance.json').read_text())
        assert provenance['checkpoint']['model_sha256']==model['model_sha256']
        assert provenance['selection_sha256']==c['selection_sha256']
    manifest=json.loads((root/'images.json').read_text());pairs={r['id']:r for r in manifest['pairs']}
    geometry=np.load(root/'correspondences.npz');truth={}
    for key,pair in pairs.items():
        hw=manifest['images'][pair['target']]['original_hw']
        assert hw==manifest['images'][pair['reference']]['original_hw'], 'displacement readout requires equal pair dimensions'
        xy,flow=sparse_truth(geometry[key],hw);truth[key]=(hw,xy,flow)
    report=dict(status='independent_holdout',protocol=__doc__,pairs=3365,scenes=10,intervals=[3,5,7,9,11,13,15],model_input_size=256,
        metric_coordinates='original image pixels',comparability='All official pairs and official aggregation, but local hard patch readout; no published-number parity or SOTA claim',
        selection_sha256=c['selection_sha256'],images_sha256=c['images_sha256'],geometry_sha256=c['geometry_sha256'],
        export_config_sha256=sha(Path(c['export'])/'config.toml'),models={})
    report['selected_comparisons']={key:selection[key] for key in ['candidate','parent','encoder_method','plain_encoder_method','decoder_method','attention_method']}
    assert selection['candidate'] in selected and selection['parent'] in selected
    assert all(selection[key] in selection['readouts'] for key in ['encoder_method','plain_encoder_method','decoder_method','attention_method'])
    for model in c['models']:
        file=Path(c['export'])/f'{model}.jsonl';rows=[];seen=set()
        for line in file.read_text().splitlines():
            r=json.loads(line);pair=pairs[r['pair']];key=(r['pair'],r['method']);assert key not in seen;seen.add(key)
            assert r['scene']==pair['scene'] and r['interval']==pair['interval'] and r['grid']==[16,16]
            hw,xy,gt=truth[r['pair']];error=np.linalg.norm(sparse_flow(r['indices'],xy,hw)-gt,axis=1);assert np.isfinite(error).all()
            row=dict(pair=r['pair'],scene=r['scene'],interval=r['interval'],method=r['method'],aepe=float(error.mean()),points=len(error))
            for t in [1,3,5]:row[f'correct{t}']=int((error<=t).sum());row[f'pck{t}']=float((error<=t).mean())
            rows.append(row)
        methods=sorted({r['method'] for r in rows});assert len(rows)==3365*len(methods)
        assert set(methods)==set(selection['readouts']), 'readouts differ from sealed selection'
        scored={method:aggregate([r for r in rows if r['method']==method]) for method in methods}
        report['models'][model]=dict(methods=scored,rows=rows,prediction_sha256=sha(file),teacher_minus_candidate={method:paired_scene_ci(scored['fixed_teacher']['scenes'],scored[method]['scenes']) for method in methods})
    # All declared methods are retained. These paired contrasts measure variation
    # across held-out scenes, not variation across independently trained seeds.
    report['paired_model_contrasts']={}
    for parent in c['models'][:1]:
        base=report['models'][parent]['methods']
        for candidate in c['models'][1:]:
            result=report['models'][candidate]['methods']
            assert base.keys()==result.keys(), 'models used different readout families'
            report['paired_model_contrasts'][f'{parent}_minus_{candidate}']={method:paired_scene_ci(base[method]['scenes'],result[method]['scenes']) for method in base}
    for result in report['models'].values():
        methods=result['methods']
        result['centered_student_minus_readout']={method:paired_scene_ci(methods['centered_student']['scenes'],value['scenes']) for method,value in methods.items()}
        if 'conditional_centered_student' in methods:
            result['conditional_centered_student_minus_readout']={method:paired_scene_ci(methods['conditional_centered_student']['scenes'],value['scenes']) for method,value in methods.items()}
            result['readout_minus_conditional_centered_student_pck3']={method:paired_scene_ci(value['scenes'],methods['conditional_centered_student']['scenes'],metric='pck3') for method,value in methods.items()}
        result['selected_encoder_minus_readout']={}
        result['readout_minus_selected_encoder_pck3']={}
        for method in ['fused_decoder',selection['decoder_method'],selection['attention_method']]:
            baseline=selection['plain_encoder_method'] if method=='fused_decoder' else selection['encoder_method']
            result['selected_encoder_minus_readout'][method]=dict(baseline=baseline,**paired_scene_ci(methods[baseline]['scenes'],methods[method]['scenes']))
            result['readout_minus_selected_encoder_pck3'][method]=dict(baseline=baseline,**paired_scene_ci(methods[method]['scenes'],methods[baseline]['scenes'],metric='pck3'))
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({model:{k:v['summary'] for k,v in result['methods'].items()} for model,result in report['models'].items()},indent=2))


if __name__=='__main__':main()
