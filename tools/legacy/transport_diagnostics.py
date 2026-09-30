#!/usr/bin/env python3
"""Independent diagnostics for unenhanced, RGB-only reference transport exports."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def inspect(directory):
    m=json.loads((directory/'sample.json').read_text());h,w=m['height'],m['width']
    target=np.fromfile(directory/'target.f32',dtype='<f4').reshape(h,w,3)
    pred=np.fromfile(directory/'prediction.f32',dtype='<f4').reshape(h,w,3)
    refs=[np.fromfile(directory/f'reference-{i}.f32',dtype='<f4').reshape(h,w,3) for i in range(m['reference_count'])]
    hidden=np.ones((h//16,w//16),bool);hidden.flat[m['visible_patch_ids']]=False
    hidden=np.repeat(np.repeat(hidden,16,0),16,1)
    mse=float(np.mean((pred[hidden]-target[hidden])**2))
    np.testing.assert_allclose(mse,m['hidden_rgb_mse'],rtol=1e-4,atol=1e-7)
    gt=[];pr=[];interior_gt=[];interior_pr=[];seam_gt=[];seam_pr=[]
    for axis in [0,1]:
        valid=np.take(hidden,range(hidden.shape[axis]-1),axis=axis)&np.take(hidden,range(1,hidden.shape[axis]),axis=axis)
        delta_gt=np.diff(target,axis=axis);delta_pr=np.diff(pred,axis=axis)
        gt.append(delta_gt[valid].ravel());pr.append(delta_pr[valid].ravel())
        seam=np.broadcast_to((np.arange(hidden.shape[axis]-1)%16==15).reshape((-1,1) if axis==0 else (1,-1)),valid.shape)
        interior_gt.append(delta_gt[valid & ~seam].ravel());interior_pr.append(delta_pr[valid & ~seam].ravel())
        seam_gt.append(delta_gt[valid & seam].ravel());seam_pr.append(delta_pr[valid & seam].ravel())
    gt=np.concatenate(gt);pr=np.concatenate(pr)
    row=dict(room_seed=m['room_seed'],target_view=m.get('target_view',0),mse=mse,psnr=float(-10*np.log10(max(mse,1e-12))),edge_cosine=float(np.dot(gt,pr)/max(np.linalg.norm(gt)*np.linalg.norm(pr),1e-12)),edge_energy=float(np.mean(pr**2)/max(np.mean(gt**2),1e-12)),edge_mae=float(np.abs(gt-pr).mean()),identity_reference_mse=[float(np.mean((r[hidden]-target[hidden])**2)) for r in refs])
    ig,ip,sg,sp=map(np.concatenate,[interior_gt,interior_pr,seam_gt,seam_pr])
    row.update(interior_edge_cosine=float(np.dot(ig,ip)/max(np.linalg.norm(ig)*np.linalg.norm(ip),1e-12)),interior_edge_energy=float(np.dot(ip,ip)/max(np.dot(ig,ig),1e-12)),seam_edge_energy=float(np.dot(sp,sp)/max(np.dot(sg,sg),1e-12)),seam_edge_mae=float(np.mean(abs(sp-sg))))
    if (directory/'mixture.f32').exists():
        mixture=np.fromfile(directory/'mixture.f32',dtype='<f4').reshape(h,w,m['reference_count']+1)
        generated=np.fromfile(directory/'generated.f32',dtype='<f4').reshape(h,w,3)
        transported=np.fromfile(directory/'transported.f32',dtype='<f4').reshape(h,w,3)
        np.testing.assert_allclose(mixture.sum(axis=2),1,atol=2e-6)
        assert np.isfinite(mixture).all() and (mixture>=0).all() and (mixture<=1).all()
        reconstructed=mixture[:,:,:1]*generated+(1-mixture[:,:,:1])*transported
        np.testing.assert_allclose(reconstructed,pred,atol=2e-6,rtol=2e-5)
        flow=[np.fromfile(directory/f'flow-{i}.f32',dtype='<f4').reshape(h,w,2) for i in range(m['reference_count'])]
        row.update(generator_weight=float(mixture[:,:,0][hidden].mean()),
                   generated_mse=float(np.mean((generated[hidden]-target[hidden])**2)),
                   transported_mse=float(np.mean((transported[hidden]-target[hidden])**2)),
                   mean_flow_pixels=float(np.mean([np.linalg.norm(f[hidden],axis=1).mean() for f in flow])))
        if (directory/'coarse-flow-0.f32').exists():
            coarse_target=target.reshape(h//4,4,w//4,4,3).mean(axis=(1,3))
            terms=[]
            for index in range(m['reference_count']):
                control=np.fromfile(directory/f'coarse-flow-{index}.f32',dtype='<f4').reshape(h//4,w//4,2)
                assert np.isfinite(control).all()
                for axis in [0,1]:
                    curvature=np.diff(control,n=2,axis=axis)
                    edge=abs(np.diff(coarse_target,axis=axis)).mean(axis=2,keepdims=True)
                    weight=np.exp(-10*(np.take(edge,range(edge.shape[axis]-1),axis=axis)+np.take(edge,range(1,edge.shape[axis]),axis=axis)))
                    terms.append(float((abs(curvature)*weight).mean()))
            row['coarse_flow_curvature']=float(np.mean(terms))
    masked=target.copy();masked[hidden]=[.22,.27,.32]
    composite=pred.copy();composite[~hidden]=target[~hidden]
    return row,dict(target=target,masked=masked,prediction=pred,completion=composite,references=refs,hidden=hidden,meta=m)


def main():
    p=argparse.ArgumentParser();p.add_argument('--samples',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    directories=sorted(args.samples.glob('room-*-view-*'))
    if not directories:
        p.error(f'no sample exports in {args.samples}')
    pairs=[inspect(d) for d in directories]
    rows=[r for r,a in pairs]
    report=dict(samples=rows,means={k:float(np.mean([r[k] for r in rows])) for k in ['mse','psnr','edge_cosine','edge_energy','edge_mae']})
    args.output.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
    selected=[pairs[i] for i in sorted(set(np.linspace(0,len(pairs)-1,min(4,len(pairs)),dtype=int)))]
    fig,axes=plt.subplots(len(selected),5,figsize=(14,3*len(selected)),squeeze=False)
    for axes,(r,a) in zip(axes,selected):
        images=[a['target'],a['masked'],*a['references'][:2],a['completion']]
        method=a['meta'].get('method','')
        completion_label=('Released Gekko (ORACLE stats)' if a['meta'].get('oracle_statistics') else ('End-to-end completion' if 'end-to-end' in method else ('Hybrid completion' if 'predicted patch statistics' in method else 'Transport completion')))
        for ax,im,label in zip(axes,images,['Target','Observed input','Reference 1','Reference 2',completion_label]):
            ax.imshow(im.clip(0,1),interpolation='nearest');ax.axis('off');ax.set_title(label,fontsize=10)
        axes[-1].text(.5,-.07,f"{r['psnr']:.2f} dB; edge cosine {r['edge_cosine']:.3f}",transform=axes[-1].transAxes,ha='center',fontsize=9)
    fig.tight_layout();fig.savefig(args.output,dpi=170,bbox_inches='tight');plt.close(fig)
    print(json.dumps(report['means']))

if __name__=='__main__':main()
