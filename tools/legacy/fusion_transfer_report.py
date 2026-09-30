#!/usr/bin/env python3
"""Build a reproducible PDF for the fusion-transfer intervention and holdout."""
import argparse
import hashlib
import json
from pathlib import Path
import textwrap
import tomllib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from hpatches_score import dense_map,truth_map

BLUE='#166b9d';ORANGE='#d27824';GREEN='#247c55';RED='#ad3c38'
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})


def read(path):return json.loads(Path(path).read_text())
def lines(path):return [json.loads(x) for x in Path(path).read_text().splitlines()]


class Report:
    def __init__(self,pdf):self.pdf=pdf;self.number=0
    def page(self,title,subtitle=''):
        self.number+=1;fig=plt.figure(figsize=(12,8.5));fig.suptitle(title,x=.065,y=.95,ha='left',fontsize=19,fontweight='bold',color='#203444')
        fig.text(.065,.91,subtitle,fontsize=9,color='#526271');fig.text(.065,.025,'burn_gekko | RGB-only latent prediction | 2026-09-29',fontsize=8,color='#526271')
        fig.text(.94,.025,str(self.number),ha='right',fontsize=8);return fig
    def save(self,fig):self.pdf.savefig(fig);plt.close(fig)
    def prose(self,fig,items,y=.83):
        for text in items:
            wrapped=textwrap.fill(text,115)
            fig.text(.07,y,wrapped,va='top',linespacing=1.45,fontsize=11)
            y-=.034*(wrapped.count('\n')+1)+.035
    def table(self,fig,headers,rows,rect=(.065,.15,.88,.69),size=10):
        ax=fig.add_axes(rect);ax.axis('off');t=ax.table(cellText=rows,colLabels=headers,loc='center',cellLoc='left',colLoc='left')
        t.auto_set_font_size(False);t.set_fontsize(size);t.scale(1,1.65)
        for (r,c),cell in t.get_celld().items():
            cell.set_edgecolor('#dce4e9')
            if r==0:cell.set_facecolor('#203444');cell.get_text().set_color('white');cell.get_text().set_weight('bold')
            elif r%2==0:cell.set_facecolor('#f0f4f7')


def rgb(view):return np.fromfile(view['file'],dtype='<f4').reshape(256,256,3)


def annotated_pair(ax,target,reference,xy,gt,pred,title):
    ax.imshow(np.concatenate([target,reference],axis=1));ax.set_xlim(0,512);ax.set_ylim(256,0);ax.axis('off');ax.set_title(title,fontsize=10,loc='left')
    ax.axvline(255.5,color='white',linewidth=.7,alpha=.6)
    for i,(q,g,p) in enumerate(zip(xy,gt,pred)):
        error=np.linalg.norm(p-g);color=GREEN if error<=3 else ORANGE if error<=15 else RED
        ax.plot([q[0],p[0]+256],[q[1],p[1]],color=color,lw=.65,alpha=.7)
        ax.scatter(q[0],q[1],s=14,c='#17d4ed',edgecolors='black',linewidths=.3)
        ax.scatter(g[0]+256,g[1],s=32,marker='x',c='#59ff80',linewidths=1)
        ax.scatter(p[0]+256,p[1],s=24,facecolors='none',edgecolors=color,linewidths=1.2)
        ax.plot([p[0]+256,g[0]+256],[p[1],g[1]],color=color,lw=.6,linestyle=':',alpha=.8)
        ax.text(q[0]+2,q[1]-2,str(i+1),fontsize=6,color='white',bbox=dict(facecolor='black',alpha=.45,pad=.2,edgecolor='none'))
        ax.text(g[0]+258,g[1]-2,str(i+1),fontsize=6,color='#59ff80',bbox=dict(facecolor='black',alpha=.45,pad=.2,edgecolor='none'),clip_on=True)


def latent_samples(report,root,parent,candidate,receipt):
    """Show fixed-order examples in a common teacher basis, with error audits."""
    root=Path(root)
    folders=sorted((root/candidate).glob('room-*-view-*'))
    if not folders:return
    targets=[];residuals=[]
    rows={name:{(x['room_seed'],x['target_view']):x for x in read(root/name/'metrics.json')['rows']} for name in [parent,candidate]}
    for folder in folders:
        meta=read(folder/'metadata.json');shape=meta['latent_shape'];hidden=meta['hidden_tokens']
        target=np.fromfile(folder/'target-latent.f32',dtype='<f4').reshape(shape);targets.append(target)
        for name in [parent,candidate]:
            other=root/name/folder.name
            assert read(other/'metadata.json')==meta, 'sample information sets differ'
            assert np.array_equal(np.fromfile(other/'target-latent.f32',dtype='<f4').reshape(shape),target), 'teacher drift'
            pred=np.fromfile(other/'cross-latent.f32',dtype='<f4').reshape(shape)
            mse=np.mean((target[hidden].astype(np.float64)-pred[hidden])**2)
            residuals.append(abs(mse-rows[name][meta['room_seed'],meta['target_view']]['cross_mse']))
    assert max(residuals)<1e-5, 'saved latent arrays disagree with metrics'
    all_target=np.concatenate(targets);center=all_target.mean(0)
    _,_,vt=np.linalg.svd(all_target-center,full_matrices=False);basis=vt[:3].T
    lo,hi=np.quantile((all_target-center)@basis,[.01,.99],axis=0)
    receipt['latent_sample_max_mse_residual']=max(residuals)
    for index in sorted({0,len(folders)//2,len(folders)-1}):
        folder=folders[index];meta=read(folder/'metadata.json');grid=meta['grid'];shape=meta['latent_shape'];rgbshape=meta['rgb_shape']
        array=lambda path:np.fromfile(path,dtype='<f4')
        t=array(folder/'target-latent.f32').reshape(shape)
        a=array(root/parent/folder.name/'cross-latent.f32').reshape(shape)
        b=array(folder/'cross-latent.f32').reshape(shape)
        color=lambda x:np.clip((((x-center)@basis)-lo)/np.maximum(hi-lo,1e-8),0,1).reshape(*grid,3)
        hidden=np.zeros(grid[0]*grid[1],bool);hidden[meta['hidden_tokens']]=True;hidden=hidden.reshape(grid)
        target=array(folder/'target-rgb.f32').reshape(rgbshape);masked=target.copy()
        masked[np.repeat(np.repeat(hidden,16,0),16,1)]=.15
        errors=[np.mean((x-t)**2,axis=1).reshape(grid) for x in [a,b]]
        vmax=max(float(np.quantile(np.concatenate([e[hidden] for e in errors]),.98)),1e-6)
        fig=report.page(f"Fresh latent example: room {meta['room_seed']}, view {meta['target_view']}",'Fixed first / middle / last exported example; shared teacher-fitted PCA and error scales. Latent maps are not RGB reconstruction.')
        fig.set_size_inches(11.7,12.8);axes=fig.subplots(4,3);fig.subplots_adjust(left=.07,right=.95,bottom=.08,top=.87,hspace=.32,wspace=.25)
        panels=[(masked,'Observed target: 90% hidden'),(array(folder/'reference-1-rgb.f32').reshape(rgbshape),'Reference 1'),(array(folder/'reference-2-rgb.f32').reshape(rgbshape),'Reference 2'),
                (color(t),'Teacher latent PCA'),(color(a),f'{parent}: predicted latent PCA'),(color(b),f'{candidate}: predicted latent PCA')]
        for ax,(value,title) in zip(axes.flat,panels):ax.imshow(value,interpolation='nearest');ax.set_title(title,fontsize=10)
        for ax,value,title in zip(axes[2],[*errors,array(folder/'gain.f32').reshape(grid)],[f'{parent}: hidden latent MSE',f'{candidate}: hidden latent MSE','Candidate: detached error gain']):
            im=ax.imshow(np.ma.masked_where(~hidden,value),cmap='magma',vmin=0,vmax=1 if 'gain' in title else vmax,interpolation='nearest');ax.set_title(title,fontsize=10);fig.colorbar(im,ax=ax,fraction=.045)
        axes[3,0].imshow(target);axes[3,0].set_title('Full target: RI input / evaluation',fontsize=10)
        for ax,filename,title in zip(axes[3,1:],['visibility-fraction.f32','ri.f32'],['Renderer visible fraction: labels','Learned RI: full-target branch']):
            value=array(folder/filename).reshape(grid);im=ax.imshow(np.ma.masked_where(~hidden | (value<0),value),cmap='viridis',vmin=0,vmax=1,interpolation='nearest');ax.set_title(title,fontsize=10);fig.colorbar(im,ax=ax,fraction=.045)
        for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
        key=meta['room_seed'],meta['target_view']
        fig.text(.07,.05,f"Hidden MSE: {parent} {rows[parent][key]['cross_mse']:.5f}; {candidate} {rows[candidate][key]['cross_mse']:.5f}. PCA colors use shared 1-99% teacher limits; geometry never enters inference.",fontsize=9)
        report.save(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args();c=tomllib.loads(a.config.read_text())
    output=Path(c['output']);assert output.resolve().is_relative_to(Path('.data').resolve()) and not output.exists();output.parent.mkdir(parents=True,exist_ok=True)
    draft=output.with_suffix('.partial.pdf');assert not draft.exists()
    output.with_suffix('.toml').write_text(a.config.read_text())
    audit=read(c['audit_score'])['models']['continued']['summary'];hp=read(c['hpatches_score']);candidate=c['candidate'];parent=c.get('parent','continued');attention=c.get('attention_method','reciprocal_log_probability');encoder_readout=c.get('encoder_method','centered_student');decoder_readout=c.get('decoder_method','fused_decoder')
    assert hp['evaluation_use']=='development'
    plain_encoder_readout=c.get('plain_encoder_method','centered_student')
    runs={name:Path(path) for name,path in c['runs'].items()};eth=read(c['eth3d_score']) if c.get('eth3d_score') else None
    metadata=dict(Title='burn_gekko: resolving fusion transfer',Author='burn_gekko experiment pipeline',Subject='Controlled positional and semantic-guidance experiments; explicit claim boundaries')
    receipt=dict(config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),candidate=candidate,inputs={key:hashlib.sha256(Path(c[key]).read_bytes()).hexdigest() for key in ['audit_score','hpatches_score']})
    receipt['budget_snapshot']=read(c['budget_ledger'])
    hp_config=Path(c['hpatches_export'])/'config.toml'
    exported={model['name']:model for model in tomllib.loads(hp_config.read_text())['models']}
    for name,path in runs.items():
        if name in hp['models']:
            assert Path(exported[name]['checkpoint']).resolve()==(path/'final').resolve(), 'report checkpoint path mismatch'
            assert exported[name]['model_sha256']==read(path/'final/metadata.json')['model_sha256'], 'report checkpoint hash mismatch'
    receipt['inputs']['hpatches_export_config']=hashlib.sha256(hp_config.read_bytes()).hexdigest()
    receipt['runs']={name:{file:hashlib.sha256((path/file).read_bytes()).hexdigest() for file in ['config.toml','metrics.jsonl','probes.json','final/metadata.json']} for name,path in runs.items()}
    receipt['report_sources']={file:hashlib.sha256((Path(__file__).parent/file).read_bytes()).hexdigest() for file in ['fusion_transfer_report.py','hpatches_score.py','eth3d_score.py','latent_report.py']}
    receipt['extra_evidence']={name:dict(path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest()) for name,path in c.get('extra_evidence',{}).items()}
    historical={}
    receipt['historical_scores']={}
    for label,item in c.get('historical_scores',{}).items():
        previous=read(item['score']);assert previous['evaluation_use']=='development'
        directory=Path(item['export']);cfg=directory/'config.toml'
        definitions={m['name']:m for m in tomllib.loads(cfg.read_text())['models']}
        receipt['historical_scores'][label]=dict(score_sha256=hashlib.sha256(Path(item['score']).read_bytes()).hexdigest(),export_config_sha256=hashlib.sha256(cfg.read_bytes()).hexdigest())
        for name,path in runs.items():
            if name not in previous['models']:continue
            model=definitions[name];assert Path(model['checkpoint']).resolve()==(path/'final').resolve()
            assert model['model_sha256']==read(path/'final/metadata.json')['model_sha256']
            assert hashlib.sha256((directory/f'{name}.jsonl').read_bytes()).hexdigest()==previous['models'][name]['prediction_sha256']
            historical[name]=previous['models'][name]
    if eth:
        receipt['inputs']['eth3d_score']=hashlib.sha256(Path(c['eth3d_score']).read_bytes()).hexdigest()
        declared=dict(candidate=candidate,parent=parent,encoder_method=encoder_readout,plain_encoder_method=plain_encoder_readout,decoder_method=decoder_readout,attention_method=attention)
        assert eth['selected_comparisons']==declared, 'report differs from sealed holdout comparisons'
        receipt['selection_sha256']=eth['selection_sha256']
    with PdfPages(draft,metadata=metadata) as pdf:
        r=Report(pdf);fig=r.page('Fusion transfer: measured progress and remaining limits','Controlled study on the existing 12-hour Pilot07 budget. No noncommercial model initialization or teacher.')
        summary=hp['models'][candidate]['summary'];base=hp['models'][parent]['summary']
        r.prose(fig,[
            c['conclusion'],
            f"HPatches viewpoint development AEPE: decoder {base['fused_decoder']['viewpoint']['aepe']:.2f} to {summary['fused_decoder']['viewpoint']['aepe']:.2f}; {attention} {base[attention]['viewpoint']['aepe']:.2f} to {summary[attention]['viewpoint']['aepe']:.2f}. These are development results after explicit reuse of HPatches for model and readout decisions.",
            'The original model had image-grid 2D RoPE, but no camera intrinsics, extrinsics or rays. Frozen-checkpoint tests rejected a readout-only repair. The controlled training screen separates cross-view RoPE from a dense semantic guidance objective.',
            'Masked latent prediction remains primary. Dense auxiliary branches use a frozen, audited V-JEPA student ancestor. They do not supply hidden target features to sparse completion and do not use renderer geometry or external correspondence labels.',
            ('Independent qualification uses all 3,365 ETH3D interval pairs, with the candidate and readouts sealed before inference. ' if eth else 'Independent qualification is not included in this artifact. ')+
            'The local hard patch readout differs from published refinement pipelines. State of the art and sharp RGB completion are not established by this study.'
        ]);r.save(fig)

        fig=r.page('Data, initialization and evaluation boundaries','Immutable disk captures and audited model ancestry; no online scene generation during training.')
        cfg=tomllib.loads((next(iter(runs.values()))/'config.toml').read_text());data_path=Path(cfg['dataset']);data=read(data_path/'manifest.json');dc=data['config']
        receipt['inputs']['training_manifest']=hashlib.sha256((data_path/'manifest.json').read_bytes()).hexdigest()
        paragraphs=[
            f"Training cache: {dc['split_scenes'][0]:,} training rooms, {dc['split_scenes'][1]} validation rooms, {dc['cameras']} views per room at {dc['width']} x {dc['height']}. Generator: {data['generator']}. Earlier screens select subsets; the descriptor and refinement phases use the complete training pool.",
            f"Data seed {dc['seed']}, density {dc['density']}, camera-baseline policy control {dc['camera_baseline']}. The baseline control is dimensionless, not a distance in metres. Camera FOV and pose are captured for verification and evaluation, but neither enters these training objectives or RGB-only matching exports.",
            'Student and primary teacher begin from the audited MIT V-JEPA 2.1 Base package. Fusion and task heads were initialized randomly in the own latent lineage. Auxiliary affinities use a frozen own student ancestor. Warm starts and exact optimizer resumes are distinguished in saved provenance.',
            'All 116 HPatches sequences are now development data, including the 59-sequence viewpoint subset used for choices in this study. The earlier synthetic test has also been observed. Bootstrap intervals on development data do not turn those selections into independent confirmation.'
        ]
        if c.get('lineage_record'):
            lineage=read(c['lineage_record'])
            assert Path(lineage['checkpoint']).resolve()==(runs[candidate]/'final').resolve()
            assert lineage['nontraining_optimizer_inputs']==0 and lineage['noncommercial_weight_dependencies']==[]
            receipt['inputs']['lineage_record']=hashlib.sha256(Path(c['lineage_record']).read_bytes()).hexdigest()
            paragraphs.append(f"Selected lineage audit: {lineage['executed_updates']:,} optimizer updates, {lineage['target_exposures']:,} target exposures, {lineage['unique_training_rooms']:,} distinct training rooms. Exact-resume prefixes count once. No validation/test optimizer inputs or noncommercial weight dependencies were found.")
        if c.get('latent_assessment'):
            assessment=tomllib.loads((Path(c['latent_assessment'])/'config.toml').read_text());fresh_path=Path(assessment['dataset']);fresh=read(fresh_path/'manifest.json')
            assert assessment['split']=='test' and assessment['rooms']==128 and assessment['references']==2 and assessment['mask_ratio']==.9
            assert fresh['config']['seed']==2610010000 and fresh['config']['cameras']==4
            receipt['inputs']['qualification_manifest']=hashlib.sha256((fresh_path/'manifest.json').read_bytes()).hexdigest()
            paragraphs.append(f"Fresh qualification: {assessment['rooms']} test rooms, {fresh['config']['cameras']} captured views, seed {fresh['config']['seed']}, camera-baseline policy {fresh['config']['camera_baseline']}. This capture and ETH3D are evaluated only after candidate/readout selection is frozen. Seed disjointness is checked against every ancestor's training and validation rooms.")
        r.prose(fig,paragraphs);r.save(fig)

        fig=r.page('Architecture and the camera question','Image position is present; a shared calibrated camera frame is absent.')
        r.prose(fig,[
            'Per-view V-JEPA encoder -> sparse target plus a set of reference features -> six shared decoder blocks -> latent prediction. Self-attention uses image-grid RoPE. Cross-view RoPE can now be controlled independently; the monocular branch retains its spatial encoding.',
            'Training-time auxiliary path: full target/reference pair -> the same fusion trunk -> dense latent preservation plus layer-wise affinity KL. A follow-up adds symmetric decoder-descriptor KL. Targets come from centered cosine similarity in a frozen ancestor encoder: semantic soft labels, not true geometric correspondences.',
            'Camera prediction could be useful as a separately declared variant: supervise relative pose with an explicit gauge and scale convention, test focal prediction against a constant prior, and evaluate using predicted cameras. Intermediate losses need a measured gradient schedule and transfer ablation.',
            'DPPE studies camera-dependent Q/K and value-frame transforms. Its particular rotation/translation coupling failure is not implemented in this decoder. ZipSplat uses optional camera conditioning and selective geometric-loss detachment for Gaussian reconstruction; it is not a direct camera-head prescription.',
            'A metadata audit of 32 training rooms found vertical FOVs from 28.2 to 106.2 degrees. The synthetic cache has focal variation, but a low supervised camera loss on it would not by itself demonstrate real-camera transfer.'
        ]);r.save(fig)

        if c.get('matched_input_preflight'):
            gate=read(c['matched_input_preflight']);rejected=read(c['rejected_input_preflight'])
            assert gate['numerical_initialization_gate']
            for key in ['matched_input_preflight','rejected_input_preflight']:
                receipt['inputs'][key]=hashlib.sha256(Path(c[key]).read_bytes()).hexdigest()
            fig=r.page('Supplying spatial features: a matched-layout experiment','An initialization check failed before quality training; the replacement control changes the experiment, not its tolerance.')
            rows=[]
            for label,values in [('Original narrow / wide control',rejected),('Matched wide / wide control',gate)]:
                d=values['initial_maximum_per_view_mse_delta']
                rows.append([label,f"{d['cross_mse']:.7g}",f"{d['monocular_mse']:.7g}",str(values['numerical_initialization_gate'])])
            r.table(fig,['Native preflight','Maximum cross-MSE drift','Maximum mono-MSE drift','Gate passed'],rows,rect=(.065,.66,.88,.15),size=9)
            r.prose(fig,[
                'Block-6 affinity targets teach the desired similarity structure, but the original trunk only receives final-layer features. The new route concatenates trained block-6 and final tokens before the shared fusion projection; sparse target features still see only observed patches.',
                'Both replacement arms use a 1536-to-384 projection, preserve its original rows, and initialize the added 294,912 weights to zero. Only the extra feature scale differs: zero for the control, one for the treatment. This matches parameter count, capture work and GEMM shape.',
                'The registered initial per-view MSE bound remains 1e-5. The rejected run is retained. A mathematically zero extension is not by itself evidence of identical native CUDA arithmetic.',
                'The primary target remains final-layer V-JEPA latents. Separate middle-layer encoder baselines, hidden-input isolation, fixed-teacher checks, and matched training examples prevent attributing input leakage or a changed baseline to fusion learning.'
            ],y=.60);r.save(fig)

        if c.get('preservation_decision'):
            decision=read(c['preservation_decision']);parity=read(c['preservation_native_parity'])
            assert decision['selected']==candidate and parity['passed']
            for key in ['preservation_decision','preservation_native_parity']:
                receipt['inputs'][key]=hashlib.sha256(Path(c[key]).read_bytes()).hexdigest()
            fig=r.page('Final development screen: descriptor preservation','One remaining matched experiment after the 6000-step continuation; independent evaluation follows the declared choice.')
            rows=[]
            for name,label in [('route-balanced-continue','6000-step parent'),('preserve-control','Weight 1 control'),('preserve-strong','Weight 4 treatment')]:
                value=hp['models'][name]['summary'];validation=read(runs[name]/'validation/metrics.json')
                rows.append([label,*[f"{value[m]['viewpoint']['aepe']:.3f}" for m in ['fused_decoder','conditional_decoder',attention]],f"{validation['mean_cross_mse']:.5f}"])
            r.table(fig,['Checkpoint','Raw decoder AEPE','Conditional decoder','Attention AEPE','Latent MSE'],rows,rect=(.065,.61,.88,.2),size=9)
            failed={name:[key for key,value in record['gates'].items() if not value] for name,record in decision['comparisons'].items()}
            labels={'route-balanced-continue_to_preserve-control':'control versus parent','route-balanced-continue_to_preserve-strong':'weight 4 versus parent','preserve-control_to_preserve-strong':'weight 4 versus control'}
            rejected='; '.join(labels.get(name,name) for name,keys in failed.items() if keys) or 'none'
            r.prose(fig,[
                'Both arms use 1500 updates, identical own weights, fresh optimizers, rates, samples and masks, with the encoder frozen. Only the descriptor KL coefficient changes; primary latent targets stay fixed.',
                f"Selected: {candidate}. Gates require all three AEPE means to improve, PCK3 loss <=1% relative, latent-error growth <=5%, and RI AUROC loss <=0.02. Failed comparisons: {rejected}.",
                'v18 rejects weight 4 before training. v20 changes only its validation range among Rust sources. A 64-update weight-1 replay matches every component loss exactly, resuming from step 52 after a preparation timeout. Failures remain charged to the study.'
            ],y=.55);r.save(fig)

        fig=r.page('Frozen checkpoint: the simple repairs did not work','Same model weights, complete 59-sequence viewpoint subset; lower AEPE is better.')
        methods=['centered_student','student_encoder','fixed_teacher','trace_centered_decoder','fused_decoder','no_cross_rope_centered_decoder','reciprocal_attention','trace_logits_mean','trace_centered_content_mean']
        r.table(fig,['Readout / intervention','Viewpoint AEPE','Illumination AEPE'],[[m,f"{audit[m]['viewpoint']['aepe']:.3f}",f"{audit[m]['illumination']['aepe']:.3f}"] for m in methods],rect=(.065,.27,.88,.52),size=10)
        r.prose(fig,['Removing cross-view RoPE at inference changes a learned input distribution. Its failure does not settle the training-time question. The next experiment retrains both positional settings under a matched objective and control.'],y=.20);r.save(fig)

        rows=[]
        for name,path in runs.items():
            metrics=lines(path/'metrics.jsonl');probes=read(path/'probes.json');result=hp['models'].get(name,historical.get(name))
            scores=result['summary'] if result else None
            rows.append([name,str(metrics[-1]['step']),f"{probes[-1]['cross_mse']:.4f}",f"{scores['fused_decoder']['viewpoint']['aepe']:.2f}" if scores else '-',f"{scores['centered_decoder']['viewpoint']['aepe']:.2f}" if scores else '-',f"{scores[attention]['viewpoint']['aepe']:.2f}" if scores else '-'])
        for start in range(0,len(rows),8):
            fig=r.page('Training arms and development transfer','Controls within each phase share examples and masks; starting weights, cohorts and update counts differ across phases.')
            r.table(fig,['Run','Schedule step','Latent MSE','Decoder AEPE','Centered decoder','Attention AEPE'],rows[start:start+8],rect=(.065,.4,.88,.4),size=9)
            r.prose(fig,['The initial guidance treatment uses attention/dense weights 0.1; descriptor alignment adds weight 0.1. A later matched strength screen tests descriptor weight 1.0 while holding the other objectives fixed. Temperature stays 0.07. Primary latent error must remain within 5% of its matched control.',
                'Historical phases use their checksummed completed exports; the current candidate comparison uses the common final export. Differences across phases are descriptive. The auxiliary teacher remains fixed across exact resume, and encoder unfreezing follows its recorded gate.'],y=.31);r.save(fig)

        rows=[]
        for name,path in runs.items():
            cfg=tomllib.loads((path/'config.toml').read_text());m=lines(path/'metrics.jsonl')
            counts=[sum(x['stage']==stage for x in m) for stage in range(3)]
            seen=len({sample[0] for x in m for sample in x['samples']})
            rows.append([name,str(cfg['train_rooms']),str(seen),str(cfg['validation_rooms']),str(cfg['batch_size']*len(m)),'/'.join(map(str,counts))])
        for start in range(0,len(rows),8):
            fig=r.page('Training phases and information sets','Phase boundaries restart optimizers unless the run provenance explicitly records exact resume.')
            r.table(fig,['Run','Room pool','Rooms sampled','Val rooms','Target exposures','Frozen / last 2 / full'],rows[start:start+8],rect=(.065,.4,.88,.4),size=9)
            r.prose(fig,['The first factorial screen uses 1,024 training rooms and 16 validation rooms. The descriptor screen expands to 8,192 training rooms and 64 validation rooms. Raw validation levels across these phases are not a matched comparison.',
                'Phase counts report actual updates with a frozen encoder, the last two blocks trainable, and the full encoder trainable. Teacher and ancestor affinity encoders remain frozen in every stage.'],y=.31);r.save(fig)

        curve_groups=c.get('convergence_groups',{'all phases':list(runs)})
        for group_name,names in curve_groups.items():
            assert names and set(names)<=runs.keys(), 'unknown convergence run'
            fig=r.page('Convergence: '+group_name.replace('_',' '),'25-step training means; inactive auxiliary losses omitted. Compare matched cohorts and preserve phase/resume boundaries.')
            axes=fig.subplots(2,3);fig.subplots_adjust(left=.075,right=.96,bottom=.12,top=.84,hspace=.45,wspace=.34)
            for name in names:
                path=runs[name];metrics=lines(path/'metrics.jsonl');probes=read(path/'probes.json')
                auxiliary=tomllib.loads((path/'config.toml').read_text()).get('fusion_auxiliary',{})
                color=plt.get_cmap('tab20')(list(runs).index(name)%20)
                axes[0,0].plot([p['step'] for p in probes],[p['cross_mse'] for p in probes],marker='.',label=name,color=color)
                for ax,key in [(axes[0,1],'cross'),(axes[0,2],'gradient_norm'),(axes[1,0],'attention_kl'),(axes[1,1],'dense_latent_mse'),(axes[1,2],'descriptor_kl')]:
                    weight_key={'attention_kl':'attention_weight','dense_latent_mse':'dense_weight','descriptor_kl':'descriptor_weight'}.get(key)
                    if weight_key and auxiliary.get(weight_key,0)<=0:continue
                    values=[m.get(key,0.) for m in metrics];groups=[values[i:i+25] for i in range(0,len(values),25)]
                    ax.plot([metrics[min((i+1)*25,len(values))-1]['step'] for i in range(len(groups))],[np.mean(g) for g in groups],label=name,color=color)
            for ax,title in zip(axes.flat,['Validation masked latent MSE','Training masked latent MSE','Unclipped gradient norm','Attention guidance KL','Dense auxiliary latent MSE','Decoder descriptor KL']):ax.set_title(title,fontsize=10);ax.set_xlabel('Schedule update');ax.grid(alpha=.2)
            axes[0,0].legend(fontsize=8);r.save(fig)

        fig=r.page('HPatches development: all declared standard readouts','59 viewpoint sequences / 295 pairs. Per-sequence bootstrap intervals are descriptive after development reuse.')
        wanted=['fixed_teacher','centered_teacher','conditional_teacher','conditional_centered_teacher','student_encoder','centered_student','conditional_student','conditional_centered_student','fused_decoder','centered_decoder','conditional_decoder','conditional_centered_decoder','fused_latent','centered_latent','reciprocal_attention','reciprocal_logits','reciprocal_log_probability','same_position']
        rows=[]
        for m in wanted:
            if m not in summary:continue
            rows.append([m,f"{base.get(m,summary[m])['viewpoint']['aepe']:.3f}",f"{summary[m]['viewpoint']['aepe']:.3f}",f"{100*summary[m]['viewpoint']['pck3']:.2f}%",f"{summary[m]['illumination']['aepe']:.3f}"])
        r.table(fig,['Method','Parent AEPE','Candidate AEPE','Candidate PCK3','Illumination AEPE'],rows,size=9);r.save(fig)

        spatial=[m for m in summary if m.startswith(('student_l06_','teacher_l06_'))]
        if spatial:
            fig=r.page('Preselected spatial encoder controls','Block 6 was selected on development images before any independent outcomes. All four fixed readouts remain visible.')
            r.table(fig,['Readout','Parent viewpoint AEPE','Candidate AEPE','Candidate PCK3'],[[m,f"{base[m]['viewpoint']['aepe']:.3f}",f"{summary[m]['viewpoint']['aepe']:.3f}",f"{100*summary[m]['viewpoint']['pck3']:.2f}%"] for m in sorted(spatial)],rect=(.065,.4,.88,.4),size=9)
            r.prose(fig,['The earlier final-layer encoder was a weaker baseline. Discovering a stronger intermediate layer changes the evidence required for a fusion claim; it does not itself demonstrate improved fusion learning.',
                f"The primary encoder control in subsequent annotated comparisons is {encoder_readout}. Main masked and dense latent targets remain final-layer even when the affinity teacher uses block 6."],y=.31);r.save(fig)

        if 'conditional_centered_student' in summary:
            fig=r.page('Fair readout controls','Feature scores and attention scores receive the same reciprocal conditional normalization.')
            r.prose(fig,[
                'Cosine feature baselines alone are insufficient for attributing a reciprocal-attention gain to learning. Reciprocal log probabilities include row and column normalization, which can improve matches even with unchanged features.',
                'Additional controls apply that identical operator to raw and centered teacher, student-encoder and decoder cosine scores. Their temperature is fixed at 0.07, matching the registered affinity target. It is not fitted to held-out geometry. All original plain cosine and attention readouts remain reported.',
                f"Candidate viewpoint AEPE: plain centered encoder {summary['centered_student']['viewpoint']['aepe']:.3f}; conditional centered encoder {summary['conditional_centered_student']['viewpoint']['aepe']:.3f}; conditional decoder {summary['conditional_decoder']['viewpoint']['aepe']:.3f}; learned attention {summary[attention]['viewpoint']['aepe']:.3f}.",
                'These controls were added before the refinement endpoints were scored and before any independent model evaluation. The training loss, examples and model parameters are unaffected by this evaluation extension.'
            ]);r.save(fig)

        if c.get('encoder_audit_score'):
            probe=read(c['encoder_audit_score'])
            assert probe['evaluation_use']=='development'
            receipt['inputs']['encoder_audit_score']=hashlib.sha256(Path(c['encoder_audit_score']).read_bytes()).hexdigest()
            assert len(probe['models'])==1
            probe_name,probe_model=next(iter(probe['models'].items()))
            fig=r.page('Encoder feature levels: a separate diagnostic','Trained V-JEPA hierarchy norms; frozen weights and no fitting. Uniform mean combines cosine matrices with equal weights.')
            rows=[]
            for backbone in ['teacher','student']:
                for level in ['l03','l06','l09','l12','mean']:
                    methods=probe_model['summary'];prefix=f'{backbone}_{level}'
                    values=[methods[f'{prefix}_{m}']['viewpoint'] for m in ['raw','centered','centered_conditional']]
                    rows.append([backbone,level.removeprefix('l'),*[f"{v['aepe']:.3f}" for v in values],f"{100*values[-1]['pck3']:.2f}%"])
            r.table(fig,['Encoder','Level','Raw AEPE','Centered AEPE','Conditional centered','Conditional PCK3'],rows,rect=(.065,.35,.88,.46),size=9)
            r.prose(fig,[f"Student probe checkpoint: {probe_name}. Teacher is the unchanged audited MIT V-JEPA package. Final-level features are checked against the corresponding standard readouts; this audit does not alter the decoder or active training weights.",
                'Block 6 improves both viewpoint error and precise matches over the final layer, exposing a target-design limitation. This diagnostic is not a learned fusion improvement. The subsequent matched experiment changes only the affinity level; the layer choice is frozen before independent qualification.'],y=.28);r.save(fig)

        if c.get('transfer_diagnostics'):
            diagnosis=read(c['transfer_diagnostics']);receipt['inputs']['transfer_diagnostics']=hashlib.sha256(Path(c['transfer_diagnostics']).read_bytes()).hexdigest()
            assert diagnosis['score_sha256']==receipt['inputs']['hpatches_score'], 'stale transfer diagnostics'
            fig=r.page('Development transfer by actual image displacement','Ground-truth displacement bins in HP-240 coordinates. These diagnostic strata pool pixels; primary scores average sequences.')
            axes=fig.subplots(1,2);fig.subplots_adjust(left=.08,right=.96,bottom=.25,top=.82,wspace=.24)
            bins=['0_to_8_px','8_to_32_px','32_px_or_more']
            for ax,method in zip(axes,['fused_decoder',attention]):
                matched_encoder=plain_encoder_readout if method=='fused_decoder' else encoder_readout
                for name,readout,color,style in [(parent,matched_encoder,'#777777','--'),(candidate,matched_encoder,'#222222','--'),(parent,method,ORANGE,'-'),(candidate,method,BLUE,'-')]:
                    values=diagnosis['models'][name][readout]['strata'];ax.plot(range(3),[values[b]['aepe'] for b in bins],style,color=color,marker='o',label=f'{name}: {readout}')
                ax.set_title(method,fontsize=11);ax.set_xticks(range(3),['<8 px','8 to <32 px','>=32 px']);ax.set_xlabel('True query displacement');ax.set_ylabel('Pixel-pooled AEPE');ax.grid(alpha=.2);ax.legend(fontsize=7)
            old_bias=diagnosis['models'][parent][attention]['exact_same_patch_fraction'];new_bias=diagnosis['models'][candidate][attention]['exact_same_patch_fraction']
            r.prose(fig,[f"Exact same-patch attention matches: {100*old_bias:.2f}% for the parent, {100*new_bias:.2f}% for the candidate. A reduction can help moving views while hurting nearly aligned pairs; it is not itself an accuracy metric."],y=.18)
            if 'geometry_only_grid_control' in diagnosis:
                control=diagnosis['geometry_only_grid_control']
                fig.text(.07,.075,f"Label-only nearest-grid diagnostic: {control['aepe']:.2f} px AEPE. Uses ground truth; unavailable to inference and not a rigorous lower bound.",fontsize=9)
            r.save(fig)

        if c.get('preservation_diagnostics'):
            diagnosis=read(c['preservation_diagnostics'])
            assert diagnosis['score_sha256']==receipt['inputs']['hpatches_score'], 'stale feature preservation diagnostics'
            receipt['inputs']['preservation_diagnostics']=hashlib.sha256(Path(c['preservation_diagnostics']).read_bytes()).hexdigest()
            fig=r.page('Which encoder matches does fusion change?','Fixed patch centres, HP-240 coordinates; pair-mean fractions. These are separate diagnostics from dense interpolated AEPE.')
            rows=[]
            for name in [parent,candidate]:
                for method,value in diagnosis['models'][name].items():
                    s=value['summary']
                    rows.append([name,method,f"{100*s['changed_fraction']:.1f}%",f"{100*s['corrected_fraction']:.1f}%",f"{100*s['damaged_fraction']:.1f}%",f"{100*s['candidate_pck15']:.1f}%"])
            r.table(fig,['Model','Readout','Changed','Corrected','Damaged','PCK15'],rows,rect=(.065,.36,.88,.45),size=9)
            r.prose(fig,['A match is correct here when its error is at most 15 pixels, one metric-grid cell. Corrected means the encoder was wrong and fusion becomes correct; damaged means the reverse. Both fractions use all valid patch queries as the denominator, with pairs weighted equally.',
                'Lower mean distance can coexist with fewer precise matches when large errors shrink but some correct matches are damaged. Both error and inlier metrics are therefore retained. These development diagnostics do not supply supervision or filter model inputs.'],y=.29);r.save(fig)

        if c.get('latent_assessment'):
            root=Path(c['latent_assessment']);old=read(root/parent/'metrics.json');new=read(root/candidate/'metrics.json')
            receipt['inputs']['latent_parent']=hashlib.sha256((root/parent/'metrics.json').read_bytes()).hexdigest()
            receipt['inputs']['latent_candidate']=hashlib.sha256((root/candidate/'metrics.json').read_bytes()).hexdigest()
            fig=r.page('Independent latent prediction with wider camera separation','128 fresh rooms, four captured views per room. Primary comparison uses two references and the same fixed 90% target mask.')
            fields=[('Masked cross-view MSE',lambda x:x['mean_cross_mse']),('Masked monocular MSE',lambda x:x['mean_monocular_mse']),('Training-only position-mean MSE',lambda x:x['mean_train_position_mean_mse']),('Relative cross-view gain',lambda x:1-x['mean_cross_mse']/x['mean_monocular_mse']),('Shuffled-reference-token MSE',lambda x:x['mean_spatially_shuffled_mse']),('Unrelated-reference MSE',lambda x:x['mean_unrelated_mse']),('Spatial variance / teacher',lambda x:x['mean_spatial_variance_ratio']),('Learned RI co-visibility AUROC',lambda x:x['learned_ri_covisibility']['auroc']),('Learned RI co-visibility AP',lambda x:x['learned_ri_covisibility']['average_precision']),('Positive prevalence / chance AP',lambda x:x['learned_ri_covisibility']['positives']/x['learned_ri_covisibility']['pixels'])]
            r.table(fig,['Metric','Parent','Candidate'],[[label,f'{fn(old):.5f}',f'{fn(new):.5f}'] for label,fn in fields],rect=(.065,.30,.88,.5),size=10)
            first={(x['room_seed'],x['target_view']):x for x in old['rows']};second={(x['room_seed'],x['target_view']):x for x in new['rows']};assert first.keys()==second.keys()
            scenes=sorted({k[0] for k in first});delta=np.array([np.mean([first[k]['cross_mse']-second[k]['cross_mse'] for k in first if k[0]==scene]) for scene in scenes]);rng=np.random.default_rng(719);samples=delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
            r.prose(fig,[f"Parent minus candidate latent MSE: {delta.mean():.5f}, room-bootstrap 95% interval [{np.quantile(samples,.025):.5f}, {np.quantile(samples,.975):.5f}]. This interval measures scene variation, not training-seed uncertainty.",
                'RI is an error-improvement ranking score from a separate full-target branch, not a calibrated visibility probability. Full-target student features never enter sparse completion. Geometry labels are evaluation-only; latent metrics do not establish sharp RGB reconstruction.'],y=.23);r.save(fig)

            from latent_report import ci_room
            receipt['latent_reference_contrasts']={}
            fig=r.page('Independent reference-use controls','Paired room bootstrap, 10,000 resamples; each room keeps its four target views together.')
            rows=[]
            for name,metrics in [(parent,old),(candidate,new)]:
                receipt['latent_reference_contrasts'][name]={}
                for label,field in [('Monocular - related','monocular_mse'),('Unrelated - related','unrelated_mse'),('Shuffled - ordered','spatially_shuffled_mse'),('Training mean - related','train_position_mean_mse')]:
                    value=ci_room(metrics['rows'],field,'cross_mse')
                    assert value and value['rooms']==128
                    receipt['latent_reference_contrasts'][name][field]=value
                    rows.append([name,label,f"{value['mean']:.5f}",f"[{value['low']:.5f}, {value['high']:.5f}]"])
            r.table(fig,['Checkpoint','Latent MSE contrast','Mean difference','95% room interval'],rows,rect=(.065,.4,.88,.4),size=9)
            r.prose(fig,['Positive differences favor prediction with the related, correctly ordered reference set. The same fixed mask and target images are used in each paired contrast.',
                'Token shuffling preserves encoded features, including positional information already inside them. Sensitivity shows use of reference layout, but does not by itself prove 3D geometry or calibrated visibility.'],y=.31);r.save(fig)

            if (root/candidate/'correspondence.json').exists():
                old_match=read(root/parent/'correspondence.json');new_match=read(root/candidate/'correspondence.json')
                fig=r.page('Fresh rooms: correspondence across all view pairs','All 12 directed pairs per four-view room. Visible patch-center EPE in 256-pixel coordinates; geometric labels only filter scoring.')
                rows=[]
                for method in [*wanted,'grid_oracle']:
                    if method not in new_match['summary']:continue
                    previous=old_match['summary'][method];current=new_match['summary'][method]
                    rows.append([method,f"{previous['mean_epe']:.3f}",f"{current['mean_epe']:.3f}",f"{100*current['pck16']:.2f}%",f"{100*current['mutual_coverage']:.2f}%" if method!='grid_oracle' and current['mutual_coverage'] is not None else '-'])
                r.table(fig,['Method','Parent EPE','Candidate EPE','Candidate PCK16','Mutual coverage'],rows,rect=(.065,.20,.88,.63),size=9)
                r.prose(fig,['The grid oracle chooses the nearest available patch center using ground truth; it is a resolution floor, not a model. Mutual coverage is the fraction of geometrically valid query points retained by reciprocal matching.'],y=.14);r.save(fig)

            if c.get('reference_assessments'):
                fig=r.page('Reference count: one checkpoint, different information sets','First 32 fresh rooms; 128 target views per count. No fitting or checkpoint changes between counts.')
                rows=[]
                for key,path in sorted(c['reference_assessments'].items()):
                    for name in [parent,candidate]:
                        x=read(Path(path)/name/'metrics.json');rows.append([key,name,f"{x['mean_cross_mse']:.5f}",f"{100*(1-x['mean_cross_mse']/x['mean_monocular_mse']):.2f}%",f"{x['learned_ri_covisibility']['auroc']:.3f}"])
                r.table(fig,['References','Checkpoint','Masked MSE','Relative gain','RI AUROC'],rows,size=10);r.save(fig)

            latent_samples(r,root,parent,candidate,receipt)

            fig=r.page('Input isolation and numerical behavior','Native audit on the first fresh target; these are measured differences, not architecture assumptions.')
            fields=[('Change hidden target RGB','hidden_rgb_max_abs_delta'),('Reverse references: monocular max','monocular_reference_permutation_max_abs_delta'),('Reverse references: cross-view max','reference_permutation_max_abs_delta'),('Reverse references: cross-view RMS','reference_permutation_rms_delta')]
            r.table(fig,['Intervention / measured output difference','Parent','Candidate'],[[label,f"{old['input_audit'][key]:.6g}",f"{new['input_audit'][key]:.6g}"] for label,key in fields],rect=(.065,.45,.88,.33),size=10)
            receipt['fresh_input_audit']={name:metrics['input_audit'] for name,metrics in [(parent,old),(candidate,new)]}
            order_max=max(metrics['input_audit']['reference_permutation_max_abs_delta'] for metrics in [old,new])
            order_result=f"The maximum reference-order difference is {order_max:.6g}; the 1e-5 float32 permutation bound {'passes' if order_max<=1e-5 else 'does not pass'} on these audited targets."
            r.prose(fig,['Changing hidden RGB must not change sparse predictions; the monocular branch must not depend on references. The table reports the actual saved audit, and the CPU tests separately exercise these information contracts.',
                order_result+' Float64 stable attention remains unsupported by the current Burn CUDA Fusion evaluation path.'],y=.37);r.save(fig)

        if eth:
            fig=r.page('Independent ETH3D: full interval correspondence benchmark','Original image pixel coordinates; all 10 scenes and seven intervals. No holdout-based readout selection.')
            e=eth['models'][candidate]['methods'];b=eth['models'][parent]['methods'];rows=[]
            for method in wanted:
                if method not in e:continue
                rows.append([method,f"{b[method]['summary']['aepe']:.3f}",f"{e[method]['summary']['aepe']:.3f}",f"{100*e[method]['summary']['pck3']:.2f}%",f"{100*e[method]['summary']['point_weighted_pck3']:.2f}%"])
            r.table(fig,['Readout','Parent AEPE','Candidate AEPE','PCK3 / image','PCK3 / points'],rows,size=9);r.save(fig)
            if spatial:
                fig=r.page('Independent ETH3D: preselected block-6 controls','Same fixed spatial feature level and readout operators as the development registration; no holdout sweep.')
                rows=[[method,f"{b[method]['summary']['aepe']:.3f}",f"{e[method]['summary']['aepe']:.3f}",f"{100*e[method]['summary']['pck3']:.2f}%",f"{100*e[method]['summary']['point_weighted_pck3']:.2f}%"] for method in sorted(spatial)]
                r.table(fig,['Readout','Parent AEPE','Candidate AEPE','PCK3 / image','PCK3 / points'],rows,rect=(.065,.39,.88,.43),size=9)
                r.prose(fig,[f'Primary encoder controls: raw decoder versus {plain_encoder_readout}; conditional decoder and attention versus {encoder_readout}. All standard final-layer controls remain on the preceding page.',
                    'A fusion claim must survive comparison with the stronger preselected encoder level. Its selection used development data only; these real-scene outcomes were unavailable at selection time.'],y=.3);r.save(fig)
            contrasts=eth.get('paired_model_contrasts',{}).get(f'{parent}_minus_{candidate}',{})
            if contrasts:
                fig=r.page('Independent transfer: paired uncertainty','10,000 bootstrap samples of the 10 scenes; all intervals of a scene stay together. Positive differences favor the readout.')
                rows=[]
                for method in ['fused_decoder','conditional_decoder',attention]:
                    from eth3d_score import paired_scene_ci
                    matched_encoder=plain_encoder_readout if method=='fused_decoder' else encoder_readout
                    encoder_contrast=paired_scene_ci(e[matched_encoder]['scenes'],e[method]['scenes'])
                    for label,value in [('Parent minus candidate',contrasts[method]),('Candidate encoder minus readout',encoder_contrast)]:
                        rows.append([label,method,f"{value['mean']:.3f}",f"[{value['low']:.3f}, {value['high']:.3f}]"])
                for method in ['fused_decoder','conditional_decoder',attention]:
                    matched_encoder=plain_encoder_readout if method=='fused_decoder' else encoder_readout
                    value=paired_scene_ci(e[method]['scenes'],e[matched_encoder]['scenes'],metric='pck3')
                    rows.append(['Readout - encoder PCK3 (pp)',method,f"{100*value['mean']:.3f}",f"[{100*value['low']:.3f}, {100*value['high']:.3f}]"])
                r.table(fig,['Contrast','Readout','Mean difference','95% scene interval'],rows,rect=(.065,.36,.88,.45),size=8)
                r.prose(fig,[f'Raw decoder uses {plain_encoder_readout}; conditional decoder and attention use {encoder_readout}. AEPE differences are pixels; PCK3 differences are percentage points. Positive values favor the candidate/readout.',
                    'Intervals describe the ten benchmark scenes, not training-seed uncertainty or foundation-pretraining disjointness. The model and primary readouts were selected before seeing these outcomes.'],y=.25);r.save(fig)
            fig=r.page('Independent ETH3D: behavior as view separation grows','Intervals are temporal sampling gaps. Metrics average pairs within scene and scenes within interval.')
            ax=fig.add_axes([.09,.2,.84,.63]);intervals=[3,5,7,9,11,13,15]
            for model,method,style in [(candidate,encoder_readout,'--'),(parent,decoder_readout,'--'),(candidate,decoder_readout,'-'),(candidate,attention,'-')]:
                val=eth['models'][model]['methods'][method]['intervals'];ax.plot(intervals,[val[str(i)]['aepe'] for i in intervals],style,marker='o',label=f'{model}: {method}')
            ax.set_xlabel('Temporal interval');ax.set_ylabel('AEPE in original pixels');ax.legend(fontsize=9);ax.grid(alpha=.2);r.save(fig)

        # Cases are selected by manifest order, never by candidate error.
        manifest=read(c['hpatches_images']);seqs=sorted([s for s in manifest['sequences'] if s['name'].startswith('v_')],key=lambda s:s['name']);geom=np.load(Path(c['hpatches_images']).parent/'homographies.npz')
        exports={name:{(v['sequence'],v['target'],v['method']):v for v in lines(Path(c['hpatches_export'])/f'{name}.jsonl')} for name in [parent,candidate]}
        for index in [0,len(seqs)//2,len(seqs)-1]:
            seq=seqs[index];views=seq['views'];gt,valid=truth_map(geom[f"{seq['name']}_6"],views[0]['original_hw'],views[5]['original_hw'],256)
            y,x=np.mgrid[8:256:32,8:256:32];xy=np.stack([x.ravel(),y.ravel()],1);xy=xy[valid[xy[:,1],xy[:,0]]];xy=xy[np.linspace(0,len(xy)-1,min(9,len(xy)),dtype=int)]
            fig=r.page(f"Annotated development case: {seq['name']}, image 6 -> 1",'Fixed first / middle / last viewpoint sequence, last indexed view. Cyan query; green cross = truth; circle = prediction.')
            axes=fig.subplots(2,2);fig.subplots_adjust(left=.055,right=.97,bottom=.13,top=.84,hspace=.26,wspace=.12)
            for ax,(name,method) in zip(axes.flat,[(candidate,encoder_readout),(parent,decoder_readout),(candidate,decoder_readout),(candidate,attention)]):
                prediction=dense_map(exports[name][seq['name'],6,method]['indices'],size=256)
                annotated_pair(ax,rgb(views[5]),rgb(views[0]),xy,gt[xy[:,1],xy[:,0]],prediction[xy[:,1],xy[:,0]],f'{name}: {method}')
            fig.text(.07,.07,'Line colors show displayed-point error at 256 pixels: green <=3; orange <=15; red >15. These sample points are not the aggregate metric.',fontsize=8);r.save(fig)

        if eth:
            from eth3d_score import sparse_flow,sparse_truth
            manifest=read(c['eth3d_images']);pairs={p['id']:p for p in manifest['pairs']};geometry=np.load(Path(c['eth3d_images']).parent/'correspondences.npz')
            exports={name:{(v['pair'],v['method']):v for v in lines(Path(c['eth3d_export'])/f'{name}.jsonl')} for name in [parent,candidate]}
            for key in ['delivery_area-15-0000','forest-15-0000','storage_room-15-0000']:
                pair=pairs[key];t=manifest['images'][pair['target']];ref=manifest['images'][pair['reference']];hw=t['original_hw'];xy,flow=sparse_truth(geometry[key],hw)
                ids=np.linspace(0,len(xy)-1,min(9,len(xy)),dtype=int);xy=xy[ids];flow=flow[ids];scale=np.array([256/hw[1],256/hw[0]])
                fig=r.page(f'Annotated independent case: {key}','First indexed pair at interval 15 in three predeclared scenes. Display scaled to 256; aggregate errors use original pixels.')
                axes=fig.subplots(2,2);fig.subplots_adjust(left=.055,right=.97,bottom=.13,top=.84,hspace=.26,wspace=.12)
                for ax,(name,method) in zip(axes.flat,[(candidate,encoder_readout),(parent,decoder_readout),(candidate,decoder_readout),(candidate,attention)]):
                    pred=sparse_flow(exports[name][key,method]['indices'],xy,hw)
                    annotated_pair(ax,rgb(t),rgb(ref),xy*scale,(xy+flow)*scale,(xy+pred)*scale,f'{name}: {method}')
                fig.text(.07,.07,'Query locations are evenly spaced in the official rasterized point list. Green crosses are evaluation labels, unavailable to model inference.',fontsize=8);r.save(fig)

        if c.get('efficiency_comparison'):
            efficiency=read(c['efficiency_comparison']);energy=read(c['efficiency_energy'])['commands'];trace=read(c['efficiency_trace'])
            assert efficiency['numerical_gate_passed']
            for key in ['efficiency_comparison','efficiency_energy','efficiency_trace']:
                receipt['inputs'][key]=hashlib.sha256(Path(c[key]).read_bytes()).hexdigest()
            fig=r.page('GPU efficiency: a measured transfer bottleneck','Matched 64-update execution check; same own weights, fresh optimizers, frozen encoder, samples, masks and losses.')
            rows=[['Warmed median update (s)',*[f'{x:.4f}' for x in efficiency['median_update_seconds']]],
                  ['Complete command (s)',*[f"{energy[n]['command_seconds']:.2f}" for n in ['before','after']]],
                  ['Observed board energy (Wh)',*[f"{energy[n]['observed_board_wh']:.3f}" for n in ['before','after']]],
                  ['Gross board joules / trained target',*[f"{energy[n]['gross_observed_board_joules_per_target']:.2f}" for n in ['before','after']]]]
            r.table(fig,['Measurement','Original NHWC upload','Contiguous upload'],rows,rect=(.065,.55,.88,.25),size=10)
            ratio=energy['after']['observed_board_joules']/energy['before']['observed_board_joules']
            r.prose(fig,[f"Nsight measured {100*trace['kernel_time_fraction']:.1f}% kernel-time coverage and {trace['rgb_transfer_seconds_per_update']:.3f} s/update in RGB transfers. High GPU activity concealed this bottleneck; remaining dispatch gaps are not claimed solved.",
                'Rank-4 NHWC caused 12-byte pitched rows. Flat upload preserves normalized RGB bitwise and cuts batch-16 upload plus normalization from 61.1 to 7.19 ms.',
                f"Matched frozen-encoder throughput improves {efficiency['warmed_update_speedup']:.2f}x and complete-command board energy falls {100*(1-ratio):.2f}%. All 64 updates have identical component losses and validation MSE. Other training stages are monitored separately.",
                'Board energy includes startup, evaluation and desktop activity. Samples near 1 Hz have full supported coverage with explicit gap accounting. This is not whole-system electricity.'],y=.50);r.save(fig)

        if c.get('shared_device_replay'):
            shared=read(c['shared_device_replay'])
            receipt['inputs']['shared_device_replay']=hashlib.sha256(Path(c['shared_device_replay']).read_bytes()).hexdigest()
            fig=r.page('Shared GPU: interpreting the later slowdown','A replay with the previous sealed binary separates a code regression from changing execution conditions.')
            timing=shared['matched_first16_warmed_step_seconds']
            r.prose(fig,[
                f"The same v16 binary repeats its first {shared['compared_updates']} optimizer updates with identical component losses and samples. Warmed median time changes from {timing[0]:.4f} to {timing[1]:.4f} seconds. This slowdown therefore cannot be attributed solely to the new spatial-input implementation.",
                'Read-only process monitoring records concurrent remote-desktop and desktop-shell GPU activity. Activity counters are not additive SM occupancy and do not quantify the causal share of the slowdown. No desktop application or device power policy was modified.',
                'Later training reports actual stage throughput and gross board energy under shared load. Sequential runs across different desktop conditions are not a controlled dedicated-GPU energy comparison.',
                'The earlier Nsight trace still exposes many small launches. The pinned Burn Autodiff attention implementation uses its differentiable fallback; calling the attention API does not prove fused training attention. The upload repair does not claim to eliminate all dispatch overhead.'
            ]);r.save(fig)

        if c.get('phase_energy'):
            energy=read(c['phase_energy'])['commands']
            receipt['inputs']['phase_energy']=hashlib.sha256(Path(c['phase_energy']).read_bytes()).hexdigest()
            fig=r.page('Current training: measured board energy','Inclusive command energy; startup, validation and concurrent desktop work are included.')
            rows=[[name,f"{value['command_seconds']:.1f}",f"{value['observed_board_wh']:.2f}",f"{value['gross_observed_board_joules_per_target']:.2f}"] for name,value in energy.items()]
            r.table(fig,['Command','Wall seconds','Board Wh','Board J / trained target'],rows,rect=(.065,.48,.88,.30),size=10)
            r.prose(fig,['Energy is integrated from device-wide samples with explicit coverage and gap accounting in the companion receipt. It is not process-isolated or whole-system energy.',
                'Use quality, update throughput and joules per target together. A high utilization percentage or lower instantaneous wattage alone does not show efficient learning.'],y=.37);r.save(fig)

        if c.get('current_efficiency_trace'):
            trace=read(c['current_efficiency_trace'])
            receipt['inputs']['current_efficiency_trace']=hashlib.sha256(Path(c['current_efficiency_trace']).read_bytes()).hexdigest()
            fig=r.page('Current execution: remaining dispatch cost','Bounded exact replay with the repaired upload path; all encoder blocks train, with concurrent desktop activity recorded.')
            rows=[['Measured update window',str(trace['selected_updates'])],
                  ['GPU kernel launches',f"{trace['kernel_count']:,}"],
                  ['Kernel union / window',f"{100*trace['kernel_time_fraction']:.1f}%"],
                  ['Kernel and copy union / window',f"{100*trace['device_event_time_fraction']:.1f}%"],
                  ['RGB host-to-device seconds / update',f"{trace['rgb_transfer_seconds_per_update']:.6f}"],
                  ['Unattributed gap seconds / update',f"{trace['unattributed_gap_seconds']/trace['selected_updates']:.6f}"],
                  ['Maximum replay total-loss difference',f"{trace['replay_max_total_loss_delta']:.6g}"]]
            r.table(fig,['Measurement','Observed value'],rows,rect=(.065,.4,.88,.43),size=10)
            r.prose(fig,['Kernel-time coverage is not SM occupancy or useful FLOPs. Gaps can contain host dispatch, synchronization or another process\'s GPU work. Profiling adds overhead; unprofiled stage timings and board energy are reported separately.',
                'This replay has a different model and objective from the earlier upload diagnosis. It measures the remaining execution bottlenecks, not an additional controlled speedup.'],y=.31);r.save(fig)

        ledger=receipt['budget_snapshot'];rows=[]
        for name,path in runs.items():
            m=lines(path/'metrics.jsonl');times=[x['seconds'] for x in m];batch=tomllib.loads((path/'config.toml').read_text())['batch_size']
            telemetry=c.get('telemetry',{}).get(name)
            peak=max((x.get('process_vram_mib',0) for x in lines(telemetry)),default=0)/1024 if telemetry else None
            rows.append([name,len(m),f'{np.median(times):.3f}',f'{batch/np.median(times):.2f}',f'{sum(times)/60:.1f}',f'{peak:.2f}' if peak is not None else '-'])
        for start in range(0,len(rows),8):
            fig=r.page('Measured training throughput','Single workstation; update-only timings exclude preparation, validation and checkpoint writing.')
            r.table(fig,['Run','Updates','Median s/update','Targets / s','Update minutes','Peak GPU GiB'],rows[start:start+8],rect=(.065,.37,.88,.43),size=9)
            r.prose(fig,['Timings span the actual encoder stages used by each run. Unfreezing adds encoder backward work; compare matched stages and objectives before attributing a speed difference to the runtime.',
                'The separate upload qualification uses identical examples, losses and frozen-encoder stages. Its energy comparison includes the complete command and should not be inferred from update timing alone.'],y=.28);r.save(fig)
        fig=r.page('Compute, provenance and limits','Measured native execution on the single workstation; all GPU work remains in the cumulative study ledger.')
        active=ledger.get('running',{}).get('command_elapsed_seconds',0)
        active_text=f" Running command elapsed at this snapshot: {active/3600:.3f} hours." if active else ''
        r.prose(fig,[f"Completed command wall time: {ledger['command_seconds']/3600:.3f} of {ledger['ceiling_seconds']/3600:.0f} hours.{active_text} The budget includes capture, preparation, training, evaluation, checkpoint writes and failures. CPU engineering/reporting is excluded; table timings cover updates only.",
            'One seed in the factorial screen is not a multi-seed qualification. HPatches was used for development. ETH3D, if present, was evaluated only after candidate and readouts were frozen. Dataset overlap with foundation-model pretraining is not independently certified.',
            'No claim of sharp RGB reconstruction follows from latent prediction or correspondence metrics. SOTA requires matched external baselines, protocol parity, multiple seeds and broader real-view qualification.']);r.save(fig)
        fig=r.page('Reproduction and primary sources','Machine-readable artifacts accompany this PDF; no external messaging or publication was performed.')
        r.prose(fig,[
            'Experiment protocol: configs/archive/pilot-07/pilot07-fusion-protocol.toml. All user configs are TOML; generated metrics and provenance are JSON. Training and evaluation source archives, binary hashes, test logs and GPU telemetry are stored in .data/pilot-07/fusion-audit/.',
            'Positional audit: src/model.rs and src/fusion_audit.rs. Dense guidance: src/fusion_objective.rs and src/latent_pilot.rs. External RGB exports: src/hpatches.rs and src/eth3d.rs. Independent CPU scorers: tools/legacy/hpatches_score.py and tools/legacy/eth3d_score.py.',
            'DPPE: https://arxiv.org/html/2606.31585v1 . ZipSplat: https://arxiv.org/html/2606.05102v1 . These inform the camera discussion; their weights and code are not dependencies.',
            'ZeroCo official evaluation: https://github.com/cvlab-kaist/ZeroCo . ETH3D data and documentation: https://www.eth3d.net/documentation . Source URLs and archive hashes are pinned in configs/archive/pilot-07/pilot07-eth3d-prepare.toml.',
            'This PDF was generated from the named experiment artifacts, without replacing earlier reports. Its JSON sidecar records input hashes and the final page count.'
        ]);r.save(fig);receipt['pages']=r.number
    draft.rename(output)
    receipt['pdf_sha256']=hashlib.sha256(output.read_bytes()).hexdigest();output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
