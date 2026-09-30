#!/usr/bin/env python3
"""Build a reproducible comparison PDF from completed, common-protocol artifacts."""
import argparse
import json
import hashlib
import textwrap
from pathlib import Path
import tomllib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from latent_report import ci_room
from hpatches_score import truth_map


def read(p): return json.loads(Path(p).read_text())


def text_page(pdf,title,paragraphs):
    def new():
        fig=plt.figure(figsize=(11.7,8.3))
        fig.text(.06,.915,title,size=18,weight='bold')
        fig.text(.06,.035,'burn_gekko | Controlled latent continuation | Research prototype',size=9,color='#555')
        return fig
    fig=new();y=.87
    for p in paragraphs:
        lines=textwrap.wrap(p,115,break_long_words=True)
        if y-.027*len(lines)<.10:
            pdf.savefig(fig);plt.close(fig);fig=new();y=.87
        fig.text(.06,y,'\n'.join(lines),va='top',size=11,linespacing=1.4)
        y-=.027*len(lines)+.035
    pdf.savefig(fig);plt.close(fig)


def external_sample_pages(pdf, dataset, export, selected):
    """Fixed first/median viewpoint sequences; never select examples by error."""
    data=read(Path(dataset)/'images.json')
    sequences=sorted([s for s in data['sequences'] if s['name'].startswith('v_')],key=lambda s:s['name'])
    chosen=[sequences[0],sequences[len(sequences)//2]]
    geometry=np.load(Path(dataset)/'homographies.npz')
    rows=[json.loads(line) for line in (Path(export)/f'{selected}.jsonl').read_text().splitlines()]
    for sequence in chosen:
        name=sequence['name']; target=sequence['views'][5];reference=sequence['views'][0]
        images=[np.fromfile(v['file'],dtype='<f4').reshape(256,256,3) for v in [target,reference]]
        gt,valid=truth_map(geometry[f'{name}_6'],reference['original_hw'],target['original_hw'],size=256)
        xy=np.array([[(i%16)*16+7.5,(i//16)*16+7.5] for i in range(256)])
        # Interpolate truth at the descriptor center, matching the flow view.
        points=(xy-.5).astype(int)
        eligible=np.array([i for i,(x,y) in enumerate(points) if valid[y:y+2,x:x+2].all()])
        ids=eligible[np.linspace(0,len(eligible)-1,min(12,len(eligible))).astype(int)]
        fig,axes=plt.subplots(3,1,figsize=(11.7,8.3))
        fig.suptitle(f'External correspondence | {name} | view 6 to 1',fontsize=17,y=.95)
        for ax,method in zip(axes,['fixed_teacher','student_encoder','reciprocal_attention']):
            row=next(r for r in rows if r['sequence']==name and r['target']==6 and r['method']==method)
            ax.imshow(np.concatenate(images,axis=1));ax.axis('off');ax.set_title(method,fontsize=10)
            for i in ids:
                q=xy[i];p=xy[row['indices'][int(i)]];x,y=points[i];truth=gt[y:y+2,x:x+2].mean((0,1))
                color='#35cc55' if np.linalg.norm(p-truth)<=16 else '#e34040'
                p=p+[256,0];truth=truth+[256,0]
                ax.plot([q[0],p[0]],[q[1],p[1]],color=color,lw=.8,alpha=.8)
                ax.scatter(*q,s=9,c=color);ax.scatter(*p,s=12,c=color,marker='x');ax.scatter(*truth,s=10,c='#28c9ef')
        fig.text(.06,.04,'Fixed first/median viewpoint sequences, view 6. Green: <=16px error at input scale; red: larger; cyan: homography truth.',size=9)
        fig.tight_layout(rect=(.02,.07,.98,.94));pdf.savefig(fig);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    a=p.parse_args();c=tomllib.loads(a.config.read_text())
    root,out=Path(c['assessment']),Path(c['pdf']);out.parent.mkdir(parents=True,exist_ok=True)
    assert out.resolve().is_relative_to(Path('.data').resolve())
    models=c['models']; labels=[m['name'] for m in models]
    vals={n:read(root/n/'metrics.json') for n in labels}
    corr={n:read(root/n/'correspondence.json') for n in labels}
    baseline,winner=c['baseline'],c['selected']
    keys=lambda v:{(r['room_seed'],r['target_view']) for r in v['rows']}
    for v in vals.values():
        assert keys(v)==keys(vals[baseline]) and v['visible_tokens']==vals[baseline]['visible_tokens']
    summary={}
    for n,v in vals.items():
        old={(r['room_seed'],r['target_view']):r for r in vals[baseline]['rows']}
        rows=[dict(r,baseline_mse=old[(r['room_seed'],r['target_view'])]['cross_mse']) for r in v['rows']]
        summary[n]=dict(latent={k:v[k] for k in v if k.startswith('mean_')},
            improvement_over_parent=ci_room(rows,'baseline_mse','cross_mse'),
            monocular_gain=ci_room(rows,'monocular_mse','cross_mse'),
            unrelated_gain=ci_room(rows,'unrelated_mse','cross_mse'),
            shuffled_gain=ci_room(rows,'spatially_shuffled_mse','cross_mse'),
            correspondence=corr[n]['summary'],input_audit=v['input_audit'])
        matching={}
        for method in ['student_encoder','centered_student','fused_decoder','fused_latent','reciprocal_attention']:
            before={(r['room_seed'],r['target_view'],r['reference_view']):r for r in corr[baseline]['rows'] if r['method']==method}
            paired=[dict(r,parent_epe=before[(r['room_seed'],r['target_view'],r['reference_view'])]['mean_epe'])
                for r in corr[n]['rows'] if r['method']==method]
            matching[method]=ci_room(paired,'parent_epe','mean_epe')
        summary[n]['room_mean_correspondence_gain']=matching
    budget=read(c['budget'])
    external=read(c['external']) if c.get('external') else None
    memory=read(c['memory']) if c.get('memory') else None
    summary['budget']=budget;summary['status']='completed_diagnostic; no SOTA claim'
    summary['selected']=winner
    summary['external']=external
    displacement=read(c['external_displacement']) if c.get('external_displacement') else None
    summary['external_displacement']=displacement
    summary['memory_audit']=memory
    fmt=lambda ci:f"{ci['mean']:.5f} [95% CI {ci['low']:.5f}, {ci['high']:.5f}]"
    w=vals[winner]; gain=1-w['mean_cross_mse']/w['mean_monocular_mse']
    transfer_cover='These measurements qualify an engineering experiment, not a state-of-the-art model. The next pages show correspondence baselines and external transfer so a lower latent loss cannot conceal geometric regressions.'
    if external:
        e=external['models'][winner]['summary']
        transfer_cover=f"Real-image transfer: HPatches viewpoint encoder AEPE improves from {e['fixed_teacher']['viewpoint']['aepe']:.2f} to {e['student_encoder']['viewpoint']['aepe']:.2f}px. Fusion fails to transfer as well: decoder {e['fused_decoder']['viewpoint']['aepe']:.2f}px and attention {e['reciprocal_attention']['viewpoint']['aepe']:.2f}px are worse than frozen features. This is not a state-of-the-art result."
    fresh_cover=''
    if c.get('fresh_assessment'):
        froot=Path(c['fresh_assessment']);fv=read(froot/winner/'metrics.json');fb=read(froot/baseline/'metrics.json')
        fresh_cover=f" Fresh 256-room test MSE is {fv['mean_cross_mse']:.5f}, an improvement of {100*(1-fv['mean_cross_mse']/fb['mean_cross_mse']):.2f}% over the parent."
    with PdfPages(out) as pdf:
        text_page(pdf,'Multi-view latent prediction: controlled continuation',[
            'Primary objective: predict fixed V-JEPA 2.1 latent features. Fusion and heads originated from random initialization; the student and teacher use audited MIT encoder weights. No noncommercial pretrained weights or teachers are in the candidate lineage. RGB reconstruction is an optional diagnostic, and its earlier blur remains unresolved.',
            f"Selected on synthetic validation: {winner}. Common evaluation covers {len(keys(w))} target views / {len({r['room_seed'] for r in w['rows']})} rooms, with {len(w['hidden_tokens'])}/256 hidden patches on the same fixed random mask. Old and new checkpoints are re-evaluated on identical inputs.",
            f"Validation latent MSE {w['mean_cross_mse']:.5f}; monocular {w['mean_monocular_mse']:.5f}; cross-view reduction {100*gain:.2f}%. Spatial variance ratio {w['mean_spatial_variance_ratio']:.3f}."+fresh_cover,
            f"Parent minus selected MSE: {fmt(summary[winner]['improvement_over_parent'])}. Monocular minus cross-view MSE: {fmt(summary[winner]['monocular_gain'])}. Intervals resample rooms, not individual pixels.",
            transfer_cover,
            f"Cumulative Pilot 07 command time: {budget['command_seconds']/3600:.2f} hours of the existing {budget['ceiling_seconds']/3600:.0f}-hour ceiling. Failed preflights are included. CPU implementation, reporting and external archive download are outside that command-time ledger."
        ])
        fig,axes=plt.subplots(2,2,figsize=(11.7,8.3));fig.suptitle('Identical-mask validation: utility and geometry',fontsize=17,y=.95)
        x=np.arange(len(labels))
        for ax,key,title in [(axes[0,0],'mean_cross_mse','Hidden latent MSE (lower)'),(axes[0,1],'mean_spatial_variance_ratio','Spatial variance / fixed teacher (higher diversity)')]:
            ax.bar(x,[vals[n][key] for n in labels],color='#4078a0');ax.set_xticks(x,labels,rotation=15);ax.set_title(title)
        methods=['fixed_teacher','centered_teacher','student_encoder','centered_student','fused_decoder','fused_latent','reciprocal_attention','same_position']
        for j,m in enumerate(methods):
            axes[1,0].bar(x+(j-3.5)*.10,[corr[n]['summary'][m]['mean_epe'] for n in labels],.10,label=m)
            axes[1,1].bar(x+(j-3.5)*.10,[corr[n]['summary'][m]['pck16'] for n in labels],.10,label=m)
        for ax,title in [(axes[1,0],'Patch-grid correspondence EPE (lower)'),(axes[1,1],'Visible queries within 16px (higher)')]:
            ax.set_xticks(x,labels,rotation=15);ax.set_title(title)
        handles,legend=axes[1,0].get_legend_handles_labels();fig.legend(handles,legend,loc='lower center',ncol=4,fontsize=8)
        fig.tight_layout(rect=(.02,.09,.98,.94));pdf.savefig(fig);plt.close(fig)
        paras=['All loss differences below are paired by room, with 10,000 bootstrap resamples. Positive values favor related cross-view prediction. The token shuffle preserves the feature multiset but cannot remove positional information embedded inside encoder features.']
        for n in labels:
            v=vals[n];s=summary[n]
            paras.append(f"{n}: mono minus cross {fmt(s['monocular_gain'])}; unrelated minus related {fmt(s['unrelated_gain'])}; shuffled minus ordered {fmt(s['shuffled_gain'])}.")
            ri=v['learned_ri_covisibility'];raw=v['latent_gain_covisibility']
            paras.append(f"{n}: raw gain AUROC {raw['auroc']:.3f}; learned RI AUROC {ri['auroc']:.3f}, AP {ri['average_precision']:.3f}; positive prevalence {ri['positives']/ri['pixels']:.3f}. RI is a learned error-improvement proxy, not a calibrated visibility probability.")
        text_page(pdf,'Controls and co-visibility',paras)
        fig,axes=plt.subplots(1,2,figsize=(11.7,5.8));fig.suptitle('Training convergence and measured update cost',fontsize=17,y=.94)
        efficiency={}
        for m in models:
            if not m.get('run'):continue
            run=Path(m['run']);r=read(run/'report.json');probes=r['probes']
            logs=[json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
            if m.get('prefix_run'):
                old=Path(m['prefix_run']);cut=r['starting_step']
                probes=[p for p in read(old/'probes.json') if p['step']<cut]+probes
                logs=[json.loads(l) for l in (old/'metrics.jsonl').read_text().splitlines() if json.loads(l)['step']<=cut]+logs
            axes[0].plot([p['step'] for p in probes],[p['cross_mse'] for p in probes],'-o',label=m['name'])
            axes[1].plot([p['step'] for p in logs],[p['seconds'] for p in logs],alpha=.7,label=m['name'])
            efficiency[m['name']]=dict(plotted_updates=len(logs),final_leg_seconds=r['run_seconds'],final_leg_targets_per_second=r['warm_targets_per_second'],final_leg_stage_steps=r['stage_steps'])
        axes[0].set(xlabel='Updates in this phase',ylabel='Validation hidden MSE');axes[1].set(xlabel='Update',ylabel='Seconds / update')
        for ax in axes:ax.legend();ax.grid(alpha=.2)
        fig.tight_layout(rect=(.02,.04,.98,.92));pdf.savefig(fig);plt.close(fig);summary['efficiency']=efficiency
        text_page(pdf,'Correspondence protocol and its limits',[
            'Each ordered camera pair is evaluated independently. Fixed teacher, adapted student and final fused decoder descriptors use cosine nearest neighbours on a 16px patch grid. The attention readout is the geometric mean of forward and transposed reverse attention, averaged across all heads and layers. This is not ZeroCo\'s complete refinement recipe.',
            'Only RGB enters feature extraction and matching. Ground-truth depth and cameras identify visible query points for scoring, after predictions. Visibility never filters a model\'s correspondence search. Query locations sample the central pixel of each patch. The grid oracle reports unavoidable quantization at this readout resolution.',
            *[f"{name}: EPE {v['mean_epe']:.2f}px; PCK16 {100*v['pck16']:.1f}%; mutual coverage {100*v['mutual_coverage']:.1f}%; mutual EPE {v['mutual_epe'] if v['mutual_epe'] is not None else 'not applicable'}." for name,v in corr[winner]['summary'].items()],
            'Do not compare these synthetic patch-grid numbers directly to published full-resolution ETH3D AEPE. They isolate whether training preserves or improves the fixed encoder under an identical local readout.'
        ])
        text_page(pdf,'Uncertainty of the correspondence improvement',[
            f"Paired room bootstrap: average pair-level errors within each room, then resample {len({r['room_seed'] for r in w['rows']})} rooms. Positive differences favor the selected checkpoint. These estimates weight rooms equally; the preceding EPE summary pools visible query points. Neither interval captures variability across independent training seeds.",
            *[f"{method}: parent minus selected room-mean EPE {fmt(ci)} pixels." for method,ci in summary[winner]['room_mean_correspondence_gain'].items()],
            'The matching evaluation supplies one reference at a time, whereas the completion task trains with two. This tests pairwise transfer of a model trained with reference sets. The frozen encoder baseline uses the same dense image batch shape to control numerical dispatch differences.'
        ])
        sample_root=Path(c['fresh_assessment']) if c.get('fresh_assessment') else root
        sample_corr=read(sample_root/winner/'correspondence.json')
        samples=sample_corr['annotated_samples']
        sample_split='Fresh test' if c.get('fresh_assessment') else 'Validation'
        seeds=sorted({s['room_seed'] for s in samples})[:2]
        for seed in seeds:
            folder=sample_root/winner/f'room-{seed}-view-0';meta=read(folder/'metadata.json')
            target=np.fromfile(folder/'target-rgb.f32',dtype='<f4').reshape(meta['rgb_shape'])
            ref=np.fromfile(folder/'reference-1-rgb.f32',dtype='<f4').reshape(meta['rgb_shape'])
            fig,axes=plt.subplots(3,1,figsize=(11.7,8.3));fig.suptitle(f'{sample_split} correspondences | room {seed}',fontsize=17,y=.95)
            for ax,method in zip(axes,['fixed_teacher','student_encoder','reciprocal_attention']):
                row=next(s for s in samples if s['room_seed']==seed and s['method']==method)
                points=row['points'];ids=np.linspace(0,len(points)-1,min(12,len(points))).astype(int)
                ax.imshow(np.concatenate([target,ref],axis=1));ax.set_title(method,fontsize=10);ax.axis('off')
                for j in ids:
                    p=points[j];q=np.array(p['query']);r=np.array(p['predicted'])+np.array([256,0]);gt=np.array(p['truth'])+np.array([256,0])
                    color='#35cc55' if p['epe']<=16 else '#e34040'
                    ax.plot([q[0],r[0]],[q[1],r[1]],color=color,lw=.8,alpha=.8)
                    ax.scatter(*q,s=9,c=color);ax.scatter(*r,s=12,c=color,marker='x');ax.scatter(*gt,s=10,c='#28c9ef',marker='o')
            fig.text(.06,.04,'12 evenly spaced valid queries. Green: error <=16px; red: larger error; cyan dots: geometric truth. No selection by model score.',size=9)
            fig.tight_layout(rect=(.02,.07,.98,.94));pdf.savefig(fig);plt.close(fig)
        folders=sorted((sample_root/winner).glob('room-*-view-0'))[:3]
        target_features=np.concatenate([np.fromfile(p/'target-latent.f32',dtype='<f4').reshape(256,-1) for p in folders])
        center=target_features.mean(0);_,_,vt=np.linalg.svd(target_features-center,full_matrices=False);basis=vt[:3].T
        projected=(target_features-center)@basis;lo,hi=np.quantile(projected,[.01,.99],axis=0)
        color=lambda x:np.clip((((x-center)@basis)-lo)/np.maximum(hi-lo,1e-8),0,1).reshape(16,16,3)
        fig,axes=plt.subplots(len(folders),4,figsize=(11.7,8.3));fig.suptitle('Latent feature maps | shared teacher PCA and color scale',fontsize=17,y=.95)
        for row,p in enumerate(folders):
            meta=read(p/'metadata.json');source=np.fromfile(p/'target-rgb.f32',dtype='<f4').reshape(meta['rgb_shape'])
            arrays=[source,color(np.fromfile(p/'target-latent.f32',dtype='<f4').reshape(256,-1)),
                color(np.fromfile(sample_root/baseline/p.name/'cross-latent.f32',dtype='<f4').reshape(256,-1)),
                color(np.fromfile(p/'cross-latent.f32',dtype='<f4').reshape(256,-1))]
            for col,(ax,array) in enumerate(zip(axes[row],arrays)):
                ax.imshow(array,interpolation='nearest');ax.axis('off')
                if row==0:ax.set_title(['Target RGB (truth)','Teacher features',baseline,winner][col],fontsize=11)
        fig.text(.06,.035,'Feature colors are not reconstructed RGB. PCA is fitted to fixed teacher features; identical basis and scale are used for predictions.',size=9)
        fig.tight_layout(rect=(.02,.07,.98,.94));pdf.savefig(fig);plt.close(fig)
        if c.get('fresh_assessment'):
            fresh=Path(c['fresh_assessment']);before=read(fresh/baseline/'metrics.json');after=read(fresh/winner/'metrics.json')
            old={(r['room_seed'],r['target_view']):r for r in before['rows']}
            assert keys(before)==keys(after) and before['visible_tokens']==after['visible_tokens']
            paired=[dict(r,parent_mse=old[(r['room_seed'],r['target_view'])]['cross_mse']) for r in after['rows']]
            match=read(fresh/winner/'correspondence.json')
            old_match=read(fresh/baseline/'correspondence.json')
            geometry=read(c['fresh_geometry']) if c.get('fresh_geometry') else None
            n=len({r['room_seed'] for r in paired})
            fresh_summary=dict(rooms=n,target_views=len(paired),parent_cross_mse=before['mean_cross_mse'],
                selected_cross_mse=after['mean_cross_mse'],selected_monocular_mse=after['mean_monocular_mse'],
                spatial_variance_ratio=after['mean_spatial_variance_ratio'],
                parent_gain=ci_room(paired,'parent_mse','cross_mse'),monocular_gain=ci_room(paired,'monocular_mse','cross_mse'),
                unrelated_gain=ci_room(paired,'unrelated_mse','cross_mse'),shuffled_gain=ci_room(paired,'spatially_shuffled_mse','cross_mse'),
                learned_ri=after['learned_ri_covisibility'],correspondence=match['summary'],
                geometry_audit={k:v for k,v in geometry.items() if k!='pairs'} if geometry else None)
            summary['fresh_synthetic']=fresh_summary
            fresh_gain={}
            for method in ['student_encoder','centered_student','fused_decoder','fused_latent','reciprocal_attention']:
                previous={(r['room_seed'],r['target_view'],r['reference_view']):r for r in old_match['rows'] if r['method']==method}
                pairs=[dict(r,parent_epe=previous[(r['room_seed'],r['target_view'],r['reference_view'])]['mean_epe'])
                       for r in match['rows'] if r['method']==method]
                fresh_gain[method]=ci_room(pairs,'parent_epe','mean_epe')
            fresh_summary['room_mean_correspondence_gain']=fresh_gain
            text_page(pdf,'Fresh synthetic holdout after checkpoint selection',[
                f"{n} newly generated rooms / {len(paired)} target views, held out from every checkpoint ancestor's training and validation seeds. The one train and one validation room required by the capture format are unused. Same published generator, 256px resolution, three cameras and 0.25 camera baseline; this tests new room seeds, not generator or camera distribution shift.",
                f"Parent MSE {before['mean_cross_mse']:.5f}; selected {after['mean_cross_mse']:.5f}; matched monocular {after['mean_monocular_mse']:.5f}. Parent minus selected {fmt(fresh_summary['parent_gain'])}; monocular minus cross-view {fmt(fresh_summary['monocular_gain'])}.",
                f"Unrelated minus related {fmt(fresh_summary['unrelated_gain'])}; shuffled minus ordered {fmt(fresh_summary['shuffled_gain'])}. Spatial variance ratio {after['mean_spatial_variance_ratio']:.3f}; learned RI AUROC {after['learned_ri_covisibility']['auroc']:.3f}.",
                *[f"{m}: EPE {match['summary'][m]['mean_epe']:.2f}px; PCK16 {100*match['summary'][m]['pck16']:.1f}%." for m in ['fixed_teacher','centered_teacher','student_encoder','centered_student','fused_decoder','fused_latent','reciprocal_attention','same_position','grid_oracle']],
                f"Parent minus selected room-mean EPE: encoder {fmt(fresh_gain['student_encoder'])}px; decoder {fmt(fresh_gain['fused_decoder'])}px; attention {fmt(fresh_gain['reciprocal_attention'])}px.",
                'All declared rooms are included. The checkpoint and readouts were fixed before this assessment. Intervals resample rooms; one training seed and one fixed masking seed limit generalization claims.',
                *([f"Independent full-pixel geometry audit: {geometry['valid_source_pixels']:,} valid source pixels across all {geometry['rooms']} captured rooms; maximum self-reprojection error {geometry['max_self_reprojection_pixels']:.5f}px and depth error {geometry['max_self_depth_error']:.3g}m."] if geometry else [])
            ])
        if external:
            paras=['HPatches is evaluated after synthetic checkpoint selection. The primary subset matches ZeroCo: 59 viewpoint sequences / 295 pairs. The other 57 illumination sequences and combined 580 pairs are supplementary. Input images are 256x256; metrics are at 240x240. Hard patch matches become bilinearly upsampled displacement. Homographies enter only the separate CPU scorer.',
                'This local readout differs from published ZeroCo refinement. Scores measure external transfer against the matched frozen encoder; they do not establish a published benchmark record.']
            for name,result in external['models'].items():
                encoder_ci=result['teacher_minus_candidate_aepe']['student_encoder']
                decoder_ci=result['teacher_minus_candidate_aepe']['fused_decoder']
                attention_ci=result['teacher_minus_candidate_aepe']['reciprocal_attention']
                paras.append(f"Main finding for {name}: adapted encoder improves over frozen encoder by {fmt(encoder_ci)}px. Fusion readouts regress: teacher minus decoder {fmt(decoder_ci)}px; teacher minus attention {fmt(attention_ci)}px. Negative differences mean the candidate is worse.")
                for method,groups in result['summary'].items():
                    m=groups['viewpoint'];paras.append(f"{name} / {method}: primary viewpoint AEPE {m['aepe']:.2f}px, PCK3 {100*m['pck3']:.1f}%; supplementary illumination AEPE {groups['illumination']['aepe']:.2f}, combined AEPE {groups['all']['aepe']:.2f}.")
                for method,ci in result['teacher_minus_candidate_aepe'].items():
                    paras.append(f"{name}: teacher minus {method} viewpoint AEPE {fmt(ci)}; 59-sequence bootstrap, positive favors candidate.")
                paras.append(f"{name}: centered teacher minus centered student viewpoint AEPE {fmt(result['centered_teacher_minus_centered_student'])}.")
            text_page(pdf,'External transfer: HPatches-240',paras)
            if displacement:
                attention=displacement['methods']['reciprocal_attention']['viewpoint']
                encoder=displacement['methods']['student_encoder']['viewpoint']
                text_page(pdf,'What the external failure says about fusion',[
                    f"On viewpoint pairs, reciprocal attention returns the identical patch position for {100*attention['exact_same_position']:.1f}% of queries and stays within one patch for {100*attention['within_one_patch']:.1f}%. The adapted encoder does so for {100*encoder['exact_same_position']:.1f}% and {100*encoder['within_one_patch']:.1f}%. These are post-evaluation diagnostics over all predictions, not ground-truth-filtered search or model selection.",
                    'Attention is close to the same-position baseline under real viewpoint changes. The observed concentration is consistent with a strong position prior in this readout; it does not by itself identify whether the cause is training data, objective, positional encoding or head/layer averaging.',
                    'Illumination sequences largely preserve geometry: the same-position baseline is already extremely accurate. Averaging illumination and viewpoint scores together can hide failed viewpoint transfer, which is why the viewpoint subset was declared primary before evaluation.',
                    'The next controlled hypotheses are broader camera-baseline training, frozen-versus-adapted encoder controls, hierarchical targets, and self-supervised augmentation correspondence or local refinement. These require new validation protocols. No further quality training or checkpoint selection used these HPatches outcomes.'
                ])
            if c.get('external_dataset') and c.get('external_export'):
                external_sample_pages(pdf,c['external_dataset'],c['external_export'],winner)
        if memory:
            old=memory['summary']['legacy-captures'];new=memory['summary']['final-only']
            text_page(pdf,'Training memory: isolated partial-unfreezing audit',[
                'The block-mask run reached 95,796 MiB of process VRAM and failed at update 896. Its intact update-800 checkpoint was resumed with both optimizers and completed update 1000. The failed tail is excluded from model selection; its cost remains in the budget.',
                'The isolated audit identifies unused trainable branches as a source of memory growth: the default V-JEPA forward captures intermediate normalized features, while the latent loss consumes only final tokens. During partial unfreezing these unused norms form disconnected autodiff graphs. The final-only path avoids constructing them. Loss logging also now uses the inner backend.',
                f"Same pretrained encoder, fixed inputs, last two blocks trainable, 200 updates, batch 48: legacy captures peaked at {old['peak_process_vram_mib']/1024:.2f} GiB and grew {old['growth_mib_per_update']:.1f} MiB/update after warmup. Final-only captures peaked and remained at {new['peak_process_vram_mib']/1024:.2f} GiB, with measured growth {abs(new['growth_mib_per_update']):.2f} MiB/update.",
                f"Warm median update time: legacy {1000*old['warm_median_seconds_per_update']:.1f}ms; final-only {1000*new['warm_median_seconds_per_update']:.1f}ms. Maximum loss-trajectory difference across all 200 updates: {memory['summary']['max_absolute_loss_difference']:.3g}; final difference {memory['summary']['final_absolute_loss_difference']:.3g}.",
                'This is a memory diagnostic using deterministic synthetic feature targets, not an additional quality-trained checkpoint. It uses no manual allocator cleanup. The comparison isolates intermediate capture; neither arm builds differentiable logging branches.',
                'A CPU regression checks exact dense/sparse final-output and trainable-QKV-gradient parity when captures are omitted. The completed quality comparisons retain their archived training binaries so the optimizer continuation remains exact.'
            ])
            fig,ax=plt.subplots(figsize=(11.7,6.5))
            for name,arm in memory['arms'].items():
                samples=[json.loads(line) for line in Path(arm['telemetry']).read_text().splitlines()]
                samples=[r for r in samples if 'process_vram_mib' in r]
                ax.plot([r['elapsed_seconds'] for r in samples],[r['process_vram_mib']/1024 for r in samples],label=name)
            ax.set(xlabel='Process elapsed seconds',ylabel='Process VRAM (GiB)',title='Unused hierarchical outputs caused sustained VRAM growth')
            ax.legend();ax.grid(alpha=.2)
            fig.text(.08,.025,'One-second process memory samples; allocator reservations included. No manual cleanup. Both paths complete 200 updates.',size=10)
            fig.tight_layout(rect=(.03,.07,.98,.96));pdf.savefig(fig);plt.close(fig)
        text_page(pdf,'Failures, reproducibility and next evidence gates',[
            'A float64 stable-attention preflight failed in Burn CUDA Fusion before training: Unsupported precision for fusion: f64. A TOML typo was rejected at parsing and corrected. The compact-mask arm later hit an allocation failure and was resumed from its intact update-800 checkpoint. All failed command costs remain in the budget.',
            'Experiments use float32. The previous narrow float64 inference audit is insufficient to qualify full training or evaluation. Native reference-order differences are measured and retained; CPU mathematical set symmetry must not be reported as bitwise GPU invariance.',
            'Model, optimizer, source archive and binary hashes are recorded under .data/pilot-07. Weights-only phase ancestry is checked recursively and exact resume preserves separate encoder/fusion optimizer states. The author-hosted HPatches archive is checksum-pinned, with RGB-only input and separate homography artifacts.',
            'Remaining claim gates: independent training seeds; camera-baseline and generator-distribution shifts; several mask seeds; co-visibility calibration; exact ETH3D/ZeroCo protocol parity and contemporary baselines. Test data must not guide another optimization phase without explicitly changing its status to development data.',
            'Architectural follow-ups should respond to measured failures: frozen-versus-adapted encoder controls, earlier or hierarchical latent targets, self-supervised augmentation correspondence, and local refinement. Geometry-supervised variants must be named separately.',
            'Primary sources and exact protocol registry: configs/train/benchmark-registry.toml; docs/sota-evidence.md. References: arxiv.org/abs/2609.01530 (Gekko); arxiv.org/abs/2603.14482 (V-JEPA 2.1); github.com/cvlab-kaist/ZeroCo; github.com/hpatches/hpatches-dataset.'
        ])
    summary['pdf_sha256']=hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(pdf=str(out),bytes=out.stat().st_size,sha256=summary['pdf_sha256'])))


if __name__=='__main__':main()
