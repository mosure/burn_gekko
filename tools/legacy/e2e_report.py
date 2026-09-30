#!/usr/bin/env python3
"""Audit raw RGB exports and write the bounded, NC-free reconstruction study PDF."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import textwrap
import tomllib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import numpy as np

from pilot_report_data import bootstrap, read
from transport_diagnostics import inspect


def ci(rows, key):
    rooms = collections.defaultdict(list)
    for r in rows:
        if np.isfinite(r[key]):
            rooms[r['room_seed']].append(r[key])
    return bootstrap([np.mean(v) for v in rooms.values()])


def exports(path):
    root = Path(path)
    evaluation = read(root / 'evaluation.json')
    rows, samples = [], {}
    expected = {(r['room_seed'], r['target_view']): r for r in evaluation['targets']}
    for p in sorted(root.glob('room-*-view-*')):
        r, a = inspect(p)
        assert not a['meta']['oracle_statistics']
        key = r['room_seed'], r['target_view']
        r['monocular_mse'] = expected[key]['monocular_hidden_rgb_mse']
        r['monocular_gain'] = r['monocular_mse'] - r['mse']
        rows.append(r)
        samples[key] = p
    assert set(samples) == set(expected), 'every target needs a raw export'
    np.testing.assert_allclose(np.mean([r['mse'] for r in rows]), evaluation['mean_hidden_rgb_mse'], rtol=1e-4)
    keys = ['mse', 'psnr', 'edge_cosine', 'edge_energy', 'interior_edge_cosine',
            'interior_edge_energy', 'seam_edge_energy', 'monocular_mse', 'monocular_gain']
    return dict(rows=rows, metrics={k: ci(rows, k) for k in keys}), samples


def performance(arm):
    root = Path(arm['run'])
    r = read(root / 'report.json')
    config = tomllib.loads((root / 'config.toml').read_text())
    steps = [json.loads(l) for l in (root / 'metrics.jsonl').read_text().splitlines()]
    gpu = [json.loads(l) for l in Path(arm['telemetry']).read_text().splitlines()]
    stages = {}
    for stage in sorted(set(s['stage'] for s in steps)):
        phase = [s for s in steps if s['stage'] == stage]
        warm = phase[min(50, len(phase)):]  # each transition may compile new kernels
        if not warm:
            continue
        times = np.array([s['seconds'] for s in warm])
        stages[str(stage)] = dict(steps=len(phase), median_ms=float(np.median(times)*1000),
                                  p90_ms=float(np.quantile(times, .9)*1000),
                                  targets_per_second=float(len(times)*config['batch_size']/times.sum()),
                                  encoder_gradient_tensors=sorted(set(s['encoder_gradient_tensors'] for s in phase)))
    return dict(completed_steps=r['completed_steps'], seconds=r['seconds'], stage_steps=r['stage_steps'],
                stage_performance=stages, first_encoder_delta=r['first_encoder_max_abs_delta'],
                last_encoder_delta=r['last_encoder_max_abs_delta'], stem_delta=r['image_stem_max_abs_delta'],
                peak_process_vram_gib=max(g.get('process_vram_mib', 0) for g in gpu)/1024,
                device_gpu_utilization_mean=float(np.mean([g['device_gpu_percent'] for g in gpu if 'device_gpu_percent' in g])),
                gradient_clip_fraction=float(np.mean([s['gradient_norm']>1 for s in steps])),
                parameter_count=arm.get('parameter_count'), stop_reason=r['stop_reason'])


def geometry_conditioned(paths, diagnostics):
    rows=[]
    for (seed,view),path in paths.items():
        _,a=inspect(path)
        labels=np.fromfile(Path(diagnostics).parent/path.name/'visibility.u8',dtype='u1').reshape(a['hidden'].shape)
        error=np.mean((a['prediction']-a['target'])**2,axis=2)
        known=a['hidden'] & (labels!=255)
        visible=known & (labels==1);absent=known & (labels==0)
        rows.append(dict(room_seed=seed,target_view=view,
                         reference_visible_mse=float(error[visible].mean()) if visible.any() else float('nan'),
                         reference_absent_mse=float(error[absent].mean()) if absent.any() else float('nan'),
                         reference_visible_fraction=float(visible.sum()/max(known.sum(),1)),
                         unknown_fraction=float((a['hidden'] & (labels==255)).sum()/a['hidden'].sum())))
    return {k:ci(rows,k) for k in ['reference_visible_mse','reference_absent_mse','reference_visible_fraction','unknown_fraction']}


def collect(c):
    result, sample_paths = {}, {}
    for arm in c['arms']:
        root = Path(arm['run'])
        config=tomllib.loads((root/'config.toml').read_text())
        manifest=read(Path(config['dataset'])/'manifest.json')
        expected=lambda split:{(s['seed'],v) for s in manifest['scenes'] if s['split']==split for v in range(manifest['config']['cameras'])}
        provenance = read(root / 'provenance.json')
        assert not provenance['noncommercial_weight_dependencies'] and provenance['teacher'] is None
        assert provenance['cached_feature_bytes'] == 0
        validation, paths = exports(root / 'validation')
        assert set(paths)==expected('validation')
        unrelated = read(root / 'unrelated/evaluation.json')
        wrong = {(r['room_seed'], r['target_view']): r['hidden_rgb_mse'] for r in unrelated['targets']}
        for r in validation['rows']:
            r['reference_gain'] = wrong[(r['room_seed'], r['target_view'])] - r['mse']
        validation['metrics']['reference_gain'] = ci(validation['rows'], 'reference_gain')
        entry = dict(label=arm['label'], validation=validation, runtime=performance(arm), provenance=provenance,
                     checkpoint=read(root / 'final/metadata.json'))
        with (root/'final/model.mpk').open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest()==entry['checkpoint']['model_sha256']
        sample_paths[arm['name']] = paths
        if arm.get('test'):
            entry['test'], sample_paths[arm['name']+'_test'] = exports(Path(arm['test']) / 'samples')
            assert set(sample_paths[arm['name']+'_test'])==expected('test')
            for key in ['test','test_unrelated']:
                if not arm.get(key):continue
                p=read(Path(arm[key])/'provenance.json')
                assert p['evaluation_split']=='test' and Path(p['resume']).resolve()==(root/'final').resolve()
                for identity in ['identity','dataset_id','encoder_id']:assert p[identity]==entry['checkpoint'][identity]
                assert not p['noncommercial_weight_dependencies'] and p['teacher'] is None
            if arm.get('test_unrelated'):
                wrong=read(Path(arm['test_unrelated'])/'samples/evaluation.json')
                controls={(r['room_seed'],r['target_view']):r for r in wrong['targets']}
                assert set(controls)==expected('test')
                entry['monocular_control_max_mse_difference']=max(abs(r['monocular_mse']-controls[(r['room_seed'],r['target_view'])]['monocular_hidden_rgb_mse']) for r in entry['test']['rows'])
                wrong={(r['room_seed'],r['target_view']):r['hidden_rgb_mse'] for r in wrong['targets']}
                for r in entry['test']['rows']:
                    r['reference_gain']=wrong[(r['room_seed'],r['target_view'])]-r['mse']
                entry['test']['metrics']['reference_gain']=ci(entry['test']['rows'],'reference_gain')
        result[arm['name']] = entry
    study = Path(c['study'])
    used = sum(read(p).get('command_seconds', 0) for p in study.glob('*/ledger.json'))
    assert used <= 7200, 'study budget exceeded'
    selected = result[c['selected']]
    test = selected['test']['metrics']
    diag = read(c['diagnostics'])
    assert diag['checkpoint_sha256']==selected['checkpoint']['model_sha256'] and diag['split']=='test'
    assert diag['dataset_id']==selected['checkpoint']['dataset_id']
    assert {(r['room_seed'],r['target_view']) for r in diag['targets']}==set(sample_paths[c['selected']+'_test'])
    conditioned=geometry_conditioned(sample_paths[c['selected']+'_test'],c['diagnostics'])
    rankings=[dict(room_seed=r['room_seed'],auroc=r['covisibility']['auroc'],
                   average_precision=r['covisibility']['average_precision']) for r in diag['targets']
              if r['covisibility']['auroc'] is not None and r['covisibility']['average_precision'] is not None]
    ranking_intervals={key:ci(rankings,key) for key in ['auroc','average_precision']}
    overfit=None
    if c.get('overfit_sample'):
        overfit,_=inspect(Path(c['overfit_sample']))
    smallset=None
    if c.get('smallset'):
        spec=c['smallset'];root=Path(spec['run'])
        provenance=read(root/'provenance.json')
        assert provenance['initialization']['kind']=='scratch' and not provenance['noncommercial_weight_dependencies']
        train,_=exports(Path(spec['audit'])/'rgb-all')
        assert read(Path(spec['audit'])/'rgb-all/evaluation.json')['split']=='train'
        assert read(Path(spec['audit'])/'rgb-all/evaluation.json')['diagnostic_only']
        validation,_=exports(root/'validation')
        smallset=dict(training=train,validation=validation,runtime=performance(spec),selection_eligible=False)
    gates = dict(no_nc_weights=all(not r['provenance']['noncommercial_weight_dependencies'] for r in result.values()),
                 encoder_trained=selected['runtime']['first_encoder_delta']>0 and selected['runtime']['stem_delta']>0,
                 edges_aligned=test['edge_cosine']['mean']>=.4,
                 edge_energy=.5<=test['edge_energy']['mean']<=1.5,
                 within_patch_detail=test['interior_edge_cosine']['mean']>=.4 and test['interior_edge_energy']['mean']>=.5,
                 monocular_advantage=test['monocular_gain']['low']>0 and 1-test['mse']['mean']/test['monocular_mse']['mean']>=.1,
                 unrelated_advantage=test.get('reference_gain',{'low':0})['low']>0,
                 hidden_independence=diag['input_audit']['hidden_target_intervention_max_abs']==0,
                 reference_permutation=diag['input_audit']['reference_permutation_max_abs']<1e-5,
                 visual_review=c.get('visual_review_passed',False))
    return dict(arms=result, selected=c['selected'], command_seconds=used, gates=gates,
                quality_qualified=all(gates.values()), diagnostics=diag,
                geometry_conditioned_test=conditioned,
                covisibility_room_intervals=ranking_intervals,
                overfit_diagnostic=overfit,
                smallset_diagnostic=smallset,
                visual_review=c.get('visual_review','Pending')), sample_paths


def text_page(pdf, title, paragraphs):
    fig = plt.figure(figsize=(11.7,8.3)); y=.87
    fig.text(.065,.94,title,size=20,weight='bold')
    for paragraph in paragraphs:
        lines = textwrap.wrap(str(paragraph), width=123, break_long_words=True)
        height=.029*len(lines)+.026
        if y-height<.075:
            pdf.savefig(fig);plt.close(fig)
            fig=plt.figure(figsize=(11.7,8.3));fig.text(.065,.94,title+' (continued)',size=20,weight='bold');y=.87
        fig.text(.065,y,'\n'.join(lines),va='top',size=11)
        y-=height
    fig.text(.065,.035,'burn_gekko | Pilot 05 | raw exported RGB; no image enhancement',size=8,color='.4')
    pdf.savefig(fig);plt.close(fig)


def table_page(pdf,title,columns,rows):
    if len(rows)>16:
        for i in range(0,len(rows),16):
            table_page(pdf,title+f' ({i//16+1})',columns,rows[i:i+16])
        return
    fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off');ax.set_title(title,size=18,pad=20)
    tab=ax.table(cellText=rows,colLabels=columns,loc='center',cellLoc='left',colLoc='left')
    tab.auto_set_font_size(False);tab.set_fontsize(10);tab.scale(1,2)
    for (r,_),cell in tab.get_celld().items():
        if r==0:cell.set_facecolor('#d7e5f0');cell.set_text_props(weight='bold')
    fig.tight_layout();pdf.savefig(fig);plt.close(fig)


def format_ci(x):
    return f"{x['mean']:.5f} [{x['low']:.5f}, {x['high']:.5f}]"


def report(c,result,paths):
    selected=result['arms'][c['selected']]
    failed=', '.join(k.replace('_',' ') for k,v in result['gates'].items() if not v)
    audit=result['diagnostics']['input_audit'];ranking=result['diagnostics']['covisibility']
    out=Path(c['output']);out.parent.mkdir(parents=True,exist_ok=True)
    with PdfPages(out) as pdf:
        status='Quality gates passed' if result['quality_qualified'] else 'Training verified; reconstruction quality remains unqualified'
        text_page(pdf,'End-to-end reconstruction: pilot 05',[
            status+'. '+c.get('conclusion',''),
            'Objective: build the reconstruction pipeline with no noncommercial pretrained weights. Every fusion decoder and RGB/RI head is newly initialized. Compare an entirely random encoder with MIT V-JEPA 2.1 initialization and progressive unfreezing. Released Gekko and pilot 04 checkpoints are baseline-only, with no parameter reuse or teacher distillation.',
            f"Selected on validation: {selected['label']}. Test hidden RGB MSE {selected['test']['metrics']['mse']['mean']:.6f}; hidden edge cosine {selected['test']['metrics']['edge_cosine']['mean']:.3f}; predicted/true gradient energy {selected['test']['metrics']['edge_energy']['mean']:.3f}. Failed checks: {failed}.",
            f"Cumulative metered capture/experiment command time: {result['command_seconds']/60:.1f} minutes of 120. CPU compilation and report preparation are outside the command budget. Measured duration includes setup, checkpoint writes, evaluation and shader compilation.",
            result['visual_review'],
        ])
        text_page(pdf,'Architecture, optimization and provenance',[
            'Shared V-JEPA image encoder: 12 blocks, width 768, 12 heads, 16px patches. Fusion: fresh 6-block, width-384, 6-head CroCo-style decoder with 2D rotary coordinates and joint attention over an unordered reference set. A 256px target supplies 64 visible tokens; each of two references supplies all 256. Optional observed RGB patches accompany encoder features.',
            'Separate cross-view and monocular branches predict normalized patch content and their own mean/log standard deviation. Fixed ImageNet channel normalization converts the prediction to RGB. Dense target features are reserved for the RI branch. Ground-truth hidden RGB, patch statistics and renderer geometry cannot enter completion inference.',
            'The adapted RGB recipe weights cross-view RGB MSE by 10, monocular RGB MSE by 5, and RI by .1, with normalized-content/statistics auxiliary terms and cross-view hidden-pixel gradients. It is not an exact reproduction of the released training objective. Unequal branch weights are a confound for error-comparison utility; a matched-loss ablation remains necessary. Optional edge-energy supervision cannot certify aligned detail.',
            'Pretrained unfreezing: fusion-only, then last two encoder blocks, then all image blocks and stem. Each transition needs at least 500 stage updates, at least 2% validation improvement since stage entry, and no >10% regression from the best probe. Random initialization trains jointly from step zero. Unused video parameters stay frozen.',
            'Independent AdamW states apply actual encoder/decoder learning rates, after a joint global norm clip of 1. Decoder peak LR 3e-4; encoder ratio 0.1 for pretrained and 0.5 for scratch; weight decay .05; 100-update warmup and cosine decay. The initialization comparison therefore includes an explicit optimizer-schedule difference.',
            'Recorded validation-only amendment: the first V-JEPA arm used gradient-energy weight .02 and showed repeated unaligned patch patterns. Upcoming scratch and comparison arms disable this penalty; the original arm remains reported. Compare initializer choices using the no-energy arms. No test data influenced this amendment.',
            'A separate Q/K-normalization ablation initializes learned per-head query/key LayerNorm from scratch in both decoder attention slots. Released Gekko uses this in self-attention; cross-attention normalization here is an explicit extension. Earlier v1 source and executables are archived for exact reproduction.',
            'V-JEPA source: facebookresearch/vjepa2, MIT. Local package c408f68d... has all 158 encoder tensors checked against official EMA values after F16 storage conversion. Computation is Burn CUDA F32; small TF32-related parity residuals were documented previously. DINOv3 has separate custom license terms and is not used in this experiment.',
            'Source licenses, URLs, checksums, resolved TOML configs and raw outputs are retained under .data/pilot-05 and .data/runs/pilot-05-*. Original Gekko CC-BY-NC-SA weights are excluded. License terms apply separately from this workspace code license.',
        ])
        text_page(pdf,'Dataset and evaluation protocol',[
            'Published bevy_zeroverse 0.23.0 and bevy_zeroverse_burn 0.6.0; procedural indoor rooms, 256x256 RGB, three cameras, density .35. Fresh 2,048 training / 32 validation / 64 test rooms, seeds 2026120000 onward. All room seeds were checked against every earlier cache. Capture config, binary identity and raw shard hashes form the immutable dataset identity.',
            'Raw scenes are cached on disk in .data. Training retains RGB tensors on the GPU and recomputes encoder features on every update; no stale frozen feature cache survives an unfreeze. Training and monitoring exclude test scenes. Four validation rooms (all target views) gate transitions; all 32 validation rooms support final comparison.',
            'All 64 reserved test rooms and three target views are exported. Metrics use only hidden pixels. Edge metrics require both adjacent pixels hidden. Separate within-patch and seam metrics expose artificial patch boundaries. Inference uses RGB only; co-visibility ground truth is decoded afterward from renderer geometry.',
            'Exploratory selection rule was frozen before the last two V-JEPA ablations: prefer candidates meeting validation detail and reference-control thresholds, then rank by hidden edge cosine (MSE breaks ties). If none qualifies, evaluate the highest-edge-cosine checkpoint as a diagnostic. The one-room model is never a candidate; there is no tuning after reserved-test evaluation.',
            'Confidence intervals use 2,000 room-level bootstrap replicates; views of a room are not treated as independent samples. They quantify room-sampling uncertainty for these checkpoints, not training-seed variation. Images are only clipped for display, without sharpening or contrast adjustment; metrics use unclipped float32 predictions. These are synthetic-room results, not real-image transfer evidence.',
        ])
        inventory=read(Path(c['study'])/'dataset-summary.json')
        fig,axes=plt.subplots(1,3,figsize=(11.7,5));fig.suptitle('Generated room diversity: complete cached dataset',size=16)
        for ax,key in zip(axes,['layout','palette','lighting']):
            pairs=sorted(inventory[key].items());ax.barh([k for k,_ in pairs],[v for _,v in pairs]);ax.set_title(key);ax.set_xlabel('Rooms')
        fig.tight_layout();pdf.savefig(fig);plt.close(fig)
        rows=[]
        for a in result['arms'].values():
            m=a['validation']['metrics'];p=a['runtime']
            rows.append([a['label'],str(p['completed_steps']),f"{m['mse']['mean']:.6f}",f"{m['edge_cosine']['mean']:.3f}",f"{m['edge_energy']['mean']:.3f}",f"{m['monocular_mse']['mean']:.6f}"])
        table_page(pdf,'Full validation comparison (32 rooms / 96 targets)', ['Arm','Updates','Hidden MSE','Edge cosine','Edge energy','Mono MSE'],rows)
        validation_keys=[key for key in sorted(paths[c['arms'][0]['name']]) if key[1]==0][:2]
        for key in validation_keys:
            _,a=inspect(paths[c['arms'][0]['name']][key])
            fig,axes=plt.subplots(2,4,figsize=(11.7,7));fig.suptitle(f'Matched validation comparison: room {key[0]}, target {key[1]}',size=16)
            for ax,im,title in zip(axes[0],[a['target'],a['masked'],*a['references'][:2]],['Target','Observed input','Reference 1','Reference 2']):
                ax.imshow(im.clip(0,1),interpolation='nearest');ax.set_title(title);ax.axis('off')
            for ax,arm in zip(axes[1],c['arms']):
                row,other=inspect(paths[arm['name']][key])
                np.testing.assert_array_equal(a['target'],other['target']);np.testing.assert_array_equal(a['hidden'],other['hidden'])
                for lhs,rhs in zip(a['references'],other['references']):np.testing.assert_array_equal(lhs,rhs)
                ax.imshow(other['completion'].clip(0,1),interpolation='nearest');ax.set_title(f"{arm['label']}\nMSE {row['mse']:.5f}; edge {row['edge_cosine']:.3f}",size=10);ax.axis('off')
            for ax in axes[1,len(c['arms']):]:ax.axis('off')
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)
        rows=[]
        for name,a in result['arms'].items():
            if 'test' not in a:continue
            for key in ['mse','edge_cosine','edge_energy','interior_edge_cosine','interior_edge_energy','seam_edge_energy','monocular_gain','reference_gain']:
                if key not in a['test']['metrics']:continue
                rows.append([a['label'],key,format_ci(a['test']['metrics'][key])])
        table_page(pdf,'Reserved test: means and room-bootstrap 95% intervals',['Arm','Metric','Estimate [low, high]'],rows)
        for a in c['arms']:
            r=result['arms'][a['name']];root=Path(a['run']);steps=[json.loads(l) for l in (root/'metrics.jsonl').read_text().splitlines()];probes=read(root/'report.json')['probes']
            fig,axes=plt.subplots(2,2,figsize=(11.7,8.3));fig.suptitle(a['label']+' | convergence and actual unfreeze stages',size=16)
            axes[0,0].plot([s['step'] for s in steps],[s['loss'] for s in steps],lw=.6);axes[0,0].set_yscale('log');axes[0,0].set_title('Training objective (log scale)')
            axes[0,1].plot([p['step'] for p in probes],[p['mse'] for p in probes],marker='o');axes[0,1].set_yscale('log');axes[0,1].set_title('Fixed 4-room validation probe')
            axes[1,0].plot([s['step'] for s in steps],[s['seconds']*1000 for s in steps],lw=.6);axes[1,0].set_yscale('log');axes[1,0].set_title('Step time incl. first-use compilation')
            axes[1,1].step([s['step'] for s in steps],[s['stage'] for s in steps]);axes[1,1].set_yticks([0,1,2],['Frozen','Tail','All image']);axes[1,1].set_title('Encoder optimization stage')
            for ax in axes.flat:ax.set_xlabel('Absolute optimizer update');ax.grid(alpha=.25)
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)
        perf=[]
        for a in result['arms'].values():
            for stage,p in a['runtime']['stage_performance'].items():
                perf.append([a['label'],stage,str(p['steps']),f"{p['median_ms']:.1f}",f"{p['targets_per_second']:.1f}",f"{a['runtime']['peak_process_vram_gib']:.1f}"])
        table_page(pdf,'GPU training efficiency (exclude first 50 updates of each stage)', ['Arm','Stage','Updates','Median ms','Targets/s','Peak GiB'],perf)
        text_page(pdf,'Encoder changes, controls and co-visibility',[
            *[f"{a['label']}: stage steps {a['runtime']['stage_steps']}; max changes first block {a['runtime']['first_encoder_delta']:.6g}, last block {a['runtime']['last_encoder_delta']:.6g}, image stem {a['runtime']['stem_delta']:.6g}. No gradients or parameter changes are allowed in a frozen encoder." for a in result['arms'].values()],
            f"Selected checkpoint input audit: hidden-target replacement changes RGB by {audit['hidden_target_intervention_max_abs']:.6g}; reference permutation by {audit['reference_permutation_max_abs']:.6g}; monocular reference permutation by {audit['monocular_reference_permutation_max_abs']:.6g}. The CUDA reference-permutation result exceeds the declared 1e-5 tolerance. Its origin was not isolated; no tolerance was relaxed. CPU reference-order and batch-equivalence properties pass.",
            f"Dense RI ranking: pooled AUROC {ranking['auroc']:.4f}, AP {ranking['average_precision']:.4f}, over {ranking['pixels']:,} known pixels ({ranking['positives']:,} reference-visible). Scores are not calibrated probabilities. Geometry annotations are evaluation-only; the head is trained from reconstruction-error differences.",
            f"Cross-view completion improves hidden test MSE by {100*(1-selected['test']['metrics']['mse']['mean']/selected['test']['metrics']['monocular_mse']['mean']):.2f}% relative to monocular completion (required: 10%). Monocular MSE changes by at most {selected['monocular_control_max_mse_difference']:.3g} between correct- and unrelated-reference runs.",
            f"Mean per-target RI ranking, with room-level bootstrap: AUROC {format_ci(result['covisibility_room_intervals']['auroc'])}; AP {format_ci(result['covisibility_room_intervals']['average_precision'])}. Pooled pixel ranking and per-target ranking are distinct estimands.",
            f"Constant-score AP baseline (visibility prevalence): {result['diagnostics']['covisibility']['positives']/result['diagnostics']['covisibility']['pixels']:.4f}. Compare AP against this prevalence, and AUROC against 0.5.",
            f"Hidden RGB conditioned on evaluation geometry (room/view means, unknown labels excluded): reference-visible MSE {format_ci(result['geometry_conditioned_test']['reference_visible_mse'])}; absent MSE {format_ci(result['geometry_conditioned_test']['reference_absent_mse'])}. Visible fraction of known hidden pixels {result['geometry_conditioned_test']['reference_visible_fraction']['mean']:.3f}; unknown fraction {result['geometry_conditioned_test']['unknown_fraction']['mean']:.3f}.",
            'GPU utilization/power are device-wide. Another bevy_zeroverse indoor_validate process was observed during the study; it was left untouched. Process VRAM is measured separately. Throughput is per target example (plus its references), not per isolated encoded image. Compilation/setup costs remain included in end-to-end duration.',
        ])
        if c.get('overfit_sample'):
            row,a=inspect(Path(c['overfit_sample']))
            fig,axes=plt.subplots(1,3,figsize=(11.7,5));fig.suptitle(f"One-room memorization diagnostic: training sample only | MSE {row['mse']:.6f}; edge {row['edge_cosine']:.3f}",size=13)
            for ax,im,title in zip(axes,[a['target'],a['masked'],a['completion']],['Training target','Fixed observed mask','Reconstruction after training']):
                ax.imshow(im.clip(0,1),interpolation='nearest');ax.set_title(title);ax.axis('off')
            fig.text(.05,.015,'Smaller random model: 192-wide, 4-block encoder and decoder; one fixed target/mask. This is not held-out evidence.',size=9)
            fig.tight_layout(rect=[0,.04,1,1]);pdf.savefig(fig);plt.close(fig)
        if result['smallset_diagnostic']:
            diagnostic=result['smallset_diagnostic'];runtime=diagnostic['runtime']
            table_page(pdf,f"16-room capacity diagnostic | {runtime['completed_steps']} updates | selection-ineligible",
                       ['Split','Targets','Hidden MSE','Edge cosine','Edge energy'],
                       [[label,str(len(diagnostic[key]['rows'])),f"{diagnostic[key]['metrics']['mse']['mean']:.6f}",
                         f"{diagnostic[key]['metrics']['edge_cosine']['mean']:.3f}",f"{diagnostic[key]['metrics']['edge_energy']['mean']:.3f}"]
                        for key,label in [('training','Training rooms'),('validation','Unseen validation rooms')]])
            text_page(pdf,'Small-set diagnostic interpretation',[
                'Random Base image encoder and fresh fusion decoder, with the same widths as the main scratch arm. Sixteen training rooms, all three target views and varying masks. Evaluation exports every training target using the shared evaluation mask, and all 32 validation rooms. Training examples are explicitly marked diagnostic-only.',
                f"Actual optimization: {runtime['completed_steps']} updates; stop reason {runtime['stop_reason']}; measured run {runtime['seconds']:.1f}s; peak process VRAM {runtime['peak_process_vram_gib']:.1f}GiB. The 4,000-step cosine horizon is unchanged from the main scratch arm; this short run is not a matched convergence comparison.",
            'This short run did not fit its own varied training targets: it is evidence of incomplete optimization under this recipe and cap, not merely a held-out generalization gap. It cannot establish reference matching, real-scene transfer or a qualified reconstruction model. Its checkpoint is excluded from candidate selection.',
            ])
        keys=sorted(paths[c['selected']+'_test'])
        rows=selected['test']['rows'];ranked=sorted(rows,key=lambda r:r['mse'])
        chosen=[('First reserved rooms',key) for key in [k for k in keys if k[1]==0][:3]]
        chosen += [('Median error',(ranked[len(ranked)//2]['room_seed'],ranked[len(ranked)//2]['target_view'])),('Worst error',(ranked[-1]['room_seed'],ranked[-1]['target_view']))]
        for title,key in chosen:
            p=paths[c['selected']+'_test'][key];row,a=inspect(p);mono=np.fromfile(p/'monocular.f32',dtype='<f4').reshape(a['target'].shape);mono[~a['hidden']]=a['target'][~a['hidden']]
            fig,axes=plt.subplots(2,3,figsize=(11.7,8.3));fig.suptitle(f"{title}: room {key[0]}, target {key[1]} | MSE {row['mse']:.6f}, edge {row['edge_cosine']:.3f}",size=14)
            images=[a['target'],a['masked'],a['references'][0],a['references'][1],a['completion'],mono]
            labels=['Ground truth','Observed target (25%)','Reference 1','Reference 2','Cross-view completion','Monocular completion']
            for ax,im,label in zip(axes.flat,images,labels):ax.imshow(im.clip(0,1),interpolation='nearest');ax.set_title(label,size=11);ax.axis('off')
            fig.tight_layout();pdf.savefig(fig)
            if title=='First reserved rooms' and key==keys[0]:fig.savefig(out.parent/'annotated-sample.png',dpi=150)
            plt.close(fig)
            # Select a detail region from target edges, independently of the prediction.
            target=a['target'];h,w=target.shape[:2]
            edges=np.zeros((h,w),dtype=float)
            edges[:-1]+=np.mean(np.diff(target,axis=0)**2,axis=2)
            edges[:,:-1]+=np.mean(np.diff(target,axis=1)**2,axis=2)
            patch_score=(edges*a['hidden']).reshape(h//16,16,w//16,16).sum(axis=(1,3))
            py,px=np.unravel_index(np.argmax(patch_score),patch_score.shape)
            y=min(max(py*16-24,0),h-64);x=min(max(px*16-24,0),w-64)
            fig,axes=plt.subplots(1,4,figsize=(11.7,4));fig.suptitle(f'Annotated detail: room {key[0]}, view {key[1]}',size=15)
            axes[0].imshow(target.clip(0,1));axes[0].add_patch(Rectangle((x,y),64,64,fill=False,color='red',lw=1.5));axes[0].set_title('Target / crop location')
            error=np.mean(np.abs(a['prediction']-target),axis=2);error[~a['hidden']]=np.nan
            im=axes[1].imshow(error,cmap='magma',vmin=0,vmax=.2);axes[1].set_title('Hidden RGB absolute error')
            fig.colorbar(im,ax=axes[1],fraction=.046,pad=.04)
            axes[2].imshow(target[y:y+64,x:x+64].clip(0,1),interpolation='nearest');axes[2].set_title('Target detail (64px)')
            axes[3].imshow(a['completion'][y:y+64,x:x+64].clip(0,1),interpolation='nearest');axes[3].set_title('Completion detail (64px)')
            for ax in axes:ax.axis('off')
            fig.text(.05,.025,'Crop rule: highest target-edge energy in a hidden patch, with 64px context. Error scale is fixed; white pixels were observed.',size=9)
            fig.tight_layout(rect=[0,.07,1,1]);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(2,3,figsize=(11.7,8.3));fig.suptitle('Co-visibility: evaluation geometry and learned RI (not probabilities)',size=15)
        for row_index,key in enumerate(keys[:2]):
            path=paths[c['selected']+'_test'][key];_,a=inspect(path)
            geometry=Path(c['diagnostics']).parent/path.name
            labels=np.fromfile(geometry/'visibility.u8',dtype='u1').reshape(a['hidden'].shape)
            ri=np.fromfile(geometry/'ri.f32',dtype='<f4').reshape(a['hidden'].shape)
            colors=np.zeros((*labels.shape,3));colors[labels==1]=[.2,.75,.4];colors[labels==0]=[.75,.2,.2];colors[labels==255]=[.5,.5,.5]
            axes[row_index,0].imshow(a['target'].clip(0,1));axes[row_index,0].set_title(f'Room {key[0]}, view {key[1]}')
            axes[row_index,1].imshow(colors);axes[row_index,1].set_title('Green: reference-visible; red: absent')
            im=axes[row_index,2].imshow(ri,cmap='coolwarm',vmin=-1,vmax=1);axes[row_index,2].set_title('Learned relative improvement')
            for ax in axes[row_index]:ax.axis('off')
            fig.colorbar(im,ax=axes[row_index,2],fraction=.035,pad=.02)
        fig.tight_layout();pdf.savefig(fig);plt.close(fig)
        text_page(pdf,'Limits and next experiments',[
            c.get('conclusion',''),
            'Next bounded study: require the full-size model to fit the 16-room varying-mask subset before scaling data. Compare matched cross-view/monocular loss weights and a delayed RI objective; ablate the forced prediction normalization against unconstrained normalized content or direct RGB. These are proposed tests, not demonstrated fixes. Resolve the strict CUDA reference-order residual separately.',
            'The scratch arm and MIT-pretrained arm have different prior knowledge, encoder learning rates and unfreeze schedules. This study tests an operational path and measures its behavior; it cannot attribute every difference solely to weight initialization. Short synthetic training is not compute-matched to the released Gekko model’s much larger pretraining history.',
            'The optional matched frozen-encoder control was displaced by the loss and attention-normalization comparisons. Parameter changes verify that unfreezing operates; they do not establish that unfreezing caused an improvement relative to keeping the encoder frozen.',
            'Better RGB error with related references can reflect scene color and context. It does not by itself establish local correspondence. The detailed one-room fit establishes capacity on that training input, while the unseen-room edge and image checks assess the unresolved generalization problem.',
            'The prior noncommercial hybrid remains a quality reference, not an admissible deliverable under the project requirement. Its prior test metrics use different rooms and masks; no cross-study percentage improvement is asserted here. Checkpoints in this report contain no Gekko pretrained parameters.',
            'For the paper: distinguish wholly random training, permissive encoder adaptation, frozen-encoder control and NC reference baseline. Report complete budgets and provenance; require repeat seeds, data scaling, real-room transfer, sparse/reference-count ablations, geometry ranking and perceptual evaluation before claiming a reusable fusion model or a research contribution.',
            'Reproduction: build target/pilot/e2e_pilot with --profile pilot --features cuda --locked. Study command plans, telemetry, resolved configs, dataset identity, encoder/model/optimizer checksums, tests and input interventions accompany this PDF. Continuation restores both optimizers and the absolute mask/sample schedule in a fresh output directory.',
        ])
    print(out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    c=tomllib.loads(a.config.read_text());result,paths=collect(c)
    Path(c['summary']).write_text(json.dumps(result,indent=2)+'\n');report(c,result,paths)

if __name__=='__main__':main()
