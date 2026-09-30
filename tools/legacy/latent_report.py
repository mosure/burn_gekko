#!/usr/bin/env python3
"""Report a completed latent diagnostic from saved outputs; performs no GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import textwrap
import tomllib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def read(path):
    return json.loads(Path(path).read_text())


def ci_room(rows, first, second, seed=719):
    grouped = {}
    for row in rows:
        if row[first] is not None and row[second] is not None:
            grouped.setdefault(row['room_seed'], []).append(row[first]-row[second])
    values = np.array([np.mean(v) for v in grouped.values()])
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    estimates = values[rng.integers(0, len(values), (10000, len(values)))].mean(1)
    return dict(mean=float(values.mean()), low=float(np.quantile(estimates, .025)),
                high=float(np.quantile(estimates, .975)), rooms=len(values))


def page(pdf, title, paragraphs):
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(.06, .93, title, size=19, weight='bold')
    y = .86
    for paragraph in paragraphs:
        lines = textwrap.wrap(str(paragraph), width=120, break_long_words=True)
        fig.text(.06, y, '\n'.join(lines), va='top', size=11, linespacing=1.4)
        y -= .030*len(lines)+.029
    fig.text(.06, .035, 'burn_gekko | Pilot 07 latent diagnostic | RGB blur remains unresolved', size=9, color='#555555')
    pdf.savefig(fig); plt.close(fig)


def rank(values):
    x = values-values.mean(0)
    # All sampled tokens remain from the same exported targets for both arrays.
    sv = np.linalg.svd(x, compute_uv=False)
    p = sv**2 / max(float(np.sum(sv**2)), 1e-20)
    p = p[p > 0]
    return float(np.exp(-np.sum(p*np.log(p))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    c = tomllib.loads(args.config.read_text())
    run, output = Path(c['run']), Path(c['pdf'])
    assert output.resolve().is_relative_to(Path('.data').resolve())
    report, provenance = read(run/'report.json'), read(run/'provenance.json')
    val, initial = read(run/'validation/metrics.json'), read(run/'initial/metrics.json')
    rows = val['rows']
    by_key={(r['room_seed'],r['target_view']):r for r in rows}
    cross_gain = ci_room(rows, 'monocular_mse', 'cross_mse')
    unrelated_gain = ci_room(rows, 'unrelated_mse', 'cross_mse')
    constant_gain = ci_room(rows, 'train_position_mean_mse', 'cross_mse')
    samples = sorted((run/'validation').glob('room-*-view-*'))
    targets, predicted, hidden_targets, hidden_predicted = [], [], [], []
    export_mse_residuals=[]
    for path in samples:
        meta = read(path/'metadata.json')
        shape = meta['latent_shape']
        t = np.fromfile(path/'target-latent.f32', dtype='<f4').reshape(shape)
        p = np.fromfile(path/'cross-latent.f32', dtype='<f4').reshape(shape)
        targets.append(t); predicted.append(p)
        hidden_targets.append(t[meta['hidden_tokens']]); hidden_predicted.append(p[meta['hidden_tokens']])
        row=by_key[(meta['room_seed'],meta['target_view'])]
        mse=float(np.mean((t[meta['hidden_tokens']].astype(np.float64)-p[meta['hidden_tokens']])**2))
        export_mse_residuals.append(abs(mse-row['cross_mse']))
    assert max(export_mse_residuals)<1e-5, 'saved arrays disagree with reported hidden-token MSE'
    all_target = np.concatenate(targets)
    center = all_target.mean(0)
    _, _, vt = np.linalg.svd(all_target-center, full_matrices=False)
    basis = vt[:3].T
    projected = (all_target-center)@basis
    lo, hi = np.quantile(projected, [.01,.99], axis=0)
    def color(x, grid):
        return np.clip((((x-center)@basis)-lo)/(hi-lo).clip(1e-8), 0, 1).reshape(*grid, 3)
    # Fixed deterministic token sample for a bounded CPU rank calculation.
    ht, hp = np.concatenate(hidden_targets), np.concatenate(hidden_predicted)
    ids = np.linspace(0, len(ht)-1, min(2048,len(ht))).astype(int)
    ranks = dict(teacher=rank(ht[ids]), prediction=rank(hp[ids]), token_count=len(ids),
                 scope='hidden tokens from exported validation samples; diagnostic only')
    gpu_path = Path(c['gpu'])
    gpu = [json.loads(l) for l in gpu_path.read_text().splitlines()] if gpu_path.exists() else []
    process_peak = max((r.get('process_vram_mib',0) for r in gpu), default=None)
    metrics = [json.loads(l) for l in (run/'metrics.jsonl').read_text().splitlines()]
    phase_times={}
    for stage in sorted({m['stage'] for m in metrics}):
        part=[m for m in metrics if m['stage']==stage]
        measured=part[min(10,len(part)//2):]
        phase_times[stage]=dict(updates=len(part),median_seconds=float(np.median([m['seconds'] for m in measured])),
                               encoder_gradient_tensors=sorted({m['encoder_gradient_tensors'] for m in part}))
    summary = dict(task=provenance['task'], status='completed_diagnostic', rgb_blur_resolved=False,
                   cross_view_gain=cross_gain, unrelated_reference_gain=unrelated_gain,
                   training_constant_gain=constant_gain, feature_effective_rank=ranks,
                   process_peak_vram_mib=process_peak, checkpoint=read(run/'final/metadata.json'),
                   source_report=str(run/'report.json'), input_audit=val['input_audit'],
                   max_export_mse_residual=max(export_mse_residuals))
    summary['stage_efficiency']=phase_times
    output.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output) as pdf:
        page(pdf, 'V-JEPA latent prediction: first controlled screen', [
            'Task amendment: predict frozen V-JEPA 2.1 features instead of RGB. Fusion and heads are random; student and fixed teacher use the audited MIT encoder. No released Gekko or noncommercial teacher is used. This experiment changes the objective; it does not resolve RGB blur.',
            f"Run completed {report['completed_steps']:,} updates on {len(provenance['training_room_seeds'])} training rooms. Validation covers {len(provenance['validation_room_seeds'])} separate rooms / {val['target_views']} targets. The test split is not used.",
            f"Hidden latent MSE: initial {initial['mean_cross_mse']:.5f}; final cross-view {val['mean_cross_mse']:.5f}; monocular {val['mean_monocular_mse']:.5f}; unrelated references {val['mean_unrelated_mse']:.5f}; training-position mean {val['mean_train_position_mean_mse']:.5f}.",
            f"Cross-view cosine {val['mean_cross_cosine']:.4f}; monocular {val['mean_monocular_cosine']:.4f}. Mean prediction/teacher spatial variance ratio {val['mean_spatial_variance_ratio']:.3f}. A low MSE or a colorful PCA map alone does not establish useful multi-view learning.",
            f"Median update {report['median_update_seconds']:.3f}s; {report['warm_targets_per_second']:.1f} targets/s; process peak VRAM {process_peak} MiB. Run time {report['run_seconds']/60:.1f} minutes including preparation and evaluation.",
            'Single-seed diagnostic on procedural rooms. Fresh-scene testing, multiple masks and seeds, downstream matching/pose, and calibrated visibility remain separate qualification steps.'
        ])
        formatted = lambda ci: 'unavailable' if ci is None else f"{ci['mean']:.6f} [95% CI {ci['low']:.6f}, {ci['high']:.6f}], {ci['rooms']} rooms"
        raw_rank, ri_rank = val['latent_gain_covisibility'], val['learned_ri_covisibility']
        page(pdf, 'Reference utility, collapse controls and visibility', [
            'Paired room bootstrap, 10,000 resamples. Positive error differences favor the cross-view model. Individual pixels/tokens are not treated as independent rooms.',
            f"Monocular minus cross-view MSE: {formatted(cross_gain)}. Unrelated minus related MSE: {formatted(unrelated_gain)}.",
            f"Training-only constant position baseline minus cross-view MSE: {formatted(constant_gain)}. The baseline averages dense teacher features from the first 16 training rooms; validation is not used to fit it.",
            f"Effective rank of centered hidden features in saved samples: teacher {ranks['teacher']:.2f}, prediction {ranks['prediction']:.2f}, over {ranks['token_count']} deterministic tokens. This measures diversity, not correspondence accuracy.",
            f"Raw latent-gain visibility AUROC {raw_rank['auroc']:.3f}, AP {raw_rank['average_precision']:.3f}. Learned RI AUROC {ri_rank['auroc']:.3f}, AP {ri_rank['average_precision']:.3f}. Positive prevalence / constant-score AP baseline {raw_rank['positives']/raw_rank['pixels']:.3f} across {raw_rank['pixels']:,} hidden patches. These are modest visibility results.",
            'Geometry is evaluated on hidden patches only. At least half of a patch must have known labels; a majority of known pixels visible in any reference makes a positive label. Compare AP with positive prevalence. Geometry never enters training. Latent gain and learned RI are not calibrated probabilities.'
        ])
        fig, axes = plt.subplots(2,2,figsize=(11.7,8.3))
        steps = [m['step'] for m in metrics]
        for key in ['cross','monocular']:
            axes[0,0].plot(steps,[m[key] for m in metrics],alpha=.55,label=key)
        axes[0,0].set(title='Training hidden-token MSE',xlabel='Update',ylabel='MSE');axes[0,0].legend()
        probe_paths = [(0,run/'initial/metrics.json')]+[(int(p.parent.name.split('-')[-1]),p) for p in sorted(run.glob('step-*/metrics.json'))]
        for key,label in [('mean_cross_mse','cross'),('mean_monocular_mse','mono'),('mean_train_position_mean_mse','constant')]:
            axes[0,1].plot([s for s,p in probe_paths],[read(p)[key] for s,p in probe_paths],marker='o',label=label)
        axes[0,1].set(title='Fixed validation masks',xlabel='Update',ylabel='MSE');axes[0,1].legend()
        axes[1,0].plot(steps,[m['seconds'] for m in metrics],linewidth=.7)
        axes[1,0].set(title='Measured update time',xlabel='Update',ylabel='Seconds')
        axes[1,1].plot([s for s,p in probe_paths],[read(p)['mean_spatial_variance_ratio'] for s,p in probe_paths],marker='o')
        axes[1,1].set(title='Prediction / teacher spatial variance',xlabel='Update',ylabel='Ratio')
        for m in metrics:
            if m is metrics[0] or m['stage'] != metrics[m['step']-metrics[0]['step']-1]['stage']:
                for ax in axes.flat: ax.axvline(m['step'],color='#aaa',linestyle=':',linewidth=.8)
        fig.suptitle('Learning and efficiency; dotted lines mark student stage changes')
        fig.tight_layout(rect=(0,0,1,.95));pdf.savefig(fig);plt.close(fig)
        sample_rows=[(p,read(p/'metadata.json')) for p in samples]
        sample_rows.sort(key=lambda item: by_key[(item[1]['room_seed'],item[1]['target_view'])]['cross_mse'])
        selections=[('first exported',samples[0]),('median exported error',sample_rows[len(sample_rows)//2][0]),('worst exported error',sample_rows[-1][0])]
        for label,path in selections:
            meta=read(path/'metadata.json');grid=meta['grid'];shape=meta['latent_shape'];rgbshape=meta['rgb_shape']
            array=lambda name:np.fromfile(path/name,dtype='<f4')
            t,p,m=[array(name).reshape(shape) for name in ['target-latent.f32','cross-latent.f32','monocular-latent.f32']]
            target=array('target-rgb.f32').reshape(rgbshape)
            hidden=np.zeros(grid[0]*grid[1],bool);hidden[meta['hidden_tokens']]=True;hidden=hidden.reshape(grid)
            masked=target.copy();mask_pixel=np.repeat(np.repeat(hidden,16,0),16,1);masked[mask_pixel]=.15
            fig,axes=plt.subplots(4,3,figsize=(11.7,12.8))
            for ax,img,title in zip(axes[0],[masked,array('reference-1-rgb.f32').reshape(rgbshape),array('reference-2-rgb.f32').reshape(rgbshape)],['Observed target (hidden patches gray)','Reference 1','Reference 2']):
                ax.imshow(img);ax.set_title(title,size=10)
            for ax,img,title in zip(axes[1],[color(t,grid),color(p,grid),color(m,grid)],['Teacher latent PCA','Cross-view latent PCA','Monocular latent PCA']):
                ax.imshow(img,interpolation='nearest');ax.set_title(title,size=10)
            errors=[np.mean((p-t)**2,axis=1).reshape(grid),np.mean((m-t)**2,axis=1).reshape(grid)]
            vmax=max(float(np.quantile(np.concatenate([e[hidden] for e in errors]),.98)),1e-6)
            for ax,img,title in zip(axes[2],errors+[array('gain.f32').reshape(grid)],['Cross-view latent MSE','Monocular latent MSE','Detached error gain']):
                im=ax.imshow(np.ma.masked_where(~hidden,img),interpolation='nearest',vmin=0,vmax=1 if 'gain' in title else vmax,cmap='magma')
                ax.set_title(title,size=10);fig.colorbar(im,ax=ax,fraction=.045)
            for ax,img,title in zip(axes[3],[target,array('visibility-fraction.f32').reshape(grid),array('ri.f32').reshape(grid)],['Full target (evaluation only)','Renderer visible fraction','Learned RI (not probability)']):
                if img.ndim==3:ax.imshow(img)
                else:
                    im=ax.imshow(np.ma.masked_where(~hidden | (img<0 if 'fraction' in title else False),img),interpolation='nearest',vmin=0,vmax=1,cmap='viridis');fig.colorbar(im,ax=ax,fraction=.045)
                ax.set_title(title,size=10)
            for ax in axes.flat:ax.set_xticks([]);ax.set_yticks([])
            row=by_key[(meta['room_seed'],meta['target_view'])]
            fig.suptitle(f"{label}: room {meta['room_seed']}, target {meta['target_view']} | cross MSE {row['cross_mse']:.4f}",size=14)
            fig.text(.04,.015,'All PCA panels share a teacher-fitted basis and 1–99% color limits. Patch maps use nearest interpolation. These are not RGB reconstructions.',size=9)
            fig.tight_layout(rect=(0,.035,1,.97));pdf.savefig(fig);plt.close(fig)
        audit=val['input_audit']
        page(pdf,'Implementation checks and reproducibility',[
            f"Input audit on the first validation target: changing hidden RGB changes predictions by {audit['hidden_rgb_max_abs_delta']:.2g}; reversing references changes the monocular branch by {audit['monocular_reference_permutation_max_abs_delta']:.2g}. Default attention's cross-view reference-order discrepancy is {audit['reference_permutation_max_abs_delta']:.3g} maximum / {audit['reference_permutation_rms_delta']:.3g} RMS, exceeding the strict 1e-5 tolerance. The next page tests an explicit numerical intervention.",
            f"Teacher first-block QKV delta {report['teacher_max_abs_delta']}; latent head delta {report['prediction_head_max_abs_delta']:.6f}; first/last student block QKV delta {report['first_encoder_max_abs_delta']:.6f} / {report['last_encoder_max_abs_delta']:.6f}. Stage update counts {report['stage_steps']}.",
            f"Teacher identity: {provenance['teacher_id']}. Dataset identity: {provenance['dataset_id']}.",
            f"Final model SHA256: {summary['checkpoint']['model_sha256']}. Run: {run}. Resolved TOML config, per-update metrics, optimizer records, probes and raw arrays accompany this PDF.",
            'Teacher is fixed and features are recomputed online. Student features are never cached between updates. All raw data and artifacts remain under .data. Resume requires matching source/config/teacher/dataset/backend identity and restores both AdamW states plus the unfreezing gate.',
            'Sources: https://arxiv.org/abs/2603.14482 (V-JEPA 2.1); https://arxiv.org/abs/2609.01530 (Gekko). This fixed-teacher last-layer experiment is an adaptation, not a reproduction of either paper.'
        ])
        if c.get('precision') and Path(c['precision']).exists():
            numeric=read(c['precision'])
            summary['precision_audit']=numeric
            fast,stable=numeric['rows']
            phase_text='; '.join(f"stage {s}: {v['updates']} updates, {v['median_seconds']:.3f}s median, {v['encoder_gradient_tensors'][0]} encoder gradient tensors" for s,v in phase_times.items())
            page(pdf,'Attention precision diagnostic',[
                'The main training run used the default F32 backend. This separate one-target inference audit keeps all learned weights unchanged and uses F64 for softmax and value accumulation. It is not a new training run or a full-validation quality result.',
                f"Default attention: reference permutation maximum {fast['permutation_max_abs']:.6g}, RMS {fast['permutation_rms']:.6g}. Higher-precision accumulation: maximum {stable['permutation_max_abs']:.6g}, RMS {stable['permutation_rms']:.6g}. Hidden-RGB intervention remains zero in both modes.",
                f"Hidden latent MSE is {fast['hidden_mse']:.8f} with default attention and {stable['hidden_mse']:.8f} with the override. Median of three batch-one forward measurements: {np.median(fast['forward_seconds'])*1000:.2f}ms and {np.median(stable['forward_seconds'])*1000:.2f}ms respectively. This is a narrow diagnostic, not a training throughput benchmark.",
                f"Per-stage training timing, excluding initial updates within each stage: {phase_text}. Stage 0 freezes the encoder; stage 1 trains the last two blocks; stage 2 trains the image encoder and stem.",
                'A successful numerical intervention would justify a separately measured precision policy. One-target inference timings do not establish full-batch training cost. Hidden-target independence and reference-order numerical tolerance are separate checks.'
            ])
    summary['pdf_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix('.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'pdf':str(output),'summary':str(output.with_suffix('.json')),'cross_view_gain':cross_gain,'unrelated_gain':unrelated_gain,'rank':ranks},indent=2))


if __name__=='__main__':
    main()
