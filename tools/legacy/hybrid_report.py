#!/usr/bin/env python3
"""Recompute full-split quality from raw exports and produce a reviewable PDF.

All configuration is TOML. JSON files are immutable measured-result artifacts.
No enhancement, target-statistic inversion, selection, training, or model changes.
"""
import argparse
import collections
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

from pilot_report_data import bootstrap, read
from reconstruction_diagnostics import inspect_sample
from transport_diagnostics import inspect


def room_ci(rows, key):
    groups = collections.defaultdict(list)
    for row in rows:
        value = row[key]
        if value is not None:
            groups[row['room_seed']].append(value)
    return bootstrap([np.mean(v) for v in groups.values()])


def performance(run, telemetry):
    report = read(run / 'report.json')
    config = tomllib.loads((run / 'config.toml').read_text())
    metrics = [json.loads(s) for s in (run / 'metrics.jsonl').read_text().splitlines()]
    warm = np.array([m['seconds'] for m in metrics[100:]])
    gpu = [json.loads(s) for s in telemetry.read_text().splitlines()]
    return dict(steps=report['completed_steps'], seconds=report['seconds'],
                warm_step_median_ms=float(np.median(warm) * 1000),
                warm_step_p90_ms=float(np.quantile(warm, .9) * 1000),
                warm_images_per_second=float(len(warm) * config['batch_size'] / warm.sum()),
                end_to_end_images_per_second=len(metrics) * config['batch_size'] / report['seconds'],
                peak_process_vram_gib=max(g.get('process_vram_mib', 0) for g in gpu) / 1024,
                device_gpu_percent_mean=float(np.mean([g['device_gpu_percent'] for g in gpu if 'device_gpu_percent' in g])),
                device_power_w_mean=float(np.mean([g['device_power_w'] for g in gpu if 'device_power_w' in g])),
                gradient_clip_fraction=float(np.mean([m['gradient_norm'] > 1 for m in metrics])),
                first_decoder_block_delta=report.get('first_decoder_block_max_abs_delta'),
                last_decoder_block_delta=report.get('last_decoder_block_max_abs_delta'),
                frozen_head_delta=report['frozen_head_max_abs_delta'])


def collect(c):
    study = Path(c['study'])
    correct_root, wrong_root = Path(c['correct']), Path(c['unrelated'])
    correct = read(correct_root / 'samples/evaluation.json')
    wrong = read(wrong_root / 'samples/evaluation.json')
    baseline = read(c['baseline_evaluation'])
    audit = read(c['input_audit'])
    rows, samples = [], {}
    visibility_rows = []
    for path in sorted((correct_root / 'samples').glob('room-*-view-*')):
        row, array = inspect(path)
        assert not array['meta'].get('oracle_statistics'), 'oracle RGB cannot qualify'
        rows.append(row)
        samples[(row['room_seed'], row['target_view'])] = path
        label_path = Path(c['covisibility']).parent / path.name / 'visibility.u8'
        labels = np.fromfile(label_path, dtype='u1').reshape(array['hidden'].shape)
        assert set(np.unique(labels)).issubset({0,1,255})
        errors = np.mean((array['prediction'] - array['target'])**2, axis=-1)
        covered = array['hidden'] & (labels == 1)
        uncovered = array['hidden'] & (labels == 0)
        visibility_rows.append(dict(room_seed=row['room_seed'],
                                    reference_visible_mse=float(errors[covered].mean()) if covered.any() else None,
                                    reference_absent_mse=float(errors[uncovered].mean()) if uncovered.any() else None,
                                    reference_visible_fraction=float(covered.sum()/max(1,(covered | uncovered).sum()))))
    # Every target, not only a selected image panel, must be independently audited.
    expected = {(t['room_seed'], t['target_view']) for t in correct['targets']}
    assert set(samples) == expected and len(rows) == len(expected)
    control = {(t['room_seed'], t['target_view']): t for t in wrong['targets']}
    old = {(t['room_seed'], t['target_view']): t for t in baseline['targets']}
    assert set(control) == set(old) == expected
    paired = []
    mono_mses = []
    for t in correct['targets']:
        key = t['room_seed'], t['target_view']
        assert t['monocular_hidden_rgb_mse'] is not None
        mono_mses.append(t['monocular_hidden_rgb_mse'])
        paired.append(dict(room_seed=t['room_seed'],
                           baseline_gain=old[key]['masked_rgb_mse'] - t['hidden_rgb_mse'],
                           reference_gain=control[key]['hidden_rgb_mse'] - t['hidden_rgb_mse'],
                           monocular_gain=t['monocular_hidden_rgb_mse'] - t['hidden_rgb_mse']))
    mono_difference = max(abs(t['monocular_hidden_rgb_mse'] - control[(t['room_seed'], t['target_view'])]['monocular_hidden_rgb_mse']) for t in correct['targets'])
    mean = correct['mean_hidden_rgb_mse']
    np.testing.assert_allclose(np.mean([r['mse'] for r in rows]), mean, rtol=1e-4)
    stats = {k: room_ci(rows, k) for k in ['mse', 'psnr', 'edge_cosine', 'edge_energy', 'edge_mae','interior_edge_cosine','interior_edge_energy','seam_edge_energy','seam_edge_mae']}
    gains = {k: room_ci(paired, k) for k in ['baseline_gain', 'reference_gain', 'monocular_gain']}
    old_samples = []
    matched_new = []
    for path in sorted(Path(c['baseline_samples']).glob('room-*-view-*')):
        a, metrics = inspect_sample(path)
        assert not metrics['oracle_statistics']
        key = metrics['room_seed'], a['meta']['target_view']
        _, new = inspect(samples[key])
        np.testing.assert_array_equal(a['target'], new['target'])
        np.testing.assert_array_equal(a['hidden'], new['hidden'])
        old_samples.append(metrics)
        matched_new.append(next(r for r in rows if (r['room_seed'], r['target_view']) == key))
    covis = read(c['covisibility'])
    covis_metric = covis['covisibility']
    ledgers = [(p, read(p)) for p in sorted(study.glob('*/ledger.json'))]
    seconds = sum(l.get('command_seconds', 0) for _, l in ledgers)
    protocol = tomllib.loads((study / 'protocol.toml').read_text())
    q = protocol['qualification']
    improvement = 1 - mean / baseline['mean_masked_rgb_mse']
    gates = dict(
        rgb_improvement=improvement >= q['minimum_relative_rgb_mse_improvement'],
        aligned_edges=stats['edge_cosine']['mean'] >= q['minimum_hidden_edge_cosine'],
        edge_energy=q['minimum_predicted_to_true_edge_energy'] <= stats['edge_energy']['mean'] <= q['maximum_predicted_to_true_edge_energy'],
        interior_edges_not_only_seams=stats['interior_edge_cosine']['mean'] >= q['minimum_hidden_edge_cosine'] and stats['interior_edge_energy']['mean'] >= q['minimum_predicted_to_true_edge_energy'],
        correct_reference_advantage=gains['reference_gain']['low'] > 0,
        monocular_advantage=gains['monocular_gain']['low'] > 0,
        hidden_target_independence=audit['hidden_target_intervention_max_abs'] == 0,
        reference_permutation=audit['reference_permutation_max_abs'] < 1e-6,
        command_budget=seconds <= protocol['command_budget_seconds'],
        fresh_test_rooms=baseline['dataset_id'] != baseline['training_dataset_id'],
    )
    manifest = read(Path(c['dataset']) / 'manifest.json')
    assert covis['dataset_id'] == baseline['dataset_id'] == manifest['dataset_id']
    seed_sets = [set(s['seed'] for s in read(Path(p) / 'manifest.json')['scenes']) for p in c['prior_datasets']]
    assert all(not {s['seed'] for s in manifest['scenes']} & seeds for seeds in seed_sets)
    result = dict(gates=gates, numeric_gates_passed=all(gates.values()),
                  visual_review=c['visual_review'], visual_review_passed=c['visual_review_passed'],
                  fresh_test_dataset_id=manifest['dataset_id'],
                  test_rooms=len({t['room_seed'] for t in correct['targets']}), targets=len(expected),
                  baseline_rgb_mse=baseline['mean_masked_rgb_mse'], hybrid_rgb_mse=mean,
                  unrelated_rgb_mse=wrong['mean_hidden_rgb_mse'], monocular_rgb_mse=float(np.mean(mono_mses)),
                  relative_rgb_improvement=improvement, statistics=stats, paired_differences=gains,
                  max_monocular_control_difference=mono_difference,
                  matched_export_baseline_edge_cosine=float(np.mean([s['edge_cosine'] for s in old_samples])),
                  matched_export_hybrid_edge_cosine=float(np.mean([s['edge_cosine'] for s in matched_new])),
                  matched_export_baseline_edge_energy=float(np.mean([s['predicted_to_true_edge_energy'] for s in old_samples])),
                  matched_export_hybrid_edge_energy=float(np.mean([s['edge_energy'] for s in matched_new])),
                  matched_export_rooms=len(old_samples), covisibility=covis_metric,
                  covisibility_ap_prevalence_baseline=covis_metric['positives']/covis_metric['pixels'],
                  covisibility_adapter_enabled=covis['adapter_enabled'], command_seconds=seconds,
                  covisibility_checkpoint_sha256=covis['checkpoint_sha256'],
                  visibility_strata={k:room_ci(visibility_rows,k) for k in ['reference_visible_mse','reference_absent_mse','reference_visible_fraction']},
                  commands=[dict(stage=p.parent.name, **r) for p,l in ledgers for r in l['commands']],
                  performance=performance(Path(c['training_run']), Path(c['training_telemetry'])),
                  training_stages=[dict(label=t['label'], **performance(Path(t['path']), Path(t['telemetry']))) for t in c['training_stages']],
                  samples=rows, input_audit=audit, findings=c['findings'], limitations=c['limitations'])
    result['accepted'] = result['numeric_gates_passed'] and c['visual_review_passed']
    (study / 'quality-summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result, samples


def page(pdf, title, paragraphs):
    def new_page(continued=False):
        fig = plt.figure(figsize=(11.7, 8.3), facecolor='white')
        fig.text(.055, .925, title + (' — continued' if continued else ''), fontsize=20, weight='bold', color='#17384a')
        return fig
    def finish(fig):
        fig.text(.06, .035, 'burn_gekko / Pilot 04 / 28 September 2026 / measured local artifacts', fontsize=8, color='#61717c')
        pdf.savefig(fig); plt.close(fig)
    fig = new_page(); y = .85
    for paragraph in paragraphs:
        lines = textwrap.wrap(paragraph, width=118, break_long_words=False)
        height = .028 * len(lines) + .022
        if y - height < .07:
            finish(fig); fig = new_page(True); y = .85
        fig.text(.06, y, '\n'.join(lines), va='top', fontsize=11.2, linespacing=1.4)
        y -= height
    assert y > .04, f'page overflow: {title}'
    finish(fig)


def ci(x, scale=1):
    return f"{x['mean']*scale:.5f} [{x['low']*scale:.5f}, {x['high']*scale:.5f}]"


def completion_panels(pdf, baseline_root, samples, seed):
    path = samples[(seed, 0)]
    metrics, a = inspect(path)
    old, old_metrics = inspect_sample(baseline_root / path.name)
    fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3))
    images = [a['target'], a['masked'], a['references'][0], a['references'][1], old['completion'], a['completion']]
    labels = ['Target (evaluation only)', 'Observed target: 75% hidden', 'Reference 1', 'Reference 2',
              f"Previous model / {old_metrics['hidden_rgb_psnr']:.2f} dB", f"Hybrid / {metrics['psnr']:.2f} dB"]
    for ax, image, label in zip(axes.flat, images, labels):
        ax.imshow(image.clip(0,1), interpolation='nearest'); ax.axis('off'); ax.set_title(label, fontsize=11)
    fig.suptitle(f'Fresh room {seed}, target 0 — same pixels and mask', fontsize=17, y=.97)
    fig.text(.06,.03,f"Hidden edge cosine: {old_metrics['edge_cosine']:.3f} → {metrics['edge_cosine']:.3f}. "
             f"Edge energy / target: {old_metrics['predicted_to_true_edge_energy']:.3f} → {metrics['edge_energy']:.3f}.\n"
             'Completions retain visible input patches. Hidden regions use unenhanced predictions; no target statistics.', fontsize=9)
    fig.subplots_adjust(left=.03,right=.97,top=.91,bottom=.11,hspace=.14,wspace=.08)
    pdf.savefig(fig);plt.close(fig)


def detail_panel(pdf, samples, baseline_root, seeds):
    fig, axes = plt.subplots(len(seeds), 4, figsize=(11.7,8.3), squeeze=False)
    for row, seed in zip(axes, seeds):
        _, a = inspect(samples[(seed,0)])
        old, _ = inspect_sample(baseline_root/samples[(seed,0)].name)
        images = [a['target'], old['completion'], a['completion']]
        for ax, im, label in zip(row[:3], images, ['Target crop','Previous model','Hybrid']):
            ax.imshow(im[64:192,64:192].clip(0,1), interpolation='nearest');ax.axis('off');ax.set_title(f'{label} / {seed}', fontsize=9)
        error = np.mean(np.abs(a['prediction'] - a['target']),axis=-1)
        error[~a['hidden']] = np.nan
        heat = row[3].imshow(error[64:192,64:192], vmin=0,vmax=.15,cmap='magma',interpolation='nearest')
        row[3].axis('off');row[3].set_title('Hidden RGB absolute error',fontsize=9)
    fig.suptitle('Fixed center crops: fine detail and remaining error',fontsize=17,y=.965)
    fig.subplots_adjust(top=.89,bottom=.14,left=.03,right=.96,wspace=.08,hspace=.22)
    cax=fig.add_axes([.25,.065,.5,.018]);fig.colorbar(heat,cax=cax,orientation='horizontal',label='Mean absolute RGB error (sRGB 0–1); white = observed input')
    pdf.savefig(fig);plt.close(fig)


def render(c, r, samples):
    out=Path(c['pdf']);out.parent.mkdir(parents=True,exist_ok=True)
    with PdfPages(out,metadata={'Title':'burn_gekko: resolving cross-view reconstruction blur','Author':'Local Burn pipeline study','Subject':'Held-out synthetic-room quality, controls, and runtime'}) as pdf:
        p=r['performance'];s=r['statistics']
        page(pdf, 'Cross-view completion: quality qualification', [
            f"Status: {'accepted for this bounded synthetic-room task' if r['accepted'] else 'not qualified'}. {r['visual_review']}",
            f"Fresh test: {r['test_rooms']} disjoint procedural rooms, {r['targets']} target views at 256 × 256; two reference views, 75% target patches hidden. Every target export is independently scored. One training seed and one fixed evaluation mask.",
            f"Hidden RGB MSE: previous {r['baseline_rgb_mse']:.6f}; hybrid {r['hybrid_rgb_mse']:.6f} ({r['relative_rgb_improvement']*100:.1f}% lower). PSNR from pooled MSE: {-10*np.log10(r['baseline_rgb_mse']):.2f} → {-10*np.log10(r['hybrid_rgb_mse']):.2f} dB.",
            f"Full-test hidden edge cosine: {ci(s['edge_cosine'])}; edge energy / target: {ci(s['edge_energy'])}. Brackets are 95% room-bootstrap intervals, not independent-pixel confidence intervals.",
            f"Same-model controls: monocular MSE {r['monocular_rgb_mse']:.6f}; unrelated-reference MSE {r['unrelated_rgb_mse']:.6f}. Hidden-target intervention max |ΔRGB| = {r['input_audit']['hidden_target_intervention_max_abs']:.1g}; reference permutation max |ΔRGB| = {r['input_audit']['reference_permutation_max_abs']:.1g}.",
            f"Metered experiment commands used {r['command_seconds']/60:.1f} of 120 minutes, including rejected GPU trials and evaluation. Compilation, downloads and Python report analysis are outside that ledger. Final adaptation: {p['steps']:,} updates in {p['seconds']/60:.1f} min; warm {p['warm_images_per_second']:.1f} images/s, peak process VRAM {p['peak_process_vram_gib']:.1f} GiB.",
            'The improvement transfers a large pretrained appearance/reconstruction prior. It is not evidence that a small V-JEPA-only decoder can learn the same capability within this budget.'
        ])
        page(pdf, 'What changed and why', c['findings'])
        page(pdf, 'Architecture, supervision and evidence boundaries', [
            'Observed target patches and complete references pass through two frozen encoders: imported V-JEPA 2.1 Base and released Gekko ViT-L. Hidden target patches are removed before encoder attention. A shared residual adapter combines V-JEPA semantics and observed RGB with 1024-dimensional appearance features.',
            'The released 12-block, 768-wide cross-attention decoder runs once per reference. Shared pairwise outputs are averaged, making the reference set permutation invariant. A learned MLP predicts patch mean and scale; standalone RGB is reconstructed without hidden target normalization statistics. The final two decoder blocks, adapter and statistics MLP are adapted on procedural rooms.',
            'Training uses masked RGB labels only: content/statistics reconstruction, raw RGB error, hidden-pixel gradient error/energy, and a separately computed monocular reconstruction. Energy matching is a training-only contrast constraint, never an inference rescaling. Camera, depth, position and geometric visibility are excluded from RGB model inputs and training losses. Dense RI is evaluated as a separate operation.',
            'The official checkpoint contains 712 F32 tensors, consumed exactly once with shape checks. Official PyTorch vs Burn parity passed: maximum relative RMS 2.96e-6 on CPU, 0.00221 on CUDA. This is not bitwise equivalence; CUDA uses the workstation matmul path. Hidden-pixel encoder interventions were exactly zero on both backends.',
            'Released weights: thibautloiseau/gekko-vitl-500k, revision 79fba28dd59ec54fffa0134fae681d88ed084513, CC-BY-NC-SA-4.0. Its 500k-step, global-batch-768 pretraining is external compute and is not included in this workstation budget. Model card: https://huggingface.co/thibautloiseau/gekko-vitl-500k'
        ])
        fig, axes=plt.subplots(2,2,figsize=(11.7,8.3))
        for stage in c['training_stages']:
            path,label=Path(stage['path']),stage['label']
            data=read(path/'report.json');m=[json.loads(l) for l in (path/'metrics.jsonl').read_text().splitlines()]
            axes[0,0].plot([x['step'] for x in data['probes']],[x['mse'] for x in data['probes']],marker='o',label=label)
            window=50;loss=np.convolve([x['loss'] for x in m],np.ones(window)/window,mode='valid')
            axes[0,1].plot(np.arange(len(loss))+window,loss,label=label)
        axes[0,0].set(title='Fixed four-room / 12-view validation probe',xlabel='Stage optimizer step',ylabel='Hidden RGB MSE',yscale='log');axes[0,0].legend()
        axes[0,1].set(title='Training objective (50-step mean)',xlabel='Stage optimizer step',ylabel='Loss (stage objectives differ)');axes[0,1].legend()
        m=[json.loads(l) for l in (Path(c['training_run'])/'metrics.jsonl').read_text().splitlines()]
        axes[1,0].plot([x['step'] for x in m],[1000*x['seconds'] for x in m],linewidth=.65)
        axes[1,0].set(title='Final adaptation: step latency',xlabel='Optimizer step',ylabel='Milliseconds',yscale='log')
        gpu=[json.loads(l) for l in Path(c['training_telemetry']).read_text().splitlines()]
        axes[1,1].plot([g['elapsed_seconds'] for g in gpu],[g.get('process_vram_mib',0)/1024 for g in gpu])
        axes[1,1].set(title='Process VRAM (1 Hz)',xlabel='Wall seconds',ylabel='GiB')
        for ax in axes.flat:ax.grid(alpha=.2)
        fig.suptitle('Convergence and workstation efficiency',fontsize=18);fig.tight_layout(rect=(0,.045,1,.95))
        fig.text(.05,.015,'Stages initialize from previous weights with new optimizers. Probes use existing validation rooms, never fresh-test rooms.',fontsize=9)
        pdf.savefig(fig);plt.close(fig)
        page(pdf, 'Metrics, controls and acceptance gates', [
            f"Paired baseline minus hybrid RGB MSE: {ci(r['paired_differences']['baseline_gain'])}. Unrelated minus correct references: {ci(r['paired_differences']['reference_gain'])}. Monocular minus cross-view: {ci(r['paired_differences']['monocular_gain'])}. All three target views are averaged within each room before 2,000 bootstrap resamples (seed 2902).",
            f"On {r['matched_export_rooms']} exactly matched target-zero exports, previous/hybrid edge cosine is {r['matched_export_baseline_edge_cosine']:.3f}/{r['matched_export_hybrid_edge_cosine']:.3f}; edge-energy ratio is {r['matched_export_baseline_edge_energy']:.3f}/{r['matched_export_hybrid_edge_energy']:.3f}. Full-test hybrid metrics above include every target, so their sample population differs from this matched subset.",
            f"Gradient metrics use adjacent pixel pairs only when both pixels are hidden. Excluding every 16px patch seam, hybrid cosine is {r['statistics']['interior_edge_cosine']['mean']:.3f} and energy ratio is {r['statistics']['interior_edge_energy']['mean']:.3f}. At patch boundaries the energy ratio is {r['statistics']['seam_edge_energy']['mean']:.3f}, showing residual seam artifacts. Energy is always evaluated alongside alignment. Metrics use unclipped predictions; displays clip only to RGB [0,1].",
            f"Numerical gates and additional artifact controls: {', '.join(k + '=' + ('PASS' if v else 'FAIL') for k,v in r['gates'].items())}. Visual qualification is a separate manual review. Separate CUDA evaluation processes differ in monocular MSE by at most {r['max_monocular_control_difference']:.2g}; this is far below the measured reference gain. The monocular API takes no reference input.",
            f"Co-visibility diagnostic: pooled RI AUROC {r['covisibility']['auroc']:.3f}, AP {r['covisibility']['average_precision']:.3f}, constant-score prevalence baseline {r['covisibility_ap_prevalence_baseline']:.3f}. Adapter enabled: {r['covisibility_adapter_enabled']}. Dense-target RI is separate from masked RGB completion; geometric labels are loaded after the forward pass. This probe does not establish calibrated probabilities or real-data transfer."
        ])
        page(pdf, 'Visibility strata and runtime detail', [
            f"Of labeled hidden pixels, the mean per-room fraction visible in at least one reference is {r['visibility_strata']['reference_visible_fraction']['mean']*100:.1f}%. Geometry is an evaluation annotation, never a reconstruction input.",
            f"Hidden RGB MSE on reference-visible pixels: {ci(r['visibility_strata']['reference_visible_mse'])}. On pixels absent from both references: {ci(r['visibility_strata']['reference_absent_mse'])}. Conditional errors are averaged per target then per room; unknown labels are excluded. Appearance changes and thin structures can remain hard even when geometry calls a pixel visible.",
            *[f"{t['label']}: {t['steps']:,} steps, {t['seconds']/60:.2f} minutes; warm median/p90 {t['warm_step_median_ms']:.1f}/{t['warm_step_p90_ms']:.1f} ms, {t['warm_images_per_second']:.1f} images/s; end-to-end {t['end_to_end_images_per_second']:.1f} images/s, peak process VRAM {t['peak_process_vram_gib']:.1f} GiB." for t in r['training_stages']],
            f"Final-stage decoder first-block Δ = {p['first_decoder_block_delta']}; last-block Δ = {p['last_decoder_block_delta']}; output-head Δ = {p['frozen_head_delta']}. Timing includes real CUDA work and sync. Warm metrics omit the first 100 steps. Device utilization/power include desktop activity; process VRAM is separate. Frozen reference features are cached in GPU memory, masked targets are encoded online."
        ])
        roots=Path(c['baseline_samples']);choices=sorted(roots.glob('room-*-view-*'))
        selected=[choices[i] for i in np.linspace(0,len(choices)-1,4,dtype=int)]
        seeds=[read(x/'sample.json')['room_seed'] for x in selected]
        for seed in seeds:completion_panels(pdf,roots,samples,seed)
        detail_panel(pdf,samples,roots,seeds[:3])
        # Predetermined worst/median selection by test RGB error is a diagnostic,
        # never a model-selection input. Include a failure, not only good images.
        ranked=sorted(r['samples'],key=lambda x:x['mse'])
        selected_failures=[ranked[-1], ranked[len(ranked)//2]]
        fig,axes=plt.subplots(2,4,figsize=(11.7,8.3))
        for row,m in zip(axes,selected_failures):
            _,a=inspect(samples[(m['room_seed'],m['target_view'])])
            for ax,im,label in zip(row,[a['target'],a['masked'],a['references'][0],a['completion']],['Target','Observed input','Reference 1','Hybrid']):
                ax.imshow(im.clip(0,1),interpolation='nearest');ax.axis('off');ax.set_title(label)
            row[0].text(0,-.10,f"Room {m['room_seed']}, view {m['target_view']}; PSNR {m['psnr']:.2f} dB, edge cosine {m['edge_cosine']:.3f}",transform=row[0].transAxes,fontsize=10)
        fig.suptitle('Fresh-test worst and median targets by RGB error',fontsize=17)
        fig.subplots_adjust(top=.90,bottom=.09,hspace=.2,wspace=.07,left=.025,right=.975)
        pdf.savefig(fig);plt.close(fig)
        covis_dir=Path(c['covisibility']).parent
        available=sorted(covis_dir.glob('room-*-view-*'))
        if available:
            fig,axes=plt.subplots(2,4,figsize=(11.7,8.3))
            for row,path in zip(axes,[available[0],available[len(available)//2]]):
                seed=int(path.name.split('-')[1]);_,a=inspect(samples[(seed,0)]);h,w=a['hidden'].shape
                ri=np.fromfile(path/'ri.f32',dtype='<f4').reshape(h,w)
                truth=np.fromfile(path/'visibility.u8',dtype='u1').reshape(h,w)
                row[0].imshow(a['target'].clip(0,1));row[0].set_title(f'Target / {seed}',fontsize=10)
                row[1].imshow(a['references'][0].clip(0,1));row[1].set_title('Reference 1 of 2',fontsize=10)
                im=row[2].imshow(ri,vmin=0,vmax=1,cmap='viridis');row[2].set_title('Max predicted pairwise RI',fontsize=10)
                labels=np.where(truth==255,np.nan,truth.astype(float));row[3].imshow(labels,vmin=0,vmax=1,cmap='viridis');row[3].set_title('Geometric visibility union',fontsize=10)
                for ax in row:ax.axis('off')
            fig.suptitle('Co-visibility annotations: a separate dense-image diagnostic',fontsize=17)
            fig.subplots_adjust(top=.9,bottom=.14,left=.025,right=.975,wspace=.09,hspace=.16)
            cax=fig.add_axes([.25,.065,.5,.018]);fig.colorbar(im,cax=cax,orientation='horizontal',label='RI display range 0–1; labels: 0 not visible, 1 visible, white unknown')
            pdf.savefig(fig);plt.close(fig)
        page(pdf, 'Limits and paper-ready claims', c['limitations'])
        page(pdf, 'Reproduction and artifact map', [
            'All generation/training/evaluation command plans, stdout and one-second GPU telemetry are preserved under .data/pilot-04. TOML stores input configurations; JSON stores measured results. Existing pilot-03 results are unchanged.',
            f"Accepted recipe/config: {c['selected_config']}. Selected checkpoint: {c['selected_checkpoint']}. SHA256: {hashlib.sha256(Path(c['selected_checkpoint']).read_bytes()).hexdigest()}.",
            f"Fresh dataset: {c['dataset']}. Training cache: {c['prior_datasets'][0]}. Published generator packages: bevy_zeroverse 0.23.0 and bevy_zeroverse_burn 0.6.0 (registry rechecked on 28 September 2026).",
            'Source/binary snapshots and receipts identify each runtime. Re-run with cargo build --profile pilot --features cuda --bins --examples, then the saved tools/study/run_study.py plans using new output directories. Checkpoints initialize subsequent stages with a new optimizer; hybrid-stage optimizer resumption is not implemented.',
            f"Report source: tools/legacy/hybrid_report.py; config: {c['_config']}; metrics: .data/pilot-04/quality-summary.json. This PDF: {c['pdf']}. Frozen report inputs are raw float32 predictions, masks, target RGB and separately exported reference/visibility annotations.",
            'Validation: workspace tests, strict CUDA-feature Clippy, imported encoder checksums, official model parity, sampler gradients, loss mask-boundary isolation, cache disjointness, hidden-pixel and reference-order interventions. Measured decoder freeze/update deltas are in the training report. See docs/studies/pilot-04.md for exact commands and remaining research work.'
        ])
    print(json.dumps({k:r[k] for k in ['accepted','hybrid_rgb_mse','relative_rgb_improvement','test_rooms','targets','command_seconds']}))
    print(out)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);args=p.parse_args()
    c=tomllib.loads(args.config.read_text());c['_config']=str(args.config)
    assert Path(c['study']).resolve().is_relative_to(Path('.data').resolve())
    assert Path(c['pdf']).resolve().is_relative_to(Path('.data').resolve())
    result,samples=collect(c);render(c,result,samples)


if __name__=='__main__':main()
