#!/usr/bin/env python3
"""Render audited completed-arm comparisons and clearly marked live probes.

This progress report never certifies a model from validation or synthetic views.
Final qualification additionally requires a selected checkpoint, fresh reserved
rooms, input controls, co-visibility evaluation, and recorded visual review.
"""
import argparse
import datetime
import json
from pathlib import Path
import tomllib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from validation_comparison import compare
from long_study_report import table_page, text_page, sample_page, comparison_page
from probe_quality import collect
from training_throughput import summarize


def live_curve(pdf, label, root, probes):
    rows = [json.loads(s) for s in (root/'metrics.jsonl').read_text().splitlines()]
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
    fig.suptitle(label + ': completed probes', y=.95, fontsize=17)
    axes[0, 0].plot([r['step'] for r in rows], [r['loss'] for r in rows], lw=.5)
    axes[0, 0].set_title('Training objective; individual batches')
    for split, name in [('training_probe', 'Captured training rooms'), ('validation_probe', 'Captured validation rooms'), ('synthetic_probe', 'Derived image copying diagnostic')]:
        selected = [r for r in probes if r['split'] == split]
        if not selected:
            continue
        x = [r['step'] for r in selected]
        style = '--' if split == 'synthetic_probe' else '-'
        axes[0, 1].plot(x, [r['all_target_mse'] for r in selected], style, marker='.', label=name)
        axes[1, 0].plot(x, [r['exported_metrics']['edge_cosine'] for r in selected], style, marker='.', label=name)
        axes[1, 1].plot(x, [r['exported_metrics']['interior_edge_energy'] for r in selected], style, marker='.', label=name)
    axes[0, 1].set_title('Probe MSE, all three target views')
    axes[1, 0].set_title('Aligned edge cosine, exported view zero')
    axes[1, 1].set_title('Within-patch detail energy, exported view zero')
    axes[1, 0].axhline(.4, color='gray', ls=':')
    axes[1, 1].axhline(.5, color='gray', ls=':')
    for ax in axes.flat:
        ax.grid(alpha=.2); ax.set_xlabel('Optimizer update in this leg')
    for ax in [axes[0,1],axes[1,0],axes[1,1]]:
        ax.legend(fontsize=7)
    fig.tight_layout(rect=[0,.035,1,.91])
    fig.text(.04,.018,'Synthetic probes are transformed images, not held-out 3D reconstruction evidence. No postprocessing.',fontsize=8)
    pdf.savefig(fig);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();c=tomllib.loads(args.config.read_text())
    result=compare(c)
    result['generated_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    result['status']='in_progress_unqualified'
    result['blur_resolved']=False
    result['budget']=json.loads((Path(c['study'])/'budget.json').read_text())
    result['runtime']={}
    result['probes']={}
    for arm in c['arms']:
        if arm.get('telemetry'):
            result['runtime'][arm['name']]=summarize(arm)
        root=Path(arm['run'])
        provenance=json.loads((root/'provenance.json').read_text())
        assert not provenance['noncommercial_weight_dependencies'] and provenance['teacher'] is None
    for arm in c.get('live_probes',[]):
        result['probes'][arm['name']]=collect(Path(arm['run']))
    output=Path(c['output']);output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    with PdfPages(output) as pdf:
        text_page(pdf,'Pilot07: reconstruction quality investigation',[
            'Status: work in progress. Blur is not resolved. This report separates completed real-room validation comparisons from a derived-image copying diagnostic. No new model has been qualified.',
            c['summary'],
            'All models use our own reconstruction weights and the audited MIT V-JEPA 2.1 encoder package. The RGB correspondence encoder starts from random weights. No noncommercial pretrained model, teacher, renderer geometry input, or renderer correspondence training label is used.',
            'The regenerated capture uses bevy_zeroverse 0.25.0 and bevy_zeroverse_burn 0.8.0: 8,192 training rooms and 128 validation rooms, three 256px views per room. All 384 validation targets are included below; room-bootstrap intervals use 2,000 replicates.',
            'Quality checks are unchanged: aligned edges, adequate within-patch detail, cross-view advantage, related-reference advantage, zero hidden-target dependence, strict reference-order stability and visual correctness. Validation progress and sharpness alone do not certify a fix.',
            'Pilot06 is preserved as a completed, unqualified study. Pilot07 has its own 43,200-second command ceiling. Fresh test seeds are reserved separately; the previously inspected test set is not used to select changes.',
            f"Generated {result['generated_utc']}. Completed Pilot07 command time: {result['budget']['command_seconds'] / 3600:.3f} hours. An active command, if any, is additional and remains bounded by the shared ledger.",
        ],study_label='Pilot07')
        names=[a['name'] for a in c['arms']]
        keys=['mse','psnr','edge_cosine','edge_energy','interior_edge_cosine','interior_edge_energy','monocular_gain','transported_mse','generator_weight']
        values=[]
        for key in keys:
            row=[key.replace('_',' ')]
            for name in names:
                m=result['arms'][name]['metrics'][key]
                row.append(f"{m['mean']:.5f}\n[{m['low']:.5f}, {m['high']:.5f}]")
            values.append(row)
        table_page(pdf,'Captured-room validation: means and 95% room intervals',['Metric',*names],values,study_label='Pilot07')
        if result['runtime']:
            table_page(pdf,'Measured training cost',['Arm','Updates','Median warm step (s)','Warm targets/s','Peak process VRAM (GiB)'],[
                [name,r['executed_updates'],f"{r['phases'][-1]['median_seconds']:.3f}",f"{r['phases'][-1]['targets_per_second']:.2f}",f"{r['sampled_peak_process_vram_mib']/1024:.2f}"]
                for name,r in result['runtime'].items()],study_label='Pilot07')
        for arm in c.get('live_probes',[]):
            live_curve(pdf,arm['name'],Path(arm['run']),result['probes'][arm['name']])
        focus=c['review_arm'];arm=result['arms'][focus];ordered=sorted(arm['rows'],key=lambda r:r['mse'])
        selected=[('First exported target',min(arm['rows'],key=lambda r:(r['room_seed'],r['target_view']))),('Median MSE target',ordered[len(ordered)//2]),('Worst MSE target',ordered[-1])]
        for label,row in selected:
            key=(row['room_seed'],row['target_view']);suffix=f'room-{key[0]}-view-{key[1]}'
            entries=[(name,Path(result['arms'][name]['run'])/'validation'/suffix) for name in names]
            comparison_page(pdf,key,entries)
            sample_page(pdf,Path(arm['run'])/'validation'/suffix,label+' — '+focus)
        text_page(pdf,'Evidence boundaries and reproducibility',[
            'All displayed completions combine unchanged observed target pixels with raw model predictions in hidden patches. Scores use hidden pixels only. No sharpening or image enhancement is applied. Detail crops are selected by target edge energy; first/median/worst examples follow deterministic rules.',
            'The projective copying curriculum constructs two transformed training references from a training image, records each transformation, and replays augmentation by absolute optimizer step. Derived validation probes are labeled synthetic and never count as real 3D view completion results.',
            'The explicit matcher compares shared CNN features at candidate reference positions, using the learned flow as a spatial prior. It is motivated by explicit matching research, not a reproduction of GMFlow. Paper: https://arxiv.org/abs/2111.13680. No external code or weights were imported for it.',
            'Source snapshots, immutable binaries, TOML configs, checkpoints, optimizer states, input interventions and one-second telemetry are stored under .data/pilot-07 and .data/runs. The companion JSON retains complete per-target metrics and paired comparisons.',
        ],study_label='Pilot07')
    print(output)


if __name__=='__main__':
    main()
