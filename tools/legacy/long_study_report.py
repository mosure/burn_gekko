#!/usr/bin/env python3
"""Audit a TOML-described longer study and render unenhanced annotated RGB evidence."""
import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path
import textwrap
import tomllib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
import numpy as np

from e2e_report import ci, exports, geometry_conditioned, performance
from pilot_report_data import read
from probe_quality import collect as collect_probes
from transport_diagnostics import inspect
from training_throughput import summarize as summarize_throughput
from checkpoint_lineage import audit as audit_lineage


def sampling_audit(root, config, manifest):
    """Check actual optimizer inputs against the configured training split."""
    seeds = [s['seed'] for s in manifest['scenes'] if s['split'] == 'train'][:config['train_rooms']]
    allowed = {(seed, view) for seed in seeds for view in range(manifest['config']['cameras'])}
    counts = collections.Counter()
    updates = 0
    for line in (root / 'metrics.jsonl').read_text().splitlines():
        row = json.loads(line)
        samples = [tuple(sample) for sample in row['samples']]
        assert len(samples) == config['batch_size']
        assert all(sample in allowed for sample in samples), 'nontraining optimizer input'
        counts.update(samples)
        updates += 1
    assert counts
    exposures = sum(counts.values())
    return dict(executed_updates=updates, target_exposures=exposures,
                unique_rooms=len({seed for seed, _ in counts}),
                unique_target_views=len(counts), eligible_target_views=len(allowed),
                minimum_exposures_per_target=min(counts[target] for target in allowed),
                maximum_exposures_per_target=max(counts.values()),
                target_view_epochs=exposures / len(allowed), nontraining_inputs=0)


def visibility_detail(paths, diagnostics):
    """Measure aligned detail where references can or cannot observe the target."""
    rows = []
    for (seed, view), path in paths.items():
        _, arrays = inspect(path)
        labels = np.fromfile(Path(diagnostics).parent / path.name / 'visibility.u8',
                             dtype='u1').reshape(arrays['hidden'].shape)
        row = dict(room_seed=seed, target_view=view)
        for label, name in [(1, 'reference_visible'), (0, 'reference_absent')]:
            support = arrays['hidden'] & (labels == label)
            truth, predicted = [], []
            for axis in [0, 1]:
                left = np.take(support, range(support.shape[axis] - 1), axis=axis)
                right = np.take(support, range(1, support.shape[axis]), axis=axis)
                valid = left & right
                truth.append(np.diff(arrays['target'], axis=axis)[valid].ravel())
                predicted.append(np.diff(arrays['prediction'], axis=axis)[valid].ravel())
            truth, predicted = np.concatenate(truth), np.concatenate(predicted)
            energy = float(np.dot(truth, truth))
            estimate = float(np.dot(predicted, predicted))
            row[name + '_edge_cosine'] = (float(np.dot(truth, predicted)) / np.sqrt(energy * estimate)
                                         if energy > 1e-12 and estimate > 1e-12 else float('nan'))
            row[name + '_edge_energy'] = estimate / energy if energy > 1e-12 else float('nan')
        rows.append(row)
    return {key: ci(rows, key) for key in rows[0] if key not in ['room_seed', 'target_view']}


def audit(config):
    study = Path(config['study'])
    budget = read(study / 'budget.json')
    assert not budget.get('running'), 'study leg still active'
    ledger_paths = sorted(study.glob('*/ledger.json'))
    ledgers = [read(p) for p in ledger_paths]
    seconds = sum(p.get('command_seconds', 0) for p in ledgers)
    assert abs(seconds - budget['command_seconds']) < .01, 'unaccounted experiment leg'
    assert seconds <= budget['ceiling_seconds'] <= 43200
    result = dict(command_seconds=seconds, ceiling_seconds=budget['ceiling_seconds'],
                  arms={}, visual_review=config.get('visual_review', 'Pending'))
    result['budget_commands'] = [dict(leg=path.parent.name, name=command['name'],
        status=command['status'], elapsed_seconds=command['elapsed_seconds'])
        for path, ledger in zip(ledger_paths, ledgers) for command in ledger['commands']]
    paths = {}
    for arm in config['arms']:
        root = Path(arm['run'])
        resolved = tomllib.loads((root / 'config.toml').read_text())
        provenance = read(root / 'provenance.json')
        checkpoint = Path(arm.get('checkpoint', str(root / 'final')))
        metadata = read(checkpoint / 'metadata.json')
        assert not provenance['noncommercial_weight_dependencies']
        assert not metadata['noncommercial_weight_dependencies']
        assert provenance['teacher'] is None and provenance['cached_feature_bytes'] == 0
        for key in ['identity', 'dataset_id', 'encoder_id']:
            assert provenance[key] == metadata[key], f'checkpoint {key} differs from run'
        with (checkpoint / 'model.mpk').open('rb') as stream:
            assert hashlib.file_digest(stream, 'sha256').hexdigest() == metadata['model_sha256']
        parent = metadata.get('warm_start')
        lineage_seen = set()
        while parent:
            parent_path = Path(parent['checkpoint'])
            assert str(parent_path.resolve()) not in lineage_seen, 'cyclic checkpoint lineage'
            lineage_seen.add(str(parent_path.resolve()))
            assert hashlib.sha256((parent_path/'metadata.json').read_bytes()).hexdigest() == parent['source_metadata_sha256']
            parent_metadata = read(parent_path/'metadata.json')
            assert not parent_metadata['noncommercial_weight_dependencies']
            assert parent_metadata['model_sha256'] == parent['model_sha256']
            with (parent_path/'model.mpk').open('rb') as stream:
                assert hashlib.file_digest(stream, 'sha256').hexdigest() == parent['model_sha256']
            assert parent['optimizer_reset']
            parent = parent_metadata.get('warm_start')
        entry = dict(label=arm['label'], config=resolved, provenance=provenance,
                     checkpoint=metadata, runtime=performance(arm),
                     diagnostic_only=arm.get('diagnostic_only', False), probes=collect_probes(root))
        manifest = read(Path(resolved['dataset']) / 'manifest.json')
        entry['sampling'] = sampling_audit(root, resolved, manifest)
        entry['throughput'] = summarize_throughput(arm)
        expected = lambda split: {(s['seed'], v) for s in manifest['scenes'] if s['split'] == split
                                  for v in range(manifest['config']['cameras'])}
        splits = {'validation': Path(arm['validation']) / 'samples' if arm.get('validation') else root / 'validation'}
        if arm.get('validation'):
            validation_provenance = read(Path(arm['validation']) / 'provenance.json')
            assert validation_provenance['evaluation_split'] == 'validation'
            assert Path(validation_provenance['resume']).resolve() == checkpoint.resolve()
            assert validation_provenance['dataset_id'] == metadata['dataset_id']
            assert validation_provenance['identity'] == metadata['identity']
        else:
            assert checkpoint.resolve() == (root / 'final').resolve(), 'periodic checkpoint needs its own validation export'
        if arm.get('training_samples'):
            splits['training'] = Path(arm['training_samples'])
        if arm.get('test'):
            assert not entry['diagnostic_only']
            splits['test'] = Path(arm['test']) / 'samples'
            test_provenance = read(Path(arm['test']) / 'provenance.json')
            assert test_provenance['evaluation_split'] == 'test'
            assert Path(test_provenance['resume']).resolve() == checkpoint.resolve()
            assert test_provenance['dataset_id'] == metadata['dataset_id']
            assert test_provenance['identity'] == metadata['identity']
            assert not test_provenance['noncommercial_weight_dependencies'] and test_provenance['teacher'] is None
        for split, source in splits.items():
            measured, sample_paths = exports(source)
            if resolved.get('appearance_transport'):
                for key in ['generator_weight', 'generated_mse', 'transported_mse', 'mean_flow_pixels']:
                    assert all(key in row for row in measured['rows']), 'appearance exports missing'
                    measured['metrics'][key] = ci(measured['rows'], key)
            if resolved.get('transport_coarse_smoothness_weight', 0) > 0:
                assert all('coarse_flow_curvature' in row for row in measured['rows'])
                measured['metrics']['coarse_flow_curvature'] = ci(measured['rows'], 'coarse_flow_curvature')
            if split != 'training':
                assert set(sample_paths) == expected(split), f'incomplete {split} exports'
            else:
                assert read(source / 'evaluation.json')['diagnostic_only']
            control = (Path(arm['test_unrelated']) / 'samples' if split == 'test' and arm.get('test_unrelated')
                       else Path(arm['validation_unrelated']) / 'samples' if split == 'validation' and arm.get('validation_unrelated')
                       else root / 'unrelated' if split == 'validation' and not arm.get('validation') else None)
            if control:
                if control.name == 'samples':
                    control_provenance = read(control.parent / 'provenance.json')
                    assert control_provenance['evaluation_split'] == split
                    assert Path(control_provenance['resume']).resolve() == checkpoint.resolve()
                    for key in ['identity', 'dataset_id', 'encoder_id']:
                        assert control_provenance[key] == metadata[key]
                    assert not control_provenance['noncommercial_weight_dependencies']
                    assert control_provenance['teacher'] is None
                unrelated = read(control / 'evaluation.json')
                assert unrelated['unrelated'] is True
                wrong = {(r['room_seed'], r['target_view']): r for r in unrelated['targets']}
                assert set(wrong) == set(sample_paths)
                for row in measured['rows']:
                    other = wrong[(row['room_seed'], row['target_view'])]
                    row['reference_gain'] = other['hidden_rgb_mse'] - row['mse']
                    row['monocular_control_difference'] = abs(other['monocular_hidden_rgb_mse'] - row['monocular_mse'])
                measured['metrics']['reference_gain'] = ci(measured['rows'], 'reference_gain')
            entry[split] = measured
            paths[arm['name'], split] = sample_paths
        result['arms'][arm['name']] = entry
    selected_name = config['selected']
    selected = result['arms'][selected_name]
    assert not selected['diagnostic_only'] and 'test' in selected
    selected_arm = next(arm for arm in config['arms'] if arm['name'] == selected_name)
    result['lineage'] = audit_lineage(Path(selected_arm.get('checkpoint', str(Path(selected_arm['run']) / 'final'))))
    selection = read(config['selection'])
    assert selection['model_sha256'] == selected['checkpoint']['model_sha256']
    assert selection['uses_test_for_selection'] is False
    if config.get('reserved_test_ledger'):
        test_started = read(config['reserved_test_ledger'])['started_utc']
        assert datetime.datetime.fromisoformat(selection['selected_utc']) < datetime.datetime.fromisoformat(test_started)
    result['selected'] = selected_name
    diagnostics = read(config['diagnostics'])
    assert diagnostics['split'] == 'test'
    assert diagnostics['checkpoint_sha256'] == selected['checkpoint']['model_sha256']
    assert diagnostics['dataset_id'] == selected['checkpoint']['dataset_id']
    assert {(r['room_seed'], r['target_view']) for r in diagnostics['targets']} == set(paths[selected_name, 'test'])
    result['diagnostics'] = diagnostics
    result['geometry_conditioned'] = geometry_conditioned(paths[selected_name, 'test'], config['diagnostics'])
    result['geometry_conditioned'].update(visibility_detail(paths[selected_name, 'test'], config['diagnostics']))
    if selected['config'].get('appearance_transport'):
        flow_audit = read(config['appearance_geometry'])
        assert flow_audit['dataset_id'] == selected['checkpoint']['dataset_id']
        assert flow_audit['split'] == 'test'
        assert {(row['room_seed'], row['target_view']) for row in flow_audit['rows']} == set(paths[selected_name, 'test'])
        exported_source = next(iter(paths[selected_name, 'test'].values())).parent
        assert Path(flow_audit['samples']).resolve() == exported_source.resolve()
        result['appearance_geometry'] = flow_audit
    rankings = [dict(room_seed=r['room_seed'], auroc=r['covisibility']['auroc'],
                     average_precision=r['covisibility']['average_precision'])
                for r in diagnostics['targets'] if r['covisibility']['auroc'] is not None]
    result['ranking_intervals'] = {k: ci(rankings, k) for k in ['auroc', 'average_precision']}
    m = selected['test']['metrics']
    controls = diagnostics['input_audit']
    result['gates'] = dict(
        no_nc_weights=True,
        encoder_updated=selected['runtime']['first_encoder_delta'] > 0 and selected['runtime']['stem_delta'] > 0,
        appearance_head_updated=(not selected['config'].get('appearance_transport') or
            read(Path(next(arm['run'] for arm in config['arms'] if arm['name'] == selected_name))/'report.json').get('appearance_head_max_abs_delta', 0) > 0),
        aligned_edges=m['edge_cosine']['mean'] >= .4,
        gradient_energy=.5 <= m['edge_energy']['mean'] <= 1.5,
        within_patch_detail=m['interior_edge_cosine']['mean'] >= .4 and m['interior_edge_energy']['mean'] >= .5,
        monocular_advantage=m['monocular_gain']['low'] > 0 and 1 - m['mse']['mean'] / m['monocular_mse']['mean'] >= .1,
        reference_advantage=m.get('reference_gain', {'low': 0})['low'] > 0,
        hidden_input_independence=controls['hidden_target_intervention_max_abs'] == 0,
        reference_permutation=controls['reference_permutation_max_abs'] < 1e-5,
        visual_review=config.get('visual_review_passed', False),
    )
    result['quality_qualified'] = all(result['gates'].values())
    result['registry'] = read(study / 'registry-versions.json')
    result['inventory'] = read(study / 'dataset-summary.json')
    assert result['inventory']['dataset_id'] == selected['checkpoint']['dataset_id']
    result['camera_policy'] = read(study / 'validation-camera-policy.json')
    assert result['camera_policy']['dataset_id'] == selected['checkpoint']['dataset_id']
    assert result['camera_policy']['split'] == 'validation' and result['camera_policy']['rooms'] == 128
    geometry = read(study / 'main-geometry-audit/report.json')
    assert geometry['dataset_id'] == selected['checkpoint']['dataset_id']
    result['geometry_oracle'] = dict(rooms=len(geometry['rooms']),
        mean_covered_fraction=float(np.mean([r['covered_fraction'] for r in geometry['rooms']])),
        mean_oracle_mse=geometry['mean_oracle_mse'], mean_unwarped_mse=geometry['mean_unwarped_mse'])
    return result, paths


def text_page(pdf, title, paragraphs, study_label="Pilot 06"):
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(.06, .94, title, size=19, weight='bold')
    y = .87
    for paragraph in paragraphs:
        lines = textwrap.wrap(str(paragraph), 119)
        height = .028 * len(lines) + .024
        if y - height < .07:
            pdf.savefig(fig); plt.close(fig)
            fig = plt.figure(figsize=(11.7, 8.3)); y = .87
            fig.text(.06, .94, title + ' (continued)', size=19, weight='bold')
        fig.text(.06, y, '\n'.join(lines), va='top', size=11)
        y -= height
    fig.text(.06, .025, f'burn_gekko | {study_label} | raw RGB, no sharpening or image enhancement', size=8, color='.4')
    pdf.savefig(fig); plt.close(fig)


def fmt(value):
    if value['mean'] is None:
        return 'undefined'
    return f"{value['mean']:.5f} [{value['low']:.5f}, {value['high']:.5f}]"


def table_page(pdf, title, columns, rows, study_label="Pilot 06"):
    """Size columns from their content so run names and intervals remain legible."""
    if len(rows) > 16:
        for offset in range(0, len(rows), 16):
            table_page(pdf, title + f' ({offset // 16 + 1})', columns, rows[offset:offset+16], study_label)
        return
    strings = [[str(value) for value in row] for row in rows]
    weights = np.array([min(54, max(9, max(len(row[index]) for row in [columns, *strings])))
                        for index in range(len(columns))], dtype=float)
    widths = weights / weights.sum()
    wrap = lambda row: ['\n'.join(textwrap.wrap(value, max(10, int(width * 140))))
                        for value, width in zip(row, widths)]
    strings = [wrap(row) for row in strings]
    headings = wrap(columns)
    fig = plt.figure(figsize=(11.7, 8.3))
    fig.text(.055, .95, title, size=17 if len(title) > 65 else 19, weight='bold', va='top')
    ax = fig.add_axes([.045, .07, .91, .80]); ax.axis('off')
    table = ax.table(cellText=strings, colLabels=headings, colWidths=widths,
                     loc='center', cellLoc='left', colLoc='left')
    table.auto_set_font_size(False); table.set_fontsize(10)
    line_counts = [max(value.count('\n') + 1 for value in row) for row in [headings, *strings]]
    heights = np.array([.046 + .028 * (lines-1) for lines in line_counts])
    if heights.sum() > .97:
        heights *= .97 / heights.sum()
    for (row, column), cell in table.get_celld().items():
        cell.set_height(heights[row])
        if row == 0:
            cell.set_facecolor('#d7e5f0'); cell.set_text_props(weight='bold')
    fig.text(.055, .025, f'burn_gekko | {study_label} | estimates from raw exported targets', size=8, color='.4')
    pdf.savefig(fig); plt.close(fig)


def sample_page(pdf, path, label, image_output=None):
    row, a = inspect(path)
    mono = np.fromfile(path / 'monocular.f32', dtype='<f4').reshape(a['target'].shape)
    mono[~a['hidden']] = a['target'][~a['hidden']]
    fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3))
    fig.suptitle(f"{label}: room {row['room_seed']}, view {row['target_view']}\n"
                 f"Hidden PSNR {row['psnr']:.2f} dB; edge cosine {row['edge_cosine']:.3f}; energy {row['edge_energy']:.3f}", size=14, y=.94)
    for ax, im, title in zip(axes.flat, [a['target'], a['masked'], a['references'][0], a['references'][1], a['completion'], mono],
                            ['Target', 'Observed input', 'Reference 1', 'Reference 2', 'Cross-view completion', 'Monocular completion']):
        ax.imshow(im.clip(0, 1), interpolation='nearest'); ax.set_title(title, size=11); ax.axis('off')
    fig.subplots_adjust(left=.025, right=.975, bottom=.035, top=.85,
                        wspace=.06, hspace=.18)
    pdf.savefig(fig)
    if image_output:
        fig.savefig(image_output, dpi=160)
    plt.close(fig)
    target = a['target']; h, w = target.shape[:2]
    edges = np.zeros((h, w))
    edges[:-1] += np.mean(np.diff(target, axis=0)**2, axis=2)
    edges[:, :-1] += np.mean(np.diff(target, axis=1)**2, axis=2)
    scores = (edges * a['hidden']).reshape(h//16, 16, w//16, 16).sum(axis=(1, 3))
    py, px = np.unravel_index(np.argmax(scores), scores.shape)
    y, x = min(max(py*16-24, 0), h-64), min(max(px*16-24, 0), w-64)
    fig, axes = plt.subplots(1, 4, figsize=(11.7, 4))
    fig.suptitle(f"{label}: target-selected detail, room {row['room_seed']}", size=14, y=.94)
    axes[0].imshow(target.clip(0, 1)); axes[0].add_patch(Rectangle((x, y), 64, 64, fill=False, color='red'))
    axes[0].set_title('Crop location')
    error = np.mean(abs(a['prediction']-target), axis=2); error[~a['hidden']] = np.nan
    im = axes[1].imshow(error, cmap='magma', vmin=0, vmax=.2); axes[1].set_title('Hidden absolute RGB error')
    fig.colorbar(im, ax=axes[1], fraction=.046)
    axes[2].imshow(target[y:y+64, x:x+64].clip(0, 1), interpolation='nearest'); axes[2].set_title('Target detail')
    axes[3].imshow(a['completion'][y:y+64, x:x+64].clip(0, 1), interpolation='nearest'); axes[3].set_title('Completion detail')
    for ax in axes: ax.axis('off')
    fig.text(.05, .02, '64px crop centered near maximum target-edge energy in a hidden patch. Fixed error scale; observed pixels are white.', size=9)
    fig.tight_layout(rect=[0, .06, 1, 1]); pdf.savefig(fig); plt.close(fig)


def curve_page(pdf, name, arm, root):
    steps = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
    probes = arm['probes']
    fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3)); fig.suptitle(name + ': optimization and detail', size=17, y=.94)
    stride = max(1, len(steps)//2000)
    shown = steps[::stride]
    axes[0, 0].plot([s['step'] for s in shown], [s['loss'] for s in shown], lw=.6)
    axes[0, 0].set_yscale('log'); axes[0, 0].set_title('Training objective (subsampled display)')
    for split, label in [('training_probe', 'Training probe'), ('validation_probe', 'Validation probe')]:
        rows = [p for p in probes if p['split'] == split]
        axes[0, 1].plot([p['step'] for p in rows], [p['all_target_mse'] for p in rows], marker='.', label=label)
        axes[1, 0].plot([p['step'] for p in rows], [p['exported_metrics']['edge_cosine'] for p in rows], marker='.', label=label)
        axes[1, 1].plot([p['step'] for p in rows], [p['exported_metrics']['interior_edge_energy'] for p in rows], marker='.', label=label)
    axes[0, 1].set_yscale('log'); axes[0, 1].set_title('Fixed probe MSE, all target views')
    axes[1, 0].set_title('Aligned edge cosine, exported view zero'); axes[1, 0].axhline(.4, ls='--', c='gray')
    axes[1, 1].set_title('Within-patch gradient energy ratio'); axes[1, 1].axhline(.5, ls='--', c='gray')
    for ax in axes.flat: ax.set_xlabel('Absolute optimizer update'); ax.grid(alpha=.2)
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]: ax.legend(fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, .91]); pdf.savefig(fig); plt.close(fig)


def comparison_page(pdf, key, entries):
    """Compare the same validation target and mask across the recorded arms."""
    loaded = [(label, *inspect(path)) for label, path in entries]
    target = loaded[0][2]['target']
    hidden = loaded[0][2]['hidden']
    for _, _, arrays in loaded[1:]:
        np.testing.assert_array_equal(arrays['target'], target)
        np.testing.assert_array_equal(arrays['hidden'], hidden)
    fig, axes = plt.subplots(1, len(entries) + 2, figsize=(11.7, 4.6))
    fig.suptitle(f'Matched validation comparison: room {key[0]}, view {key[1]}', size=15, y=.94)
    images = [target, loaded[0][2]['masked']] + [a['completion'] for _, _, a in loaded]
    labels = ['Target', 'Observed input'] + [label for label, _, _ in loaded]
    for ax, im, label in zip(axes, images, labels):
        ax.imshow(im.clip(0, 1), interpolation='nearest')
        ax.set_title(label, size=10); ax.axis('off')
    for ax, (_, row, _) in zip(axes[2:], loaded):
        ax.text(.5, -.06, f"PSNR {row['psnr']:.2f} dB\nEdge {row['edge_cosine']:.3f}; energy {row['edge_energy']:.3f}",
                transform=ax.transAxes, ha='center', va='top', size=9)
    fig.text(.05, .03, 'Matched validation inputs; identical RGB and evaluation mask. Raw outputs, no sharpening.', size=9)
    fig.subplots_adjust(left=.02, right=.98, top=.83, bottom=.18, wspace=.06)
    pdf.savefig(fig); plt.close(fig)


def appearance_curve_page(pdf, arm):
    if not arm['config'].get('appearance_transport'):
        return
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 4.5))
    fig.suptitle(arm['label'] + ': learned appearance usage', size=16, y=.94)
    for split, label in [('training_probe', 'Training'), ('validation_probe', 'Validation')]:
        rows = [p for p in arm['probes'] if p['split'] == split]
        steps = [row['step'] for row in rows]
        metrics = [row['exported_metrics'] for row in rows]
        axes[0].plot(steps, [m['generator_weight'] for m in metrics], label=label)
        axes[2].plot(steps, [m['mean_flow_pixels'] for m in metrics], label=label)
        if split == 'validation_probe':
            for key, name in [('mse', 'Final blend'), ('generated_mse', 'Generator'), ('transported_mse', 'Reference sampling')]:
                axes[1].plot(steps, [m[key] for m in metrics], label=name)
    axes[0].set_title('Hidden generator mixture weight'); axes[0].set_ylim(0, 1)
    axes[1].set_title('Validation component RGB MSE'); axes[1].set_yscale('log')
    axes[2].set_title('Mean sampling displacement (pixels)')
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.2); ax.set_xlabel('Run optimizer update')
    fig.tight_layout(rect=[0, 0, 1, .94]); pdf.savefig(fig); plt.close(fig)


def runtime_drift_page(pdf, arms):
    tracked = [arm for arm in arms if arm.get('runtime_drift')]
    if not tracked:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.8))
    elapsed_offset = 0.
    for arm in tracked:
        root = Path(arm['run'])
        rows = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
        groups = [rows[i:i+100] for i in range(0, len(rows), 100)]
        axes[0].plot([np.mean([r['step'] for r in g]) for g in groups],
                     [np.median([r['seconds'] for r in g]) for g in groups], label=arm['label'])
        gpu = [json.loads(line) for line in Path(arm['telemetry']).read_text().splitlines()]
        groups = [gpu[i:i+60] for i in range(0, len(gpu), 60)]
        axes[1].plot([(elapsed_offset + np.mean([r['elapsed_seconds'] for r in g]))/3600 for g in groups],
                     [np.mean([r['device_gpu_percent'] for r in g if 'device_gpu_percent' in r]) for g in groups],
                     label=arm['label'])
        elapsed_offset += gpu[-1]['elapsed_seconds']
    axes[0].set(xlabel='Absolute optimizer update', ylabel='Median seconds / update', title='100-update timing bins')
    axes[1].set(xlabel='Combined process time (hours)', ylabel='Device GPU utilization (%)', title='60-sample telemetry bins')
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.suptitle('Long-process slowdown and exact continuation', size=17, y=.94)
    fig.text(.05, .025, 'Idle time between processes is omitted. Utilization includes desktop activity. Restart behavior is not a root-cause profile.', size=9)
    fig.tight_layout(rect=[0, .08, 1, .94]); pdf.savefig(fig); plt.close(fig)


def pyramid_loss_diagnostic_page(pdf):
    """Analytic translation example paired with the Rust displacement test."""
    target = np.zeros(64); target[22:30] = 1
    reference = np.zeros(64); reference[38:46] = 1
    shifts = np.linspace(-4, 32, 577)
    pooled_target = target.reshape(4, 16).mean(axis=1)
    pooled_reference = reference.reshape(4, 16).mean(axis=1)
    old, new = [], []
    for shift in shifts:
        warped = np.interp(np.arange(64) + shift, np.arange(64), reference)
        reduced_warp = np.interp(np.arange(4) + shift/16, np.arange(4), pooled_reference)
        old.append(np.mean((warped.reshape(4, 16).mean(axis=1) - pooled_target)**2))
        new.append(np.mean((reduced_warp - pooled_target)**2))
    fig, axes = plt.subplots(1, 2, figsize=(11.7, 5.4))
    fig.suptitle('Controlled loss diagnostic: a 16-pixel translation', size=17, y=.94)
    axes[0].step(np.arange(64), target, where='mid', label='Target')
    axes[0].step(np.arange(64), reference, where='mid', label='Reference')
    for x in [16, 32, 48]: axes[0].axvline(x-.5, color='gray', ls=':', lw=.8)
    axes[0].set(xlabel='Pixel coordinate', ylabel='Intensity', title='Stripe spans a different pooling cell')
    axes[1].plot(shifts, old, label='Warp full image, then pool')
    axes[1].plot(shifts, new, label='Pool image, then warp')
    axes[1].axvline(.5, color='gray', ls=':', label='Test displacement: 0.5 px')
    axes[1].axvline(16, color='green', ls='--', label='Correct displacement: 16 px')
    axes[1].set(xlabel='Backward sampling displacement (pixels)', ylabel='Coarse RGB MSE',
                title='Image-pyramid warp supplies a coarse gradient')
    for ax in axes: ax.legend(fontsize=9); ax.grid(alpha=.15)
    fig.text(.04, .035, 'Scale 16, bilinear sampling with border padding. At 0.5 px, the pooled-output term is flat; image-pyramid warping points toward the correct shift.\nThis illustrates one coarse loss term, not the full objective or a learned-model quality result. The corresponding Rust gradient test passes.', size=9)
    fig.tight_layout(rect=[0, .1, 1, .93]); pdf.savefig(fig); plt.close(fig)


def appearance_page(pdf, path):
    row, arrays = inspect(path)
    if 'generator_weight' not in row:
        return
    h, w, _ = arrays['target'].shape
    mixture = np.fromfile(path/'mixture.f32', dtype='<f4').reshape(h, w, -1)
    generated = np.fromfile(path/'generated.f32', dtype='<f4').reshape(h, w, 3)
    transported = np.fromfile(path/'transported.f32', dtype='<f4').reshape(h, w, 3)
    flow = np.fromfile(path/'flow-0.f32', dtype='<f4').reshape(h, w, 2)
    fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3))
    fig.suptitle(f"Appearance components: room {row['room_seed']}, view {row['target_view']}", size=16, y=.94)
    for ax, im, label in zip(axes[0], [arrays['target'], generated, transported],
                             ['Target', 'Generated RGB (all pixels)', 'Sampled reference RGB (all pixels)']):
        ax.imshow(im.clip(0, 1), interpolation='nearest'); ax.set_title(label, size=11)
    im = axes[1, 0].imshow(mixture[:, :, 0], cmap='viridis', vmin=0, vmax=1)
    axes[1, 0].set_title('Generator mixture weight', size=11)
    fig.colorbar(im, ax=axes[1, 0], fraction=.046)
    im = axes[1, 1].imshow(np.linalg.norm(flow, axis=2), cmap='magma', vmin=0, vmax=64*np.sqrt(2))
    axes[1, 1].set_title('Reference 1 sampling displacement (pixels)', size=10)
    fig.colorbar(im, ax=axes[1, 1], fraction=.046)
    yy, xx = np.mgrid[8:h:16, 8:w:16]
    axes[1, 1].quiver(xx, yy, flow[8::16, 8::16, 0], -flow[8::16, 8::16, 1],
                      color='cyan', angles='uv', scale_units='xy', scale=1, width=.003)
    axes[1, 2].imshow(arrays['completion'].clip(0, 1), interpolation='nearest')
    axes[1, 2].set_title('Final completion', size=11)
    for ax in axes.flat: ax.axis('off')
    fig.text(.04, .025, f"Hidden-pixel mean generator weight {row['generator_weight']:.3f}; generated MSE {row['generated_mse']:.5f}; sampled MSE {row['transported_mse']:.5f}.\nMixture identity checked against raw final RGB. Flow is learned from RGB, without geometry labels.", size=9)
    fig.subplots_adjust(left=.035, right=.975, top=.88, bottom=.09, hspace=.22, wspace=.16)
    pdf.savefig(fig); plt.close(fig)


def render(config, result, paths):
    out = Path(config['output']); out.parent.mkdir(parents=True, exist_ok=True)
    selected = result['arms'][result['selected']]
    m = selected['test']['metrics']; c = selected['config']
    diagnostic = result['diagnostics']; ranking = diagnostic['covisibility']
    failed = ', '.join(k.replace('_', ' ') for k, v in result['gates'].items() if not v)
    with PdfPages(out) as pdf:
        text_page(pdf, 'Burn Gekko: regenerated-data study', [
            ('Quality checks passed. ' if result['quality_qualified'] else 'Quality is not fully qualified. ') + config['conclusion'],
            f"Selected checkpoint: {selected['label']}. Reserved-test hidden RGB MSE {m['mse']['mean']:.6f}; mean per-target PSNR {m['psnr']['mean']:.2f} dB; aligned edge cosine {m['edge_cosine']['mean']:.3f}; gradient energy {m['edge_energy']['mean']:.3f}. Failed checks: {failed or 'none'}.",
            f"Actual cumulative capture, training and GPU evaluation command time: {result['command_seconds']/3600:.2f} hours of {result['ceiling_seconds']/3600:.0f}. Overlapping capture/fit durations are both counted. Setup, compilation inside GPU commands, checkpoint writes and evaluation are included. CPU engineering and reporting are excluded.",
            result['visual_review'],
            'Candidate weights contain no noncommercial pretrained Gekko initialization or teacher. MIT V-JEPA 2.1 is the only external pretrained model permitted in this study; fusion and RGB/RI heads start randomly at the root of the recorded checkpoint lineage. Explicit transfers of our own trained weights retain parent hashes and disclose optimizer resets. These are synthetic-room findings, not real-image or repeat-seed qualification.',
        ])
        table_page(pdf, 'Cumulative budget: every capture / training / GPU-evaluation command',
                   ['Study leg', 'Command', 'Status', 'Minutes'],
                   [[row['leg'], row['name'], row['status'], f"{row['elapsed_seconds']/60:.2f}"]
                    for row in result['budget_commands']])
        inventory = result['inventory']
        oracle = result['geometry_oracle']
        text_page(pdf, 'Data, camera spacing and evaluation', [
            'Capture uses published bevy_zeroverse 0.25.0 and bevy_zeroverse_burn 0.8.0. Registry timestamps and checksums, tool lockfile and capture binary identity are retained. Old caches are preserved; new room seeds were checked for overlap.',
            f"Dataset: {c['dataset']}. {inventory.get('description', '')} Images are 256×256 with three simultaneous cameras, mixed procedural layouts, density 0.35, no humans and portable rendering. Continuous camera baseline is 0.25; wider baselines are not qualified by this study.",
            f"The baseline control 0.25 is dimensionless. In this published generator it expands to minimum pair separation 0.155 m and camera-zero reference distances 0.635-2.7625 m, with a proposal overlap setting of 0.411. Across the 128 validation rooms, actual all-pair distances have median {result['camera_policy']['measured_pairwise_metres']['median']:.3f} m, 95th percentile {result['camera_policy']['measured_pairwise_metres']['p95']:.3f} m, and maximum {result['camera_policy']['measured_pairwise_metres']['maximum']:.3f} m. Distances between the two nonzero cameras may exceed the camera-zero reference cap. Proposal overlap is not measured pixel co-visibility.",
            'Raw RGB, depth, position and camera metadata remain in immutable compressed shards under .data. Training loads RGB only into a bounded host cache and uploads the current batch; trainable encoder features are always recomputed. Geometry is decoded only for diagnostic/evaluation labels after RGB inference.',
            f"A separate validation-only geometry diagnostic covers {100*oracle['mean_covered_fraction']:.1f}% of pixels on average across {oracle['rooms']} rooms, target view zero. Covered-pixel RGB MSE is {oracle['mean_oracle_mse']:.6f} with ground-truth reprojection, versus {oracle['mean_unwarped_mse']:.6f} at unchanged reference coordinates. This checks capture coherence. It uses privileged geometry and a different pixel support from masked model evaluation; it is not a model result or a matched performance comparison.",
            'Development uses training and validation rooms. Candidate selection is recorded before reserved-test inference. Every held-out room and all three target views are exported. RGB metrics use unclipped float predictions on hidden pixels. Displayed completions retain observed input pixels and clip to [0,1], without sharpening, resizing enhancement or contrast adjustment. Co-visibility ranking uses all geometry-known pixels.',
            'Training draws a new mask per update, shared across examples within that batch and across the two reconstruction branches. Evaluation uses one fixed 75% hidden-patch mask, shared across rooms, views and checkpoints. This study does not establish robustness across a distribution of evaluation masks.',
            'Intervals use 2,000 room-bootstrap replicates: views from the same room are grouped. Edge metrics require both endpoints hidden. Separate patch-interior and seam metrics expose artificial block boundaries. These intervals capture room-sampling uncertainty, not training-seed uncertainty.',
        ])
        sizing = read(config.get('sizing', str(Path(config['study']) / 'sizing.json')))
        sampling = selected['sampling']
        assert result['lineage']['target_exposures'] - sampling['target_exposures'] == sizing['parent_curriculum_target_exposures']
        text_page(pdf, 'Dataset sizing and actual training exposure', [
            f"The selected run was sized from measured batch-{c['batch_size']} throughput: {sizing['planned_steps']:,} requested updates over {sampling['eligible_target_views']:,} distinct target views, {sizing['planned_target_view_epochs']:.2f} equivalent passes. The measured command ceiling takes precedence over the requested step count. The raw training RGB host cache is {sizing['training_rgb_cache_bytes']/2**30:.1f} GiB; immutable compressed shards for all splits occupy {inventory['bytes']/1e9:.2f} GB.",
            f"Recorded selected-run optimizer inputs: {sampling['executed_updates']:,} updates, {sampling['target_exposures']:,} target-view exposures, {sampling['unique_rooms']:,} distinct rooms, {sampling['unique_target_views']:,} distinct target views, and {sampling['target_view_epochs']:.2f} equivalent passes. Per-target exposure counts range from {sampling['minimum_exposures_per_target']} to {sampling['maximum_exposures_per_target']}. Every logged input was checked against the configured training split; no validation or test input appears in an optimizer batch.",
            f"The preceding recorded curriculum contributes another {sizing['parent_curriculum_target_exposures']:,} target exposures across {sizing['parent_curriculum_unique_training_rooms']:,} training rooms. Some rooms may be reused by the selected run. " + ('This run exactly resumes its own checkpoint with both optimizer states and the original schedule restored.' if selected['provenance'].get('resume') else 'This run transfers its own checkpoint with both optimizers reset.') + ' Earlier training is included in the study budget and lineage; fitting-set detail is not held-out quality evidence.',
            f"A recursive audit follows both exact-resume and own-weight transfer records, verifies model hashes, and counts overlapping checkpoint prefixes once. The complete selected ancestry contains {result['lineage']['executed_updates']:,} optimizer updates and {result['lineage']['target_exposures']:,} target exposures across {result['lineage']['unique_training_rooms']:,} distinct training rooms. Every ancestral logged input belongs to its configured training split.",
            'Dataset count alone does not establish adequacy. Training and validation probe curves, actual exposure counts, all-target evaluation, aligned-detail measurements and raw samples determine whether this budget has learned useful reconstruction. A run that still underfits or produces incoherent detail is reported as unresolved.',
        ])
        restart_path = Path(config['study']) / 'restart-comparison.json'
        if restart_path.exists():
            restart = read(restart_path)
            before = restart['arms']['refine']['metrics']
            after = restart['arms']['refine_continue']['metrics']
            text_page(pdf, 'Extended training and checkpoint selection', [
                'The main-data lineage includes 12,001 direct-head updates, 3,000 image-pyramid updates, the 400-update mild-curvature screen, and 11,829 extended updates. The last 493 extended updates run in a fresh process with exact optimizer and scheduler restoration. Other diagnostic arms consume budget but are not ancestors of the selected weights.',
                'The extended process stopped at its wall cap after 11,336 updates, short of the requested 14,000. Its available periodic checkpoints were screened on the same fixed validation probe. The final long-run checkpoint improved both MSE and detail over the available 2k/4k/6k/8k/10k snapshots. The bounded restart compares against that final checkpoint on all 384 validation targets.',
                f"Restart validation MSE: {before['mse']['mean']:.6f} to {after['mse']['mean']:.6f}; aligned edge cosine: {before['edge_cosine']['mean']:.4f} to {after['edge_cosine']['mean']:.4f}; patch-interior energy: {before['interior_edge_energy']['mean']:.4f} to {after['interior_edge_energy']['mean']:.4f}. The selected continuation trades a small energy decrease for lower error and better alignment. Neither checkpoint passes the detail gate. Selection timestamp and model SHA precede reserved-test inference.",
                'The parent process slowed from about 0.90 to 2.24 seconds per update while sampled GPU utilization fell from about 85% to 32%. GPU clocks stayed near 2.8 GHz; temperature and power fell. A fresh exact continuation restored a 1.064-second median per update after 50 warmup updates (2.11 times faster than the preceding last 100 updates). This is evidence for a process-lifetime throughput problem and a measured restart workaround, not a diagnosed thermal issue or an identified backend defect. Future long runs need process profiling and bounded exact-resume segments.',
                'Full startup validates and decodes metadata from every dataset shard, including when evaluating a small subset. Large-data training preparation takes about 130 seconds. Initial preparation, all diagnostics, checkpoint finalization and failed command attempts remain in the cumulative budget.',
            ])
            table_page(pdf, 'Exact continuation: paired validation differences',
                       ['Metric', 'Continuation minus parent [95% room interval]'],
                       [[key, fmt(value)] for key, value in restart['paired_differences'][0]['metrics'].items()])
        if all(k in inventory for k in ['layout', 'palette', 'lighting']):
            fig, axes = plt.subplots(1, 3, figsize=(11.7, 5)); fig.suptitle('Regenerated room diversity', size=17, y=.94)
            for ax, key in zip(axes, ['layout', 'palette', 'lighting']):
                items = sorted(inventory[key].items()); ax.barh([k for k, _ in items], [v for _, v in items]); ax.set_title(key)
            fig.tight_layout(rect=[0, 0, 1, .91]); pdf.savefig(fig); plt.close(fig)
        text_page(pdf, 'Architecture and direct RGB objective', [
            f"Shared V-JEPA 2.1 image encoder with sparse target encoding: a 16px patch stem, 12 blocks, width 768. Fusion decoder: {c['decoder_depth']} blocks, width {c['decoder_width']}, {c['decoder_heads']} heads, 2D rotary positions and joint attention over two references. Fusion weights start randomly at the root of the recorded curriculum. The target exposes 25% of its patches; each reference exposes all patches. Available RGB patches accompany semantic tokens.",
            'The direct head predicts unconstrained RGB in fixed ImageNet-normalized channel space. It removes the forced per-patch unit-variance prediction and predicted mean/scale product used in pilot 05. Dense target features cannot enter RGB completion; the dense target is reserved for the separate RI operation.',
            'Cross-view and monocular branches now receive identical hidden-pixel losses: 10×RGB MSE + RGB L1 + aligned derivative L1 at offsets 1, 2, 4 and 8. Both derivative endpoints must be hidden. There is no unaligned gradient-energy reward. RI targets and weights are detached reconstruction-error differences; the RI branch is delayed during early reconstruction fitting. This is an adapted objective, not exact released-paper parity.',
            f"Derivative coefficient: {c['edge_loss_weight']}/4 for each of eight axis/offset terms. AdamW: peak decoder LR {c['learning_rate']}, encoder ratio {c['encoder_lr_ratio']}, weight decay {c['weight_decay']}, warmup {c['warmup_steps']} updates, fixed cosine horizon {c['decay_steps']}. Batch {c['batch_size']}; mask ratio {c['mask_ratio']}; RI begins at zero-indexed update {c['ri_start_step']}. Global gradient norm clipping is 1. The initial run uses measured unfreezing gates; explicit transfers of our own weights inherit the encoder stage.",
            'Training uses Burn CUDA, F32 tensors, operator fusion and automatic differentiation of matrix-product/softmax attention. Backend kernel selection governs internal arithmetic; the tensor dtype alone does not establish strict IEEE FP32 arithmetic for every operation.',
            'Periodic snapshots atomically save model, both optimizers, encoder stage and absolute step with checksums. Resumption reconstructs the exact mask and room/view schedule and verifies backend/source/config identity. Source and binary archives retain each experimental implementation.',
        ])
        if c.get('appearance_transport'):
            text_page(pdf, 'Optional learned appearance transport', [
                'The direct-head baseline improved aligned edges but retained low detail energy. A recorded adaptive amendment preserves that baseline and spends the remaining budget on an appearance-sampling extension. This combines an architectural change and further training; it is not an isolated causal ablation.',
                f"The same decoder also processes each reference separately. A shared linear head predicts dx, dy, reference confidence and generator confidence at four samples per patch axis. Bilinear interpolation produces a dense flow; tanh bounds each displacement component to {c['transport_max_displacement']} pixels. The model samples reference RGB bilinearly and mixes those samples with its existing generated RGB. Shared parameters and symmetric aggregation preserve mathematical reference-order symmetry; native numerical residuals are measured separately.",
                f"Both reconstruction branches retain their primary hidden-pixel RGB/derivative objectives. The cross-view path additionally receives {c['transport_loss_weight']} times an RGB photometric auxiliary: full-image MSE, multiscale MSE, derivative L1 coefficient 0.05, and flow-smoothness MSE coefficient 0.00001. These are RGB-derived training targets, with no geometry or correspondence labels. This auxiliary means the two branches do not receive identical total training signals.",
                ('The selected run uses image-pyramid warping at scales 2/4/8/16/32: pool reference RGB, target RGB, mixture weights and flow; divide flow coordinates by the scale; warp each reduced reference; and compare its mixture with the reduced target. A controlled 16-pixel translation test verifies a useful displacement gradient when reducing an already warped image gives a flat coarse loss. The earlier pooled-output attempt is retained as a diagnostic. The correction supplies a broader matching signal; its real-room effectiveness must still be established by the measurements.' if c.get('transport_pyramid_loss') else 'This run pools already warped RGB and target RGB at scales 2/4/8/16. A subsequent controlled translation test exposed a flat coarse displacement gradient in that formulation, motivating a separate image-pyramid correction.'),
                f"Native-flow curvature coefficient: {c.get('transport_coarse_smoothness_weight', 0):g}, inside the outer auxiliary weight. When enabled, quarter-resolution displacement controls receive edge-aware absolute second-derivative loss, averaged over references and axes. Target RGB is pooled by four and detached; adjacent color differences attenuate the penalty. Constant and affine fields have zero curvature. This constrains distortion without camera, depth or flow labels. The native tanh control field and upsample-then-tanh inference field are distinct; both are retained in exports.",
                'RI targets compare the actual blended completion with the monocular branch; a contract checks their RGB normalization identity. Hidden target RGB never enters the sampling predictor. The encoder remains trainable after the preceding stabilized trunk, with the smaller recorded encoder learning rate. The sampling head began randomly when the extension was added; later transfers and exact resumptions preserve its own learned weights.',
                'Every exported prediction retains the generated RGB, transported RGB, source-mixture weights and backward-sampling flow. Independent analysis checks that the mixture reconstructs the saved final RGB, then reports component error, source usage and flow magnitude. Higher gradient energy without aligned structure does not qualify as a fix.',
            ])
            if c.get('transport_pyramid_loss'):
                pyramid_loss_diagnostic_page(pdf)
            decision_path = Path(config['study']) / 'coarse-decision.json'
            if decision_path.exists():
                choice = read(decision_path)
                text_page(pdf, 'Matched curvature-loss comparison', [
                    'Three 400-update continuations use the same image-pyramid parent, seed, optimizer resets, room/view inputs, masks and learning-rate schedule. Only the native curvature coefficient differs: 0, 0.01 and 0.0025. Initial raw predictions are bitwise equal; every logged training batch matches. Full validation covers 128 rooms and all three target views.',
                    'Coefficient 0.01 reduces hidden-visible flow endpoint error from 32.48 to 31.39 pixels but lowers RGB edge energy from 0.1808 to 0.1738. Its reconstruction MSE is 0.002292 versus 0.002287 for the unregularized control. This stronger penalty is retained as a diagnostic.',
                    f"The weaker coefficient 0.0025 meets the engineering continuation tolerances recorded before that arm began: MSE ratio {choice['ratios']['mse']:.6f}, patch-interior energy ratio {choice['ratios']['interior_edge_energy']:.6f}, interior edge-cosine difference {choice['ratios']['interior_edge_cosine_difference']:.6f}, and flow-error ratio {choice['ratios']['flow_error']:.6f}, each versus coefficient zero. These tolerances choose a recipe to train longer; they do not replace the final quality gates.",
                    'Paired differences and room-bootstrap intervals are retained in coarse-final-comparison.json. These experiments isolate this coefficient over 400 updates in one training lineage. The later long continuation changes training duration and does not establish that curvature alone causes its final improvement.',
                ])
        rows = []
        for arm in result['arms'].values():
            for split in ['training', 'validation', 'test']:
                if split not in arm: continue
                r = arm[split]; mm = r['metrics']
                rows.append([arm['label'], split, str(len(r['rows'])), f"{mm['mse']['mean']:.6f}", f"{mm['edge_cosine']['mean']:.3f}", f"{mm['edge_energy']['mean']:.3f}"])
        table_page(pdf, 'Complete exported-target measurements', ['Run', 'Split', 'Targets', 'Hidden MSE', 'Edge cosine', 'Edge energy'], rows)
        table_page(pdf, 'Reserved test: room-bootstrap 95% intervals', ['Metric', 'Estimate [low, high]'],
                   [[key, fmt(value)] for key, value in m.items()])
        table_page(pdf, 'Retained acceptance checks', ['Check', 'Result'],
                   [[key.replace('_', ' '), 'PASS' if passed else 'FAIL']
                    for key, passed in result['gates'].items()])
        for arm in config['arms']:
            if arm.get('show_curves', True):
                curve_page(pdf, arm['label'], result['arms'][arm['name']], Path(arm['run']))
                appearance_curve_page(pdf, result['arms'][arm['name']])
        rows = []
        for arm in result['arms'].values():
            for perf in arm['throughput']['phases']:
                phase = f"{perf['stage']}, RI {'on' if perf['ri_enabled'] else 'off'}"
                rows.append([arm['label'], phase, str(perf['measured_updates']), f"{perf['median_seconds']*1000:.1f}", f"{perf['targets_per_second']:.1f}", f"{arm['runtime']['peak_process_vram_gib']:.1f}"])
        table_page(pdf, 'Training efficiency: exclude first 50 updates per stage / RI phase', ['Run', 'Phase', 'Measured', 'Median ms', 'Targets/s', 'Peak GiB'], rows)
        runtime_drift_page(pdf, config['arms'])
        text_page(pdf, 'Efficiency scope and batch choice', [
            'Workstation inventory at closeout: NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97,887 MiB reported device memory, driver 610.43.02; host RAM 94.10 GiB. Rust 1.98.0, Burn 0.21.0, native CUDA F32/Fusion backend. The study uses one GPU. Dependency lockfiles and binary hashes are archived.',
            'A target per second means one reconstruction training example, with its two reference views and matched monocular branch; it does not count every encoded view as a separate example. Update timings include batch transfer, forward/backward computation, optimizer work and synchronization. Probe evaluation, checkpoint writes and initial preparation are excluded from the phase table and included in total run time.',
            *[f"{arm['label']}: {arm['throughput']['executed_updates']:,} executed updates in {arm['throughput']['run_seconds']/3600:.2f} trainer hours; {arm['throughput']['prepare_seconds']:.1f} seconds of preparation; {arm['throughput']['end_to_end_targets_per_second']:.2f} target examples/s including preparation, probes, checkpoints and final validation. Sampled host RSS peak {arm['throughput']['sampled_peak_process_rss_mib']/1024:.2f} GiB." for arm in result['arms'].values()],
            'Separate uncontended screens measured batch 16 at 19.47 targets/s with RI enabled and batch 32 at 20.22 targets/s, excluding 50 warmup updates per phase. Batch 32 gained 3.88% while sampled process memory increased from 30,740 to 60,532 MiB. Batch 16 was retained for shorter updates and lower memory use. The screens establish throughput only, not model quality.',
            'The pooled-warp appearance extension received a separate batch-16 screen: 18.03 targets/s without RI and 15.37 with RI, with 35,924 MiB sampled process VRAM. The image-pyramid version measured 18.02 and 15.23 targets/s, respectively, at 36,020 MiB. Each phase has 30 measured updates after excluding 50 warmup updates. These fitting-set screen weights are not transferred into the larger-data candidate. Its measured runtime and throughput appear separately above.',
        ])
        audit = diagnostic['input_audit']
        text_page(pdf, 'Reference controls and co-visibility', [
            f"Cross-view MSE reduction versus the paired monocular branch: {100*(1-m['mse']['mean']/m['monocular_mse']['mean']):.2f}%. Related-reference advantage versus unrelated rooms: {fmt(m['reference_gain'])}. Scene-level color/context can help without local correspondence, so these controls are read alongside edge alignment and images.",
            f"Hidden-target replacement changes completion by {audit['hidden_target_intervention_max_abs']:.6g}; reference reordering by {audit['reference_permutation_max_abs']:.6g}; monocular reference reordering by {audit['monocular_reference_permutation_max_abs']:.6g}. Reference-order residual is reported against the earlier strict 1e-5 check without silently changing its threshold.",
            f"Native GPU batching audit uses {audit.get('batch_audit_unique_rooms', 0)} distinct rooms at batch {audit.get('batched_targets', 0)}. Independent-versus-batched hidden RGB RMS difference is {audit.get('batched_rgb_hidden_rms', float('nan')):.6g}; maximum RGB difference {audit.get('batched_rgb_max_abs', float('nan')):.6g}. Reference-order RGB RMS difference is {audit.get('reference_permutation_rms', float('nan')):.6g}. These are numerical-control measurements, separate from reconstruction error.",
            f"Learned dense RI pooled AUROC {ranking['auroc']:.4f}, AP {ranking['average_precision']:.4f}; constant-score AP baseline {ranking['positives']/ranking['pixels']:.4f}. Scores are relative reconstruction utility, not calibrated visibility probabilities. Per-target AUROC with room intervals: {fmt(result['ranking_intervals']['auroc'])}.",
            *[f"Evaluation-geometry-conditioned {key}: {fmt(value)}" for key, value in result['geometry_conditioned'].items()],
            'Process VRAM is separate from device-wide utilization/power. The initial fitting diagnostic overlaps capture and is not an uncontended throughput benchmark. Main training follows capture; separately budgeted native batching (25.6 seconds) and appearance preflight (135.8 seconds) overlap baseline updates. Unrelated desktop or external GPU activity can still affect device telemetry. Dedicated screens establish throughput without another study training or capture job running.',
        ])
        if 'appearance_geometry' in result:
            text_page(pdf, 'Learned sampling flow: geometry-only evaluation', [
                'After model inference, saved flows are compared with projection through captured camera matrices and depth-consistent world positions. Pixel-center convention, self reprojection, raw target RGB and reference order are checked against immutable room shards. Geometry never modifies a prediction or contributes to an optimizer step.',
                *[f"{support}: mean endpoint error {row['endpoint_error_pixels']:.2f} pixels; unchanged-coordinate baseline {row['identity_endpoint_error_pixels']:.2f} pixels; fraction below 3 pixels {100*row['fraction_within_3_pixels']:.2f}%; true displacement beyond the head's 64-pixel per-axis range {100*row['fraction_gt_beyond_64_per_axis']:.2f}%." for support, row in result['appearance_geometry']['means'].items()],
                *[f"Union across references, {support}: {100*row['visible_fraction']:.2f}% of target pixels have a valid corresponding reference; {100*row['in_range_fraction']:.2f}% have at least one corresponding reference within the head's range. {100*row['visible_but_all_references_out_of_range_fraction']:.2f}% are reference-visible but every valid reference is outside the range. This union differs from the per-reference out-of-range rate above." for support, row in result['appearance_geometry'].get('coverage_means', {}).items()],
                'Means weight each exported target/reference pair equally. Only depth-consistent, in-frame correspondences contribute. Textureless regions can have low RGB error despite incorrect flow, so photometric quality and geometric accuracy are reported separately. This is a diagnostic of the learned sampler, not an additional training target or a requirement imposed on the generative branch.',
            ])
        for arm in config['arms']:
            if arm.get('training_samples'):
                key = sorted(paths[arm['name'], 'training'])[0]
                sample_page(pdf, paths[arm['name'], 'training'][key], arm['label'] + ': training-only sample')
        compared = [arm for arm in config['arms'] if arm['name'] in ['main', result['selected']]]
        if len(compared) > 1:
            common = set.intersection(*(set(paths[arm['name'], 'validation']) for arm in compared))
            for key in [key for key in sorted(common) if key[1] == 0][:2]:
                comparison_page(pdf, key, [(arm['label'], paths[arm['name'], 'validation'][key]) for arm in compared])
        selected_paths = paths[result['selected'], 'test']
        keys = sorted(selected_paths); ranked = sorted(selected['test']['rows'], key=lambda r: r['mse'])
        chosen = [('First reserved room', k) for k in keys if k[1] == 0][:2]
        chosen += [('Median error', (ranked[len(ranked)//2]['room_seed'], ranked[len(ranked)//2]['target_view'])),
                   ('Worst error', (ranked[-1]['room_seed'], ranked[-1]['target_view']))]
        for i, (label, key) in enumerate(chosen):
            sample_page(pdf, selected_paths[key], label, out.parent/'annotated-sample.png' if i == 0 else None)
            if i == 0:
                appearance_page(pdf, selected_paths[key])
        fig, axes = plt.subplots(2, 3, figsize=(11.7, 8.3)); fig.suptitle('Co-visibility labels and learned relative improvement', size=16, y=.94)
        for row, key in enumerate(keys[:2]):
            p = selected_paths[key]; _, a = inspect(p); source = Path(config['diagnostics']).parent / p.name
            labels = np.fromfile(source/'visibility.u8', dtype='u1').reshape(a['hidden'].shape)
            scores = np.fromfile(source/'ri.f32', dtype='<f4').reshape(a['hidden'].shape)
            colors = np.zeros((*labels.shape, 3)); colors[labels == 1] = [.2, .75, .4]; colors[labels == 0] = [.75, .2, .2]; colors[labels == 255] = [.5, .5, .5]
            axes[row, 0].imshow(a['target'].clip(0, 1)); axes[row, 0].set_title(f'Room {key[0]}, view {key[1]}')
            axes[row, 1].imshow(colors); axes[row, 1].set_title('Green: visible; red: absent; gray: unknown')
            im = axes[row, 2].imshow(scores, cmap='coolwarm', vmin=-1, vmax=1); axes[row, 2].set_title('RI score, not probability')
            for ax in axes[row]: ax.axis('off')
            fig.colorbar(im, ax=axes[row, 2], fraction=.04)
        fig.tight_layout(rect=[0, 0, 1, .91]); pdf.savefig(fig); plt.close(fig)
        text_page(pdf, 'Interpretation, limits and reproducibility', [
            config['conclusion'],
            *([f"A diagnostic of the first 128 main training rooms, all three targets, has hidden MSE {selected['training']['metrics']['mse']['mean']:.6f} and gradient energy {selected['training']['metrics']['edge_energy']['mean']:.3f}. Blur is also present on these training inputs; the failure is not confined to unseen rooms. This prefix is not a random estimate of the whole training set and does not identify a specific optimization or model-capacity cause."] if 'training' in selected else []),
            *config.get('limitations', []),
            'Pilot 05 used different generated rooms and a different camera policy; its metrics are historical context, not a matched numerical control. Improved RGB error alone cannot certify reconstruction detail. No automatic sharpening, oracle statistics or geometry warps are used in model outputs.',
            'For a paper, separate wholly random training from permissive encoder adaptation and any noncommercial reference baseline. Report the exact adapted objective, complete budget, unsuccessful diagnostics, data provenance and reference controls. Multiple training seeds, camera-baseline scaling and real-room transfer remain separate research qualification steps.',
            f"Reproduction artifacts: {config['study']}; resolved configs and checkpoints: {config['arms'][-1]['run']}. Use the archived binary matching checkpoint identity; exact continuation restores both optimizers and keeps the original decay horizon. Dataset hashes, source/binary receipts, raw exports, telemetry and this report's machine-readable summary are retained.",
        ])
        if config.get('next_steps'):
            text_page(pdf, 'Remaining work, not completed in this budget', config['next_steps'])
        text_page(pdf, 'Verification and primary references', [
            'The v9 implementation passed 68 workspace tests, strict CUDA Clippy and native builds. At closeout, all 53 Rust source and dependency files matched the tested v9 archive. Five Python report/runner tests, Python syntax checks and Cargo formatting passed. These implementation checks do not override the failed image-quality and numerical acceptance gates.',
            'Loiseau, Bourmaud and Lepetit (2026), Revisiting Cross-View Completion: Self-Supervised Pre-Training via Reconstruction Error Comparison. Original Gekko method and attribution: https://arxiv.org/abs/2609.01530 . This study adapts the architecture and objectives and does not reuse released Gekko weights.',
            'Jonschkowski et al. (2020), What Matters in Unsupervised Optical Flow. Motivation for examining photometric matching and native-resolution smoothness: https://arxiv.org/abs/2006.04902 . The present appearance extension is not a reproduction of UFlow and imports no UFlow weights.',
            'V-JEPA source revision 204698b45b3712590f06245fbfba32d3be539812: https://github.com/facebookresearch/vjepa2 . Encoder checkpoint: https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt . License and provenance copies are retained in .data/pilot-06/licenses and each selected checkpoint.',
            'Published capture records: https://crates.io/api/v1/crates/bevy_zeroverse/0.25.0 and https://crates.io/api/v1/crates/bevy_zeroverse_burn/0.8.0 . Registry checksums and the September 29 recheck are retained alongside the immutable capture identity.',
        ])
    print(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    result, paths = audit(config)
    Path(config['summary']).write_text(json.dumps(result, indent=2) + '\n')
    render(config, result, paths)


if __name__ == '__main__':
    main()
