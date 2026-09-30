#!/usr/bin/env python3
"""Verify completed fusion experiments and summarize their reproducible evidence.

Reads saved artifacts only. This does not train, score a holdout, or select a model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def flatten(value, prefix=''):
    result = {}
    for key, item in value.items():
        name = f'{prefix}.{key}' if prefix else key
        if isinstance(item, dict):
            result.update(flatten(item, name))
        else:
            result[name] = item
    return result


def inspect_run(path):
    config = tomllib.loads((path / 'config.toml').read_text())
    report = read(path / 'report.json')
    snapshot = read(path / 'final/metadata.json')
    rows = [json.loads(line) for line in (path / 'metrics.jsonl').read_text().splitlines()]
    assert rows and report['stop_reason'] == 'step_limit', f'incomplete run: {path}'
    assert snapshot['completed_steps'] == report['completed_steps'] == config['steps']
    assert [row['step'] for row in rows] == list(range(report['starting_step'] + 1, config['steps'] + 1))
    assert report['teacher_max_abs_delta'] == 0, f'teacher changed: {path}'
    assert snapshot['noncommercial_weight_dependencies'] == []
    for name, key in [('model', 'model_sha256'), ('encoder-optimizer', 'encoder_optimizer_sha256'), ('fusion-optimizer', 'fusion_optimizer_sha256')]:
        assert sha(path / 'final' / f'{name}.mpk') == snapshot[key], f'{name} checksum mismatch'
    for row in rows:
        assert (row['encoder_gradient_tensors'] == 0) == (row['stage'] == 0)
        assert all(np.isfinite(row[key]) for key in ['total', 'cross', 'monocular', 'gradient_norm', 'seconds'])
    if all(row['stage'] == 0 for row in rows):
        assert report['first_encoder_max_abs_delta'] == report['last_encoder_max_abs_delta'] == 0
    if any(row['stage'] >= 1 for row in rows):
        assert report['last_encoder_max_abs_delta'] > 0, f'unfrozen final encoder block did not update: {path}'
    if any(row['stage'] == 2 for row in rows):
        assert report['first_encoder_max_abs_delta'] > 0, f'fully unfrozen encoder did not update: {path}'
    samples = [sample for row in rows for sample in row['samples']]
    sample_hash = hashlib.sha256(json.dumps(samples, separators=(',', ':')).encode()).hexdigest()
    stages = {}
    for stage in sorted({row['stage'] for row in rows}):
        part = [row for row in rows if row['stage'] == stage]
        measured = part[min(10, len(part) // 2):]
        median = float(np.median([row['seconds'] for row in measured]))
        stages[str(stage)] = dict(updates=len(part), median_seconds=median,
                                 targets_per_second=config['batch_size'] / median,
                                 encoder_gradient_tensors=sorted({row['encoder_gradient_tensors'] for row in part}))
    validation = read(path / 'validation/metrics.json')
    summary = dict(path=str(path), model_sha256=snapshot['model_sha256'], teacher_id=snapshot['teacher_id'],
                   config_sha256=sha(path / 'config.toml'), metrics_sha256=sha(path / 'metrics.jsonl'),
                   starting_step=report['starting_step'], completed_steps=config['steps'],
                   updates=len(rows), target_exposures=len(samples), unique_rooms=len({x[0] for x in samples}),
                   unique_room_targets=len({tuple(x) for x in samples}), sample_sequence_sha256=sample_hash,
                   run_seconds=report['run_seconds'], stage_efficiency=stages,
                   encoder_parameter_max_abs_delta=dict(first_block=report['first_encoder_max_abs_delta'],
                                                        last_block=report['last_encoder_max_abs_delta']),
                   validation={key: value for key, value in validation.items() if key not in ['rows']},
                   noncommercial_weight_dependencies=[])
    return summary, config, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    output = Path(config['output'])
    assert output.resolve().is_relative_to(Path('.data').resolve()) and not output.exists()
    inspected = {name: inspect_run(Path(path)) for name, path in config['runs'].items()}
    result = dict(config_sha256=sha(args.config), runs={name: item[0] for name, item in inspected.items()}, matched_comparisons={})
    for name, pair in config.get('comparisons', {}).items():
        left, lc, lr = inspected[pair['left']]
        right, rc, rr = inspected[pair['right']]
        assert left['sample_sequence_sha256'] == right['sample_sequence_sha256'], f'sample mismatch: {name}'
        assert [(r['step'], r['learning_rate']) for r in lr] == [(r['step'], r['learning_rate']) for r in rr], f'learning-rate schedule mismatch: {name}'
        same_stages = [r['stage'] for r in lr] == [r['stage'] for r in rr]
        if pair.get('match_encoder_stages', True):
            assert same_stages, f'encoder stage mismatch: {name}'
        a, b = flatten(lc), flatten(rc)
        changed = sorted(key for key in a.keys() | b.keys() if a.get(key) != b.get(key))
        assert set(changed) == set(pair['changed_fields']), f'unexpected configuration differences: {name}: {changed}'
        old = left['validation']['mean_cross_mse']
        new = right['validation']['mean_cross_mse']
        result['matched_comparisons'][name] = dict(left=pair['left'], right=pair['right'],
            changed_fields=changed, same_samples=True, same_learning_rate_schedule=True,
            same_encoder_stages=same_stages,
            relative_latent_error_change=new / old - 1,
            within_five_percent_latent_regression=new <= 1.05 * old)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(runs=len(inspected), matched_comparisons=result['matched_comparisons'], output=str(output)), indent=2))


if __name__ == '__main__':
    main()
