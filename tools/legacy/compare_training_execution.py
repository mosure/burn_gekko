#!/usr/bin/env python3
"""Check a matched execution-only training change before using it for quality runs."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import tomllib


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    c = tomllib.loads(args.config.read_text())
    out = Path(c['output'])
    assert out.resolve().is_relative_to(Path('.data').resolve()) and not out.exists()
    roots = [Path(c[key]) for key in ['before', 'after']]
    configs = [tomllib.loads((p/'config.toml').read_text()) for p in roots]
    for config in configs:
        config['fusion_auxiliary'].setdefault('teacher_layers', [])
        config.setdefault('spatial_input_scale', 1.0)
    assert configs[0] == configs[1], 'execution comparison changed the task'
    rows = [[json.loads(line) for line in (p/'metrics.jsonl').read_text().splitlines()] for p in roots]
    reports = [read(p/'report.json') for p in roots]
    assert len(rows[0]) == len(rows[1]) == configs[0]['steps']
    for report in reports:
        assert report['stop_reason'] == 'step_limit' and report['starting_step'] == 0
        assert report['teacher_max_abs_delta'] == 0 and report['first_encoder_max_abs_delta'] == 0 and report['last_encoder_max_abs_delta'] == 0
    fields = ['total','cross','monocular','visible','ri','attention_kl','dense_latent_mse','descriptor_kl','gradient_norm']
    deltas = {key: 0.0 for key in fields}
    within = True
    for a,b in zip(*rows):
        for key in ['samples','step','learning_rate','stage','encoder_gradient_tensors']:
            assert a[key] == b[key]
        for key in fields:
            delta = abs(a[key]-b[key]);deltas[key] = max(deltas[key],delta)
            within &= delta <= c['absolute_tolerance'] + c['relative_tolerance'] * abs(a[key])
    validation = [read(p/'validation/metrics.json')['mean_cross_mse'] for p in roots]
    within &= abs(validation[0]-validation[1]) <= c['validation_absolute_tolerance']
    medians = [statistics.median(r['seconds'] for r in part[c['warmup_updates']:]) for part in rows]
    result = dict(config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  same_task_examples_and_schedule=True, updates=len(rows[0]),
                  maximum_absolute_deltas=deltas, validation_mse=validation,
                  numerical_gate_passed=bool(within), median_update_seconds=medians,
                  warmed_update_speedup=medians[0]/medians[1],
                  scope='Same pretrained own checkpoint, fresh optimizers, frozen encoder and inputs; excludes declared warmup updates from speed ratio. Energy is evaluated separately.')
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    assert within, 'execution change failed the registered numerical tolerance'


if __name__ == '__main__':
    main()
