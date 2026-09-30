#!/usr/bin/env python3
"""Summarize completed runs by encoder stage and reconstruction/RI phase."""
import argparse
import json
from pathlib import Path
import tomllib

import numpy as np


def summarize(arm):
    root = Path(arm['run'])
    config = tomllib.loads((root / 'config.toml').read_text())
    report = json.loads((root / 'report.json').read_text())
    steps = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
    gpu = [json.loads(line) for line in Path(arm['telemetry']).read_text().splitlines()]
    assert steps and gpu
    assert steps[-1]['step'] == report['completed_steps']
    phases = {}
    for step in steps:
        # Logs use completed updates; the training schedule is zero-indexed.
        ri = step['step'] - 1 >= config['ri_start_step'] and config['ri_weight'] > 0
        phases.setdefault((step['stage'], ri), []).append(step)
    rows = []
    for (stage, ri), phase in phases.items():
        timed = phase[50:]
        if not timed:
            continue
        seconds = np.array([step['seconds'] for step in timed], dtype=float)
        assert np.isfinite(seconds).all() and (seconds > 0).all()
        rows.append(dict(stage=stage, ri_enabled=ri, phase_updates=len(phase), measured_updates=len(timed),
            first_measured_step=timed[0]['step'], last_measured_step=timed[-1]['step'],
            median_seconds=float(np.median(seconds)), p90_seconds=float(np.quantile(seconds, .9)),
            targets_per_second=float(len(timed) * config['batch_size'] / seconds.sum()),
            gradient_clip_fraction=float(np.mean([step['gradient_norm'] > 1 for step in timed])),
            encoder_gradient_tensors=sorted({step['encoder_gradient_tensors'] for step in timed})))
    return dict(name=arm['name'], run=str(root), batch_size=config['batch_size'],
        requested_steps=config['steps'], completed_steps=report['completed_steps'],
        executed_updates=len(steps), stop_reason=report['stop_reason'], run_seconds=report['seconds'],
        end_to_end_targets_per_second=len(steps) * config['batch_size'] / report['seconds'],
        prepare_seconds=report['prepare_seconds'], phases=rows,
        sampled_peak_process_vram_mib=max(row.get('process_vram_mib', 0) for row in gpu),
        sampled_peak_process_rss_mib=max(row.get('process_rss_mib', 0) for row in gpu),
        sampled_device_utilization_mean=float(np.mean([row['device_gpu_percent'] for row in gpu if 'device_gpu_percent' in row])),
        cached_feature_bytes=report['cached_feature_bytes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    result = dict(scope='Measured step times include batch transfer, forward, backward, optimizer and synchronization; first 50 updates per stage/RI phase excluded. Telemetry peaks are sampled once per second and device utilization includes other GPU applications.',
                  arms=[summarize(arm) for arm in config['arms']])
    Path(config['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
