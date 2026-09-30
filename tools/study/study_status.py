#!/usr/bin/env python3
"""Read progress from an active or completed study without touching its process."""
import argparse
import json
from pathlib import Path
import statistics


def read(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def status(run, budget_path):
    result = dict(run=str(run))
    metrics = run / 'metrics.jsonl'
    if metrics.exists():
        with metrics.open('rb') as stream:
            stream.seek(max(0, metrics.stat().st_size - 262144))
            lines = stream.read().splitlines()
        steps = []
        for line in lines:
            try:
                steps.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # Seek may start within a row; the writer may be mid-append.
        if steps:
            recent = steps[-100:]
            # RGB pilots and the latent trainer use different metric names.
            loss_key = 'total' if 'total' in recent[-1] else 'loss'
            ri_key = 'ri' if 'ri' in recent[-1] else 'ri_loss'
            result.update(step=steps[-1]['step'], stage=steps[-1]['stage'],
                recent_updates=len(recent), median_step_seconds=statistics.median(s['seconds'] for s in recent),
                mean_loss=statistics.mean(s[loss_key] for s in recent),
                mean_ri_loss=statistics.mean(s[ri_key] for s in recent),
                clip_fraction=statistics.mean(s['gradient_norm'] > 1 for s in recent))
            if loss_key == 'total':
                result['mean_cross_latent_mse'] = statistics.mean(s['cross'] for s in recent)
                result['encoder_gradient_tensors'] = steps[-1]['encoder_gradient_tensors']
    probes = read(run / 'probes.json')
    if probes:
        result['last_probe'] = probes[-1]
    budget = read(budget_path)
    if budget:
        active = budget.get('running', {}).get('command_elapsed_seconds', 0)
        result['approx_study_seconds'] = budget['command_seconds'] + active
        result['approx_remaining_seconds'] = budget['ceiling_seconds'] - result['approx_study_seconds']
        result['active_command'] = budget.get('running', {}).get('command')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--budget', type=Path, default=Path('.data/pilot-06/budget.json'))
    args = parser.parse_args()
    print(json.dumps(status(args.run, args.budget), indent=2))
