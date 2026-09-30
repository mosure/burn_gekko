#!/usr/bin/env python3
"""Integrate device-wide board energy without hiding telemetry gaps.

GPU activity is not SM occupancy or useful FLOPs. Board energy includes desktop
activity and command preparation/evaluation; it is not training-kernel energy.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import tomllib


def integrate(rows, duration, max_gap=3.0):
    assert math.isfinite(duration) and duration > 0 and max_gap > 0
    times = [r['elapsed_seconds'] for r in rows]
    assert times and all(math.isfinite(t) and 0 <= t <= duration for t in times)
    assert all(b > a for a, b in zip(times, times[1:])), 'non-monotonic telemetry'
    samples = [(r['elapsed_seconds'], r['device_power_w']) for r in rows
               if 'device_power_w' in r]
    assert samples and all(math.isfinite(w) and w >= 0 for _, w in samples)
    energy = covered = 0.0
    gaps = []
    # Short boundary intervals use the nearest measured power. Long gaps remain
    # unknown, including internal gaps caused by unsuccessful device queries.
    for dt, power in [(samples[0][0], samples[0][1]),
                      (duration - samples[-1][0], samples[-1][1])]:
        if dt <= max_gap:
            energy += dt * power
            covered += dt
        else:
            gaps.append(dt)
    for (ta, pa), (tb, pb) in zip(samples, samples[1:]):
        dt = tb - ta
        if dt <= max_gap:
            energy += dt * (pa + pb) / 2
            covered += dt
        else:
            gaps.append(dt)
    assert covered > 0, 'no supported energy interval'
    activity = [r['device_gpu_percent'] for r in rows if 'device_gpu_percent' in r]
    return dict(observed_board_joules=energy, observed_board_wh=energy / 3600,
                covered_seconds=covered, coverage_fraction=covered / duration,
                unknown_seconds=max(0.0, duration - covered), gaps_seconds=gaps,
                time_weighted_board_power_w=energy / covered,
                median_gpu_activity_percent=statistics.median(activity) if activity else None,
                scope=__doc__.strip(), integration='trapezoids; short endpoints held; long gaps excluded')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    output = Path(config['output'])
    assert output.resolve().is_relative_to(Path('.data').resolve()) and not output.exists()
    results = {}
    for name, item in config['commands'].items():
        ledger = json.loads(Path(item['ledger']).read_text())
        commands = [r for r in ledger['commands'] if r['name'] == item['command']]
        assert len(commands) == 1 and commands[0]['status'] == 'complete'
        command = commands[0]
        rows = [json.loads(line) for line in Path(item['telemetry']).read_text().splitlines()]
        result = integrate(rows, command['elapsed_seconds'], config.get('max_gap_seconds', 3.0))
        result.update(command_seconds=command['elapsed_seconds'], purpose=item['purpose'],
                      ledger_sha256=digest(item['ledger']), telemetry_sha256=digest(item['telemetry']))
        if item.get('training'):
            root = Path(item['training'])
            train = tomllib.loads((root / 'config.toml').read_text())
            updates = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
            report = json.loads((root / 'report.json').read_text())
            assert report['stop_reason'] == 'step_limit' and len(updates) == report['completed_steps'] - report['starting_step']
            assert all(len(r['samples']) == train['batch_size'] for r in updates)
            exposures = len(updates) * train['batch_size']
            result.update(updates=len(updates), target_exposures=exposures,
                          gross_observed_board_joules_per_target= result['observed_board_joules'] / exposures,
                          command_targets_per_second=exposures / command['elapsed_seconds'],
                          median_update_seconds=statistics.median(r['seconds'] for r in updates),
                          training_metrics_sha256=digest(root / 'metrics.jsonl'))
        results[name] = result
    output.write_text(json.dumps(dict(config_sha256=digest(args.config), commands=results), indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
