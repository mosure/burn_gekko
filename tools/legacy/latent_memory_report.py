#!/usr/bin/env python3
"""Compare bounded native encoder capture audits and their GPU telemetry."""
import argparse
import json
from pathlib import Path
import tomllib
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    config = tomllib.loads(parser.parse_args().config.read_text())
    runner = Path(config['runner'])
    ledger = read(runner / 'ledger.json')
    assert ledger['stop_reason'] == 'plan_complete'
    result = {'status': 'isolated_memory_diagnostic', 'arms': {}, 'summary': {}}
    losses = []
    for arm in config['arms']:
        name, path = arm['name'], Path(arm['output'])
        report = read(path / 'report.json')
        steps = [json.loads(line) for line in (path / 'metrics.jsonl').read_text().splitlines()]
        telemetry = [json.loads(line) for line in (runner / f'{name}-gpu.jsonl').read_text().splitlines()]
        command = next(row for row in ledger['commands'] if row['name'] == name)
        assert command['status'] == 'complete' and len(steps) == report['steps'] == 200
        assert report['batch'] == 48 and report['cleanup_calls'] == 0
        losses.append(np.array([s['loss'] for s in steps]))
        assert np.isfinite(losses[-1]).all()
        # The runner samples at one-second intervals. Approximate loop start by
        # subtracting the measured loop duration from total process duration.
        offset = command['elapsed_seconds'] - report['seconds']
        ends = np.cumsum([s['seconds'] for s in steps]) + offset
        samples = [(np.interp(r['elapsed_seconds'], ends, np.arange(1, 201)), r['process_vram_mib'])
                   for r in telemetry if 'process_vram_mib' in r and ends[39] <= r['elapsed_seconds'] <= ends[-1]]
        assert len(samples) >= 10
        x, y = np.array(samples).T
        slope = float(np.polyfit(x, y, 1)[0])
        stats = dict(peak_process_vram_mib=max(r['process_vram_mib'] for r in telemetry if 'process_vram_mib' in r),
                     warm_initial_vram_mib=float(np.median(y[:5])),
                     final_vram_mib=float(np.median(y[-5:])),
                     growth_mib_per_update=slope, warm_median_seconds_per_update=float(np.median([s['seconds'] for s in steps[40:]])),
                     command_seconds=command['elapsed_seconds'])
        result['summary'][name] = stats
        result['arms'][name] = dict(report=report, metrics=str(path / 'metrics.jsonl'), telemetry=str(runner / f'{name}-gpu.jsonl'))
    difference = np.abs(losses[0] - losses[1])
    result['summary']['max_absolute_loss_difference'] = float(difference.max())
    result['summary']['final_absolute_loss_difference'] = float(difference[-1])
    result['measurement_limits'] = 'Process VRAM sampled every second; allocator reservations included. Slopes exclude first 40 updates. Loop alignment approximated from process and loop durations.'
    output = Path(config['output'])
    assert output.resolve().is_relative_to(Path('.data').resolve())
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['summary'], indent=2))


if __name__ == '__main__':
    main()
