#!/usr/bin/env python3
"""Measure completed RGB probes during training, without changing model selection."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from transport_diagnostics import inspect


def collect(run, cached=()):
    records = []
    cached = {row.get('source_directory'): row for row in cached}
    provenance = json.loads((run / 'provenance.json').read_text())
    resume = provenance.get('resume')
    initial_step = json.loads((Path(resume) / 'metadata.json').read_text())['completed_steps'] if resume else 0
    for path in sorted([*run.glob('step-*'), *run.glob('train-step-*'), *run.glob('synthetic-step-*')]):
        if not (path / 'evaluation.json').exists():
            continue
        evaluation_text = (path / 'evaluation.json').read_text()
        evaluation_hash = hashlib.sha256(evaluation_text.encode()).hexdigest()
        prior = cached.get(path.name)
        if prior and prior.get('source_evaluation_sha256') == evaluation_hash:
            records.append(prior)
            continue
        label = path.name.removeprefix('train-').removeprefix('synthetic-').removeprefix('step-')
        rows = [inspect(d)[0] for d in sorted(path.glob('room-*-view-*'))]
        if not rows:
            continue
        evaluation = json.loads(evaluation_text)
        metric_keys = ['mse', 'psnr', 'edge_cosine', 'edge_energy', 'interior_edge_cosine',
                       'interior_edge_energy', 'seam_edge_energy']
        if 'generator_weight' in rows[0]:
            metric_keys += ['generator_weight', 'generated_mse', 'transported_mse', 'mean_flow_pixels']
        if 'coarse_flow_curvature' in rows[0]:
            metric_keys += ['coarse_flow_curvature']
        records.append(dict(source_directory=path.name, source_evaluation_sha256=evaluation_hash,
            step=initial_step if label == 'initial' else int(label),
            split='training_probe' if path.name.startswith('train-') else 'synthetic_probe' if path.name.startswith('synthetic-') else 'validation_probe',
            exported_targets=len(rows), evaluated_targets=len(evaluation['targets']),
            all_target_mse=evaluation['mean_hidden_rgb_mse'],
            all_target_monocular_mse=float(np.mean([r['monocular_hidden_rgb_mse'] for r in evaluation['targets']])),
            exported_metrics={k: float(np.mean([r[k] for r in rows])) for k in metric_keys}))
    return sorted(records, key=lambda r: (r['step'], r['split']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fingerprint = hashlib.sha256(Path(__file__).read_bytes()
        + Path(__file__).with_name('transport_diagnostics.py').read_bytes()).hexdigest()
    prior = json.loads(args.output.read_text()) if args.output.exists() else {}
    cached = prior.get('probes', []) if prior.get('analysis_sha256') == fingerprint and prior.get('run') == str(args.run) else []
    result = dict(run=str(args.run), analysis_sha256=fingerprint,
        scope='fixed probe rooms only; detailed metrics on exported view zero, MSE on all evaluated views; completed progress summaries cached by evaluation hash and analysis source; final report recomputes raw metrics',
        probes=collect(args.run, cached))
    stage = args.output.with_suffix('.partial')
    stage.write_text(json.dumps(result, indent=2) + '\n')
    stage.replace(args.output)
    print(json.dumps(result['probes'][-2:], indent=2))


if __name__ == '__main__':
    main()
