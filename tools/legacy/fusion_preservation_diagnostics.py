#!/usr/bin/env python3
"""Measure which encoder matches fusion corrects or damages on development data.

Uses fixed HP-240 patch centres, not interpolated dense pixels. These diagnostics
do not replace the primary benchmark metric or supply labels to inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np

from hpatches_score import truth_map


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    output = Path(config['output'])
    assert output.resolve().is_relative_to(Path('.data').resolve()) and not output.exists()
    score = json.loads(Path(config['score']).read_text())
    assert score['evaluation_use'] == 'development'
    root = Path(config['dataset'])
    assert sha(root/'images.json') == config['images_sha256']
    assert sha(root/'homographies.npz') == score['geometry_sha256']
    images = json.loads((root/'images.json').read_text())
    geometry = np.load(root/'homographies.npz')
    centres = np.arange(16)*15+7
    truth = {}
    for sequence in images['sequences']:
        if not sequence['name'].startswith('v_'):
            continue
        for target in range(2, 7):
            gt, valid = truth_map(geometry[f'{sequence["name"]}_{target}'],
                                 sequence['views'][0]['original_hw'],
                                 sequence['views'][target-1]['original_hw'])
            truth[sequence['name'], target] = (
                gt[centres[:, None], centres[None, :]].reshape(-1, 2),
                valid[centres[:, None], centres[None, :]].ravel())
    assert len(truth) == 295
    result = dict(protocol=__doc__, coordinate_size=240, grid=16, correctness_radius_px=15,
                  aggregation='per-pair means over valid patch centres; all five views of all 59 viewpoint sequences',
                  score_sha256=sha(config['score']), config_sha256=sha(args.config), models={})
    for model in config['models']:
        file = Path(config['export']) / f'{model}.jsonl'
        assert sha(file) == score['models'][model]['prediction_sha256']
        exports = {(r['sequence'], r['target'], r['method']):np.asarray(r['indices'])
                   for line in file.read_text().splitlines() if (r:=json.loads(line))['sequence'].startswith('v_')}
        result['models'][model] = {}
        for comparison in config['comparisons']:
            rows = []
            for pair, (gt, valid) in truth.items():
                assert valid.any()
                a = exports[*pair, comparison['baseline']][valid]
                b = exports[*pair, comparison['candidate']][valid]
                xy = lambda ids:np.stack([ids % 16, ids // 16], -1)*15+7
                old = np.linalg.norm(xy(a)-gt[valid], axis=-1)
                new = np.linalg.norm(xy(b)-gt[valid], axis=-1)
                corrected = np.mean((old > 15) & (new <= 15))
                damaged = np.mean((old <= 15) & (new > 15))
                assert abs(corrected-damaged-(np.mean(new <= 15)-np.mean(old <= 15))) < 1e-12
                rows.append(dict(sequence=pair[0], target=pair[1], valid_patches=int(valid.sum()),
                    changed_fraction=float(np.mean(a != b)), encoder_epe=float(old.mean()),
                    candidate_epe=float(new.mean()), encoder_pck15=float(np.mean(old <= 15)),
                    candidate_pck15=float(np.mean(new <= 15)), corrected_fraction=float(corrected),
                    damaged_fraction=float(damaged)))
            metrics = ['changed_fraction','encoder_epe','candidate_epe','encoder_pck15',
                       'candidate_pck15','corrected_fraction','damaged_fraction']
            result['models'][model][comparison['name']] = dict(comparison=comparison,
                summary={key:float(np.mean([r[key] for r in rows])) for key in metrics}, rows=rows)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({m:{name:v['summary'] for name,v in comparisons.items()} for m,comparisons in result['models'].items()}, indent=2))


if __name__ == '__main__':
    main()
