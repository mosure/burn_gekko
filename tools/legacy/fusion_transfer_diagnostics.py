#!/usr/bin/env python3
"""Development-only transfer contrasts and error stratified by true motion.

All predictions are already fixed. Homographies are used only by this CPU report.
Primary HP-240 sequence-mean metrics remain in hpatches_score.py. The motion
strata below pool valid pixels and are explicitly a different aggregation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np

from hpatches_score import dense_map, paired_ci, truth_map


def truth_nearest_grid(gt, grid=16):
    """Quantize GT at patch centres; diagnostic only, never an inference input.

    This is not a mathematical lower bound after displacement interpolation:
    nearby discrete choices can compensate for one another. Out-of-frame centre
    projections are clipped to the reference grid, as a real match must be.
    """
    size = gt.shape[0]
    assert gt.shape == (size, size, 2)
    centres = (np.arange(grid) + .5) * size / grid - .5
    assert np.array_equal(centres, centres.astype(int)), 'control requires integer patch centres'
    xy = gt[centres.astype(int)[:, None], centres.astype(int)[None, :]]
    assert np.isfinite(xy).all()
    indices = np.rint(np.clip((xy + .5) * grid / size - .5, 0, grid-1)).astype(int)
    return (indices[..., 1] * grid + indices[..., 0]).ravel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    output = Path(config['output'])
    assert output.resolve().is_relative_to(Path('.data').resolve()) and not output.exists()
    score = json.loads(Path(config['score']).read_text())
    assert score['evaluation_use'] == 'development'
    result = dict(scope=__doc__, score_sha256=hashlib.sha256(Path(config['score']).read_bytes()).hexdigest(), contrasts={}, precision_contrasts={}, models={})
    rows = []
    for model in config['models']:
        for row in score['models'][model]['rows']:
            rows.append(dict(row, method=f"{model}/{row['method']}"))
    for contrast in config.get('contrasts', []):
        result['contrasts'][contrast['name']] = paired_ci(rows, contrast['first'], contrast['second'], 'viewpoint')
        result['precision_contrasts'][contrast['name']] = dict(
            **paired_ci(rows, contrast['first'], contrast['second'], 'viewpoint', metric='pck3'),
            metric='PCK3 fraction, first minus second; positive favors first')
    root = Path(config['dataset'])
    assert hashlib.sha256((root/'images.json').read_bytes()).hexdigest() == config['images_sha256']
    assert hashlib.sha256((root/'homographies.npz').read_bytes()).hexdigest() == score['geometry_sha256']
    manifest = json.loads((root/'images.json').read_text())
    geometry = np.load(root/'homographies.npz')
    yy, xx = np.mgrid[:240, :240]
    query = np.stack([xx, yy], -1)
    truth = {}
    oracle = []
    for sequence in manifest['sequences']:
        name = sequence['name']
        if not name.startswith('v_'):
            continue
        for target in range(2, 7):
            gt, valid = truth_map(geometry[f'{name}_{target}'], sequence['views'][0]['original_hw'], sequence['views'][target-1]['original_hw'])
            motion = np.linalg.norm(gt-query, axis=-1)
            masks = [valid & (motion < 8), valid & (motion >= 8) & (motion < 32), valid & (motion >= 32)]
            assert np.array_equal(np.sum(masks, axis=0), valid.astype(int))
            truth[name, target] = gt, masks
            error = np.linalg.norm(dense_map(truth_nearest_grid(gt))-gt, axis=-1)[valid]
            oracle.append(dict(sequence=name, target=target, aepe=float(error.mean()),
                               pck3=float((error <= 3).mean())))
    result['geometry_only_grid_control'] = dict(
        protocol='GT projected at 16x16 patch centres, nearest reference cell, same bilinear displacement readout. Uses labels, unavailable to the model; not a rigorous lower bound.',
        aggregation='mean over 295 viewpoint pairs; five pairs per sequence',
        aepe=float(np.mean([x['aepe'] for x in oracle])),
        pck3=float(np.mean([x['pck3'] for x in oracle])), rows=oracle)
    for model in config['models']:
        predictions = Path(config['export']) / f'{model}.jsonl'
        assert hashlib.sha256(predictions.read_bytes()).hexdigest() == score['models'][model]['prediction_sha256']
        groups = {method: dict(sums=np.zeros(3), counts=np.zeros(3, dtype=np.int64), pck3=np.zeros(3, dtype=np.int64), same_patch=[], pairs=0) for method in config['methods']}
        for line in predictions.read_text().splitlines():
            row = json.loads(line)
            if not row['sequence'].startswith('v_') or row['method'] not in groups:
                continue
            group = groups[row['method']]
            gt, masks = truth[row['sequence'], row['target']]
            error = np.linalg.norm(dense_map(row['indices'])-gt, axis=-1)
            group['pairs'] += 1
            group['same_patch'].append(float(np.mean(np.asarray(row['indices']) == np.arange(256))))
            for index, mask in enumerate(masks):
                group['counts'][index] += mask.sum()
                group['sums'][index] += error[mask].sum()
                group['pck3'][index] += (error[mask] <= 3).sum()
        methods = {}
        for method, group in groups.items():
            assert group['pairs'] == 295 and (group['counts'] > 0).all()
            methods[method] = dict(exact_same_patch_fraction=float(np.mean(group['same_patch'])), strata={})
            for index, label in enumerate(['0_to_8_px', '8_to_32_px', '32_px_or_more']):
                n = int(group['counts'][index])
                methods[method]['strata'][label] = dict(valid_pixels=n, aepe=float(group['sums'][index]/n), pck3=float(group['pck3'][index]/n))
        result['models'][model] = methods
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(contrasts=result['contrasts'], output=str(output)), indent=2))


if __name__ == '__main__':
    main()
