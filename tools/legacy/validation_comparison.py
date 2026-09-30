#!/usr/bin/env python3
"""Compare complete validation exports, with paired room-bootstrap intervals."""
import argparse
import json
from pathlib import Path
import tomllib

from e2e_report import ci, exports


def compare(config):
    arms = {}
    masks = {}
    for arm in config['arms']:
        root = Path(arm['run'])
        resolved = tomllib.loads((root / 'config.toml').read_text())
        manifest = json.loads((Path(resolved['dataset']) / 'manifest.json').read_text())
        measured, paths = exports(root / 'validation')
        expected = {(scene['seed'], view) for scene in manifest['scenes']
                    if scene['split'] == 'validation' for view in range(manifest['config']['cameras'])}
        assert set(paths) == expected, 'full validation required'
        masks[arm['name']] = {key: json.loads((path / 'sample.json').read_text())['visible_patch_ids']
                             for key, path in paths.items()}
        for key in ['generator_weight', 'generated_mse', 'transported_mse', 'mean_flow_pixels',
                    'coarse_flow_curvature']:
            if key in measured['rows'][0]:
                measured['metrics'][key] = ci(measured['rows'], key)
        metadata = json.loads((root / 'final' / 'metadata.json').read_text())
        assert not metadata['noncommercial_weight_dependencies']
        measured.update(run=str(root), model_sha256=metadata['model_sha256'],
                        completed_steps=metadata['completed_steps'], config=resolved)
        if arm.get('geometry'):
            geometry = json.loads(Path(arm['geometry']).read_text())
            assert geometry['split'] == 'validation'
            assert geometry['dataset_id'] == manifest['dataset_id']
            assert Path(geometry['samples']).resolve() == (root / 'validation').resolve()
            assert {(r['room_seed'], r['target_view']) for r in geometry['rows']} == expected
            measured['flow_geometry'] = geometry['means']
        arms[arm['name']] = measured
    differences = []
    for pair in config.get('comparisons', []):
        control, candidate = (arms[pair[key]] for key in ['control', 'candidate'])
        a = {(r['room_seed'], r['target_view']): r for r in control['rows']}
        b = {(r['room_seed'], r['target_view']): r for r in candidate['rows']}
        assert set(a) == set(b)
        assert control['config']['dataset'] == candidate['config']['dataset']
        assert control['config']['mask_ratio'] == candidate['config']['mask_ratio']
        assert masks[pair['control']] == masks[pair['candidate']], 'evaluation masks differ'
        keys = sorted(set(control['metrics']) & set(candidate['metrics']))
        rows = [dict(room_seed=key[0], target_view=key[1],
                     **{metric: b[key][metric] - a[key][metric] for metric in keys}) for key in sorted(a)]
        differences.append(dict(control=pair['control'], candidate=pair['candidate'],
                                convention='candidate minus control, same room and target view',
                                metrics={key: ci(rows, key) for key in keys}))
    return dict(scope='All validation rooms and target views; raw hidden RGB; 2000 room-bootstrap replicates. No test inputs.',
                arms=arms, paired_differences=differences)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    result = compare(config)
    Path(config['output']).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({name: {key: value['mean'] for key, value in arm['metrics'].items()}
                      for name, arm in result['arms'].items()}, indent=2))


if __name__ == '__main__':
    main()
