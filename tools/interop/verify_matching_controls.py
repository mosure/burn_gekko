#!/usr/bin/env python3
"""Compare saved readouts without geometry, fitting, or additional GPU work."""
import argparse
import hashlib
import json
from pathlib import Path
import tomllib

import numpy as np


def read_export(path):
    methods = {}
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        pair = row['pair'] if 'pair' in row else (row['sequence'], row['target'], row['reference'])
        target = methods.setdefault(row['method'], {})
        assert pair not in target, 'duplicate readout pair'
        assert len(row['grid']) == 2 and all(isinstance(x,int) and x > 0 for x in row['grid'])
        n = int(np.prod(row['grid']))
        indices = np.asarray(row['indices'])
        mutual = np.asarray(row['mutual'])
        assert indices.shape == mutual.shape == (n,)
        assert np.issubdtype(indices.dtype, np.integer) and mutual.dtype == np.bool_
        assert (indices >= 0).all() and (indices < n).all()
        target[pair] = indices, mutual
    assert methods, 'empty export'
    return methods


def compare(first, second):
    assert first, 'empty readout comparison'
    assert first.keys() == second.keys(), 'readouts contain different pairs'
    queries = indices_changed = mutual_changed = max_pair_changes = 0
    pairs_changed = 0
    for key, (a, ma) in first.items():
        b, mb = second[key]
        assert a.shape == b.shape, 'readouts use different grids'
        count = int(np.sum(a != b))
        queries += len(a)
        indices_changed += count
        mutual_changed += int(np.sum(ma != mb))
        pairs_changed += int(count > 0)
        max_pair_changes = max(max_pair_changes, count)
    return dict(pairs=len(first), queries=queries, indices_changed=indices_changed,
                mutual_changed=mutual_changed, pairs_with_changed_indices=pairs_changed,
                max_changed_indices_in_one_pair=max_pair_changes,
                changed_index_fraction=indices_changed/queries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text())
    output = Path(config['output'])
    assert output.resolve().is_relative_to(Path('.data').resolve()) and not output.exists()
    exports = {name: read_export(path) for name, path in config['exports'].items()}
    result = dict(config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                  exports={name:dict(path=path,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
                           for name,path in config['exports'].items()}, comparisons={})
    passed = True
    for item in config['comparisons']:
        get = lambda key: exports[key.split('/',1)[0]][key.split('/',1)[1]]
        comparison = compare(get(item['first']), get(item['second']))
        comparison.update(first=item['first'],second=item['second'],required_identical=item.get('identical', True))
        comparison['identical'] = comparison['indices_changed'] == comparison['mutual_changed'] == 0
        passed &= comparison['identical'] or not comparison['required_identical']
        result['comparisons'][item['name']] = comparison
    result['passed'] = passed
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=passed,comparisons=result['comparisons']),indent=2))
    if not passed:
        raise SystemExit('declared identical controls differ; inspect the saved receipt')


if __name__ == '__main__':
    main()
