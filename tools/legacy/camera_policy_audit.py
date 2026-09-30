#!/usr/bin/env python3
"""Measure camera spacing from immutable capture tensors, without model inference."""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load
import zstandard


def collect(dataset, split):
    manifest = json.loads((dataset/'manifest.json').read_text())
    assert manifest['generator'].split(';')[0] == 'bevy_zeroverse=0.25.0', 'policy formula is version-specific'
    rows = []
    for scene in manifest['scenes']:
        if scene['split'] != split:
            continue
        with (dataset/'raw'/scene['file']).open('rb') as compressed:
            with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
                blob = stream.read(128 * 1024 * 1024 + 1)
        assert len(blob) <= 128 * 1024 * 1024
        tensors = load(blob)
        camera = tensors['world_from_view'][0, 0]
        centres = camera[:, 3, :3]
        assert centres.shape == (manifest['config']['cameras'], 3)
        for left in range(len(centres)):
            for right in range(left+1, len(centres)):
                rows.append(dict(room_seed=scene['seed'], left=left, right=right,
                    baseline_metres=float(np.linalg.norm(centres[left]-centres[right]))))
    assert rows
    distance = np.array([row['baseline_metres'] for row in rows])
    baseline = manifest['config']['camera_baseline']
    policy = None if baseline is None else dict(
        min_overlap=.65-.55*baseline**.6, min_pair_separation_metres=.04+.46*baseline,
        min_reference_separation_metres=.06+2.30*baseline,
        max_reference_separation_metres=.35+9.65*baseline,
        min_horizontal_spread=.20+.10*baseline)
    reference_distance = np.array([row['baseline_metres'] for row in rows if row['left'] == 0])
    if policy:
        assert distance.min() >= policy['min_pair_separation_metres']-1e-4
        assert reference_distance.min() >= policy['min_reference_separation_metres']-1e-4
        assert reference_distance.max() <= policy['max_reference_separation_metres']+1e-4
    return dict(dataset_id=manifest['dataset_id'], split=split, rooms=len({r['room_seed'] for r in rows}),
        baseline_control=baseline, control_units='dimensionless [0,1], not metres',
        published_policy=policy, policy_source='bevy_zeroverse 0.25.0 cameras/baseline.rs; proposal constraints, not measured pixel visibility',
        measured_pairwise_metres=dict(minimum=float(distance.min()), median=float(np.median(distance)),
                                     p95=float(np.quantile(distance,.95)), maximum=float(distance.max())),
        measured_camera_zero_reference_metres=dict(minimum=float(reference_distance.min()), median=float(np.median(reference_distance)),
            p95=float(np.quantile(reference_distance,.95)),maximum=float(reference_distance.max())),
        pairs=rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--split',choices=['train','validation','test'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=collect(args.dataset,args.split)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='pairs'},indent=2))


if __name__=='__main__':
    main()
