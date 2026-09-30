#!/usr/bin/env python3
"""CPU-only comparison of saved model flows with evaluation geometry.

Ground-truth geometry is read after model inference. It never changes a model
prediction. Input RGB and reference order must match the immutable room shard.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.numpy import load
import zstandard

from transport_diagnostics import inspect


def project(world, camera, fovy, h, w):
    local = (world - camera[3, :3]) @ camera[:3, :3].T
    z = -local[:, 2]
    fy = h / (2 * np.tan(fovy / 2))
    u = fy * local[:, 0] / np.maximum(z, 1e-12) + w / 2
    v = -fy * local[:, 1] / np.maximum(z, 1e-12) + h / 2
    return u.reshape(h, w), v.reshape(h, w), z.reshape(h, w)


def collect(dataset, samples, split):
    manifest = json.loads((dataset/'manifest.json').read_text())
    entries = {entry['seed']: entry for entry in manifest['scenes'] if entry['split'] == split}
    rows, coverage_rows = [], []
    cached_seed, tensors = None, None
    for path in sorted(samples.glob('room-*-view-*')):
        row, arrays = inspect(path)
        if 'generator_weight' not in row:
            raise ValueError(f'{path}: no saved appearance flow')
        seed, target = row['room_seed'], row['target_view']
        assert seed in entries, 'sample is outside the requested dataset split'
        if seed != cached_seed:
            with (dataset/'raw'/entries[seed]['file']).open('rb') as compressed:
                with zstandard.ZstdDecompressor().stream_reader(compressed) as stream:
                    blob = stream.read(128 * 1024 * 1024 + 1)
            assert len(blob) <= 128 * 1024 * 1024
            tensors = load(blob)
            cached_seed = seed
        rgb = tensors['color'][0, 0]
        depth = tensors['depth'][0, 0, ..., 0]
        aabb = tensors['aabb'][0]
        world = (tensors['position'][0, 0, target] * (aabb[1] - aabb[0]) + aabb[0]).reshape(-1, 3)
        cameras = tensors['world_from_view'][0, 0]
        fovy = tensors['fovy'].reshape(len(rgb))
        _, h, w, _ = rgb.shape
        yy, xx = np.mgrid[:h, :w]
        any_visible = np.zeros((h, w), dtype=bool)
        any_in_range = np.zeros((h, w), dtype=bool)
        u, v, _ = project(world, cameras[target], fovy[target], h, w)
        self_error = np.hypot(u - xx - .5, v - yy - .5)[depth[target] > 0]
        assert len(self_error) and self_error.max() < .1, 'pixel-center or camera convention mismatch'
        np.testing.assert_array_equal(arrays['target'], rgb[target])
        for index, reference_rgb in enumerate(arrays['references']):
            reference = (target + index + 1) % len(rgb)
            np.testing.assert_array_equal(reference_rgb, rgb[reference])
            u, v, z = project(world, cameras[reference], fovy[reference], h, w)
            ix = np.floor(u).astype(int).clip(0, w - 1)
            iy = np.floor(v).astype(int).clip(0, h - 1)
            valid = ((depth[target] > 0) & (z > 0) & (u >= .5) & (u < w - .5)
                     & (v >= .5) & (v < h - .5) & (depth[reference, iy, ix] > 0)
                     & (abs(z - depth[reference, iy, ix]) <= np.maximum(.02, .01 * z)))
            actual = np.stack([u - xx - .5, v - yy - .5], axis=2)
            any_visible |= valid
            any_in_range |= valid & (abs(actual).max(axis=2) <= 64)
            predicted = np.fromfile(path/f'flow-{index}.f32', dtype='<f4').reshape(h, w, 2)
            assert np.isfinite(predicted).all()
            error = np.linalg.norm(predicted - actual, axis=2)
            initial = np.linalg.norm(actual, axis=2)
            for support_name, support in [('all_visible', valid), ('hidden_visible', valid & arrays['hidden'])]:
                if not support.any():
                    continue
                predicted_deltas, true_deltas = [], []
                for axis in [0, 1]:
                    adjacent = (np.take(support, range(support.shape[axis]-1), axis=axis)
                                & np.take(support, range(1, support.shape[axis]), axis=axis))
                    predicted_deltas.append(abs(np.diff(predicted, axis=axis)[adjacent]).ravel())
                    true_deltas.append(abs(np.diff(actual, axis=axis)[adjacent]).ravel())
                predicted_deltas, true_deltas = np.concatenate(predicted_deltas), np.concatenate(true_deltas)
                coherence = {} if not len(true_deltas) else dict(
                    flow_gradient_median_abs=float(np.median(predicted_deltas)),
                    true_flow_gradient_median_abs=float(np.median(true_deltas)),
                    flow_gradient_p90_abs=float(np.quantile(predicted_deltas,.9)),
                    true_flow_gradient_p90_abs=float(np.quantile(true_deltas,.9)))
                rows.append(dict(room_seed=seed, target_view=target, reference_view=reference,
                    support=support_name, pixels=int(support.sum()),
                    endpoint_error_pixels=float(error[support].mean()),
                    identity_endpoint_error_pixels=float(initial[support].mean()),
                    fraction_within_3_pixels=float((error[support] < 3).mean()),
                    fraction_gt_beyond_64_per_axis=float((abs(actual[support]).max(axis=1) > 64).mean()),
                    max_self_reprojection_pixels=float(self_error.max()), **coherence))
        for support_name, support in [('all_pixels', np.ones((h, w), dtype=bool)),
                                      ('hidden_pixels', arrays['hidden'])]:
            visible = int((any_visible & support).sum())
            within = int((any_in_range & support).sum())
            coverage_rows.append(dict(room_seed=seed, target_view=target, support=support_name,
                pixels=int(support.sum()), any_visible_reference_pixels=visible,
                any_in_range_reference_pixels=within,
                visible_fraction=float(visible / support.sum()),
                in_range_fraction=float(within / support.sum()),
                visible_but_all_references_out_of_range_fraction=float((visible - within) / support.sum())))
    assert rows, 'no matching model flow exports'
    return dict(dataset_id=manifest['dataset_id'], split=split, samples=str(samples),
                artifact_kind='evaluation_geometry_diagnostic_not_training_supervision', rows=rows,
                coverage_rows=coverage_rows,
                coverage_means={support: {key: float(np.mean([row[key] for row in coverage_rows if row['support'] == support]))
                    for key in ['visible_fraction', 'in_range_fraction', 'visible_but_all_references_out_of_range_fraction']}
                    for support in ['all_pixels', 'hidden_pixels']},
                means={support: {key: float(np.mean([row[key] for row in rows if row['support'] == support and key in row]))
                    for key in ['endpoint_error_pixels', 'identity_endpoint_error_pixels',
                                'fraction_within_3_pixels', 'fraction_gt_beyond_64_per_axis',
                                'flow_gradient_median_abs', 'true_flow_gradient_median_abs',
                                'flow_gradient_p90_abs', 'true_flow_gradient_p90_abs']}
                    for support in ['all_visible', 'hidden_visible']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'validation', 'test'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = collect(args.dataset, args.samples, args.split)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['means'], indent=2))


if __name__ == '__main__':
    main()
