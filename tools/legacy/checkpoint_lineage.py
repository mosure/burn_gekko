#!/usr/bin/env python3
"""Audit own-weight ancestry and count actual optimizer inputs up to a checkpoint."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import tomllib

REVIEWED_VJEPA_ID = 'c408f68dd18a38824d0fa1d615e6f9f9f04f111d71a6f7c846f41dc187a8795f'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(checkpoint):
    checked, active, inputs, checkpoints = set(), set(), {}, []

    def visit(path):
        path = path.resolve()
        assert path not in active, 'cyclic checkpoint lineage'
        if path in checked:
            return
        assert len(active) < 32, 'checkpoint lineage too deep'
        active.add(path)
        root = path.parent
        metadata = read(path / 'metadata.json')
        provenance = read(root / 'provenance.json')
        config = tomllib.loads((root / 'config.toml').read_text())
        assert sha(path / 'model.mpk') == metadata['model_sha256']
        assert not metadata['noncommercial_weight_dependencies']
        assert not provenance['noncommercial_weight_dependencies']
        latent = provenance.get('task') == 'fixed_vjepa21_latent_prediction'
        if latent:
            assert provenance['teacher_id'] == REVIEWED_VJEPA_ID, 'unreviewed latent teacher'
            assert provenance['teacher_update'] == 'never'
            encoder_key = 'teacher_id'
        else:
            assert provenance['teacher'] is None
            assert provenance['evaluation_split'] is None, 'expected a training checkpoint'
            encoder_key = 'encoder_id'
        identity_keys = ['dataset_id', encoder_key, 'identity', 'backend']
        for key in identity_keys:
            assert metadata[key] == provenance[key]
        manifest = read(Path(config['dataset']) / 'manifest.json')
        assert manifest['dataset_id'] == metadata['dataset_id']
        eligible = [s['seed'] for s in manifest['scenes'] if s['split'] == 'train'][:config['train_rooms']]
        if latent:
            assert eligible == provenance['training_room_seeds'], 'training room provenance mismatch'
        allowed = {(seed, view) for seed in eligible for view in range(manifest['config']['cameras'])}
        rows = [json.loads(line) for line in (root / 'metrics.jsonl').read_text().splitlines()]
        rows = [row for row in rows if row['step'] <= metadata['completed_steps']]
        assert rows and rows[-1]['step'] == metadata['completed_steps']
        assert [r['step'] for r in rows] == list(range(rows[0]['step'], rows[-1]['step'] + 1))
        for row in rows:
            samples = [tuple(s) for s in row['samples']]
            assert len(samples) == config['batch_size'] and all(s in allowed for s in samples)
            key = (str(root), row['step'])
            value = (metadata['dataset_id'], samples)
            assert key not in inputs or inputs[key] == value
            inputs[key] = value
        resumed = provenance.get('resume')
        if resumed:
            parent_path = Path(resumed)
            parent = read(parent_path / 'metadata.json')
            assert parent['completed_steps'] == rows[0]['step'] - 1
            for key in identity_keys:
                assert parent[key] == metadata[key]
            visit(parent_path)
        else:
            assert rows[0]['step'] == 1
        warm = config.get('warm_start') if latent else metadata.get('warm_start')
        if latent:
            assert provenance.get('warm_start') == warm
            assert metadata.get('weight_ancestors', []) == ([warm] if warm else [])
            if warm:
                parent_path = Path(warm['checkpoint'])
                assert read(parent_path / 'metadata.json')['model_sha256'] == warm['model_sha256']
                visit(parent_path)
        elif warm:
            parent_path = Path(warm['checkpoint'])
            parent = read(parent_path / 'metadata.json')
            assert sha(parent_path / 'metadata.json') == warm['source_metadata_sha256']
            assert parent['model_sha256'] == warm['model_sha256']
            assert warm['optimizer_reset']
            visit(parent_path)
        checkpoints.append(dict(path=str(path), model_sha256=metadata['model_sha256'],
                                completed_steps=metadata['completed_steps'], resume=resumed,
                                warm_start=warm, task=provenance.get('task', 'legacy_rgb'),
                                encoder_or_teacher_id=metadata[encoder_key]))
        active.remove(path)
        checked.add(path)

    visit(checkpoint)
    by_run = collections.defaultdict(lambda: dict(executed_updates=0, target_exposures=0))
    rooms, targets = set(), set()
    for (run, step), (dataset, samples) in inputs.items():
        by_run[run]['executed_updates'] += 1
        by_run[run]['target_exposures'] += len(samples)
        rooms.update((dataset, seed) for seed, _ in samples)
        targets.update((dataset, seed, view) for seed, view in samples)
    return dict(checkpoint=str(checkpoint), checkpoints=checkpoints, runs=dict(by_run),
                executed_updates=len(inputs), target_exposures=sum(r['target_exposures'] for r in by_run.values()),
                unique_training_rooms=len(rooms), unique_training_target_views=len(targets),
                nontraining_optimizer_inputs=0, noncommercial_weight_dependencies=[],
                scope='Only logged optimizer updates through each ancestor checkpoint; exact-resume prefixes counted once.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to(Path('.data').resolve()) and not args.output.exists()
    result = audit(args.checkpoint)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ['checkpoints', 'runs']}, indent=2))


if __name__ == '__main__':
    main()
