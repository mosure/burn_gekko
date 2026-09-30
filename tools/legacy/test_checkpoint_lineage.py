"""Ancestry must retain license checks and count exact-resume inputs only once."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from checkpoint_lineage import audit, REVIEWED_VJEPA_ID


class CheckpointLineage(unittest.TestCase):
    def setUp(self):
        root = Path('.data/tests'); root.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=root)
        self.root = Path(self.temp.name).resolve()
        self.data = self.root / 'dataset'; self.data.mkdir()
        (self.data/'manifest.json').write_text(json.dumps(dict(dataset_id='fixture', scenes=[dict(seed=11, split='train'), dict(seed=22, split='validation')], config=dict(cameras=2))))

    def tearDown(self):
        self.temp.cleanup()

    def checkpoint(self, name, steps, warm=None, resume=None, latent=True):
        root = self.root/name; path = root/'final'; path.mkdir(parents=True)
        (path/'model.mpk').write_bytes(name.encode())
        model_hash = hashlib.sha256((path/'model.mpk').read_bytes()).hexdigest()
        identity = 'second-phase' if warm else 'first-phase'
        key = 'teacher_id' if latent else 'encoder_id'
        identifier = REVIEWED_VJEPA_ID if latent else 'legacy-fixture'
        common = dict(dataset_id='fixture', identity=identity, backend='fixture', noncommercial_weight_dependencies=[])
        common[key] = identifier
        metadata = dict(common, completed_steps=max(steps), model_sha256=model_hash)
        provenance = dict(common, resume=str(resume) if resume else None)
        ancestor = dict(checkpoint=str(warm), model_sha256=json.loads((warm/'metadata.json').read_text())['model_sha256']) if warm else None
        if latent:
            metadata['weight_ancestors'] = [ancestor] if ancestor else []
            provenance.update(task='fixed_vjepa21_latent_prediction', teacher_update='never', warm_start=ancestor, training_room_seeds=[11])
        else:
            provenance.update(teacher=None, evaluation_split=None)
        config = f'dataset = {json.dumps(str(self.data))}\ntrain_rooms = 1\nbatch_size = 1\n'
        if ancestor:
            config += f'\n[warm_start]\ncheckpoint = {json.dumps(ancestor["checkpoint"])}\nmodel_sha256 = {json.dumps(ancestor["model_sha256"])}\n'
        (root/'config.toml').write_text(config)
        (root/'provenance.json').write_text(json.dumps(provenance))
        (path/'metadata.json').write_text(json.dumps(metadata))
        (root/'metrics.jsonl').write_text(''.join(json.dumps(dict(step=step, samples=[[11, step % 2]]))+'\n' for step in steps))
        return path

    def test_latent_warm_start_and_resume_count_shared_ancestor_once(self):
        parent = self.checkpoint('parent', [1, 2])
        phase = self.checkpoint('phase', [1, 2], warm=parent)
        resumed = self.checkpoint('resumed', [3, 4], warm=parent, resume=phase)
        result = audit(resumed)
        self.assertEqual(result['executed_updates'], 6)
        self.assertEqual(result['target_exposures'], 6)
        self.assertEqual(result['unique_training_rooms'], 1)
        self.assertEqual(len(result['checkpoints']), 3)

    def test_latent_rejects_noncommercial_ancestor(self):
        parent = self.checkpoint('parent', [1, 2])
        child = self.checkpoint('child', [1], warm=parent)
        meta = json.loads((parent/'metadata.json').read_text())
        meta['noncommercial_weight_dependencies'] = ['forbidden-fixture']
        (parent/'metadata.json').write_text(json.dumps(meta))
        with self.assertRaises(AssertionError):
            audit(child)

    def test_legacy_rgb_resume_remains_supported(self):
        parent = self.checkpoint('parent', [1, 2], latent=False)
        child = self.checkpoint('resumed', [3, 4], resume=parent, latent=False)
        self.assertEqual(audit(child)['executed_updates'], 4)


if __name__ == '__main__':
    unittest.main()
