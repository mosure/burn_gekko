"""Live monitoring accepts both trainer schemas and an incomplete final row."""
import json
from pathlib import Path
import tempfile
import unittest

from study_status import status


class StudyStatus(unittest.TestCase):
    def test_rgb_and_latent_logs_with_partial_append(self):
        root = Path('.data/tests')
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as directory:
            run = Path(directory)
            budget = run / 'budget.json'
            budget.write_text(json.dumps(dict(ceiling_seconds=1000, command_seconds=200,
                running=dict(command='active', command_elapsed_seconds=30, completed_command_seconds=70))))
            common = dict(step=42, stage=1, seconds=.5, gradient_norm=2)
            for loss in [dict(loss=.3, ri_loss=.02),
                         dict(total=.3, ri=.02, cross=.2, encoder_gradient_tensors=26)]:
                with self.subTest(schema=sorted(loss)):
                    (run / 'metrics.jsonl').write_text(json.dumps(dict(common, **loss)) + '\n{"step":')
                    result = status(run, budget)
                    self.assertEqual(result['step'], 42)
                    self.assertEqual(result['mean_loss'], .3)
                    self.assertEqual(result['mean_ri_loss'], .02)
                    self.assertEqual(result['clip_fraction'], 1)
                    # command_seconds already includes completed commands in the active leg.
                    self.assertEqual(result['approx_study_seconds'], 230)
                    self.assertEqual(result['approx_remaining_seconds'], 770)
                    if 'cross' in loss:
                        self.assertEqual(result['mean_cross_latent_mse'], .2)
                        self.assertEqual(result['encoder_gradient_tensors'], 26)


if __name__ == '__main__':
    unittest.main()
