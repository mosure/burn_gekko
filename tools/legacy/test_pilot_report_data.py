"""CPU checks for report statistics and the exported-array visualization contract."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from pilot_report_data import room_metrics, sample_arrays


class ReportDataTests(unittest.TestCase):
    def test_room_weighting_and_single_class_auc(self):
        def target(seed, auc):
            return dict(room_seed=seed, cross_mse=.4, mae_mse=.5, total_loss=.95,
                        ri_loss=.05, covisibility=dict(auroc=auc, average_precision=.6))
        report = dict(targets=[target(1, 1.), target(1, 0.), target(1, None), target(2, .2)])
        result = room_metrics(report)["auroc"]["bootstrap"]
        self.assertEqual(result["rooms"], 2)
        self.assertAlmostEqual(result["mean"], .35)
        self.assertAlmostEqual(result["low"], .2)
        self.assertAlmostEqual(result["high"], .5)

    def test_patch_normalization_and_layout_round_trip(self):
        root = Path(".data/tests/report-arrays")
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temp:
            directory = Path(temp)
            target = np.arange(32*32*3, dtype=np.float32).reshape(32,32,3)/(32*32*3)
            target.astype("<f4").tofile(directory/"target.f32")
            target.astype("<f4").tofile(directory/"reference-0.f32")
            patches = target.reshape(2,16,2,16,3).transpose(0,2,1,3,4).reshape(4,-1)
            norm = (patches-patches.mean(axis=1,keepdims=True))/np.sqrt(patches.var(axis=1,ddof=1,keepdims=True)+1e-6)
            for name in ["cross", "mae"]:
                norm.astype("<f4").tofile(directory/f"{name}.f32")
            np.repeat(np.arange(4,dtype="<f4"),256).tofile(directory/"ri.f32")
            np.ones((32,32),dtype=np.uint8).tofile(directory/"visibility.u8")
            (directory/"sample.json").write_text(json.dumps(dict(height=32,width=32,patch_size=16,
                reference_views=[1],normalize_targets=True,visible_patch_ids=[0,3])))
            result = sample_arrays(directory)
            np.testing.assert_allclose(result["cross"],target,atol=1e-7)
            self.assertLess(result["mae_error"].max(),1e-12)
            self.assertEqual(result["hidden"].sum(),512)
            np.testing.assert_array_equal(result["ri"][::16,::16],[[0,1],[2,3]])
            self.assertTrue(result["hidden"][0,16])
            self.assertFalse(result["hidden"][16,16])
            meta=json.loads((directory/"sample.json").read_text())
            meta["predicted_patch_statistics"]=True
            (directory/"sample.json").write_text(json.dumps(meta))
            statistics=np.concatenate([patches.mean(axis=1,keepdims=True),np.log(np.sqrt(patches.var(axis=1,ddof=1,keepdims=True)+1e-6))],axis=1)
            for name in ["cross","mae"]:statistics.astype("<f4").tofile(directory/f"{name}-statistics.f32")
            before=sample_arrays(directory)["cross"]
            (target*.3).astype("<f4").tofile(directory/"target.f32")
            after=sample_arrays(directory)["cross"]
            np.testing.assert_allclose(before,after,atol=0,rtol=0,err_msg="predicted-statistic display must not use hidden target RGB")


if __name__ == "__main__":
    unittest.main()
