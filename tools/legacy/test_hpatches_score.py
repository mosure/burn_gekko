import unittest
import numpy as np
from hpatches_score import truth_map,dense_map,paired_ci
from fusion_transfer_diagnostics import truth_nearest_grid


class GeometryProtocol(unittest.TestCase):
    def test_paired_precision_interval_has_declared_direction_and_scene_units(self):
        rows=[]
        for scene in ['v_first','v_second']:
            for target in range(2,7):
                for method,error,precision in [('a',2.,.6),('b',1.,.8)]:
                    rows.append(dict(sequence=scene,target=target,method=method,subset='viewpoint',aepe=error,pck3=precision))
        error=paired_ci(rows,'a','b','viewpoint')
        precision=paired_ci(rows,'a','b','viewpoint',metric='pck3')
        self.assertEqual(error['sequences'],2)
        self.assertEqual(error['low'],1.)
        self.assertAlmostEqual(precision['mean'],-.2)
        self.assertAlmostEqual(precision['high'],-.2)

    def test_identity_has_zero_dense_flow_even_at_borders(self):
        gt,valid=truth_map(np.eye(3),(120,480),(120,480))
        predicted=dense_map(np.arange(256))
        self.assertTrue(valid.all())
        np.testing.assert_allclose(predicted,gt,atol=1e-9)

    def test_forward_homography_is_inverted_and_rescaled(self):
        h=np.array([[1.,0.,32.],[0.,1.,0.],[0.,0.,1.]])
        gt,valid=truth_map(h,(480,480),(480,480))
        np.testing.assert_allclose(gt[100,100],[84,100])
        self.assertFalse(valid[100,15]);self.assertTrue(valid[100,16])

    def test_patch_offset_is_scaled_to_metric_pixels(self):
        indices=np.arange(256).reshape(16,16)
        indices[:,:-1]+=1
        got=dense_map(indices.ravel())
        np.testing.assert_allclose(got[120,100],[115,120])

    def test_label_only_grid_control_preserves_centres_and_clips_border(self):
        gt,_=truth_map(np.eye(3),(240,240),(240,240))
        np.testing.assert_array_equal(truth_nearest_grid(gt),np.arange(256))
        moved=gt+np.array([15.,0.])
        expected=np.arange(256).reshape(16,16)
        expected[:,:-1]+=1
        np.testing.assert_array_equal(truth_nearest_grid(moved),expected.ravel())


if __name__=='__main__':unittest.main()
