import unittest
import cv2
import numpy as np
from eth3d_score import sparse_flow,sparse_truth,aggregate,paired_scene_ci


class Eth3dProtocol(unittest.TestCase):
    def test_precision_interval_resamples_scenes_and_keeps_all_intervals(self):
        a={f'{r}/{s}':dict(aepe=5.,pck3=.4) for r in [3,5,7,9,11,13,15] for s in range(10)}
        b={k:dict(aepe=3.,pck3=.5) for k in a}
        result=paired_scene_ci(a,b,metric='pck3')
        self.assertEqual(result['scenes'],10)
        self.assertAlmostEqual(result['mean'],-.1)
        self.assertAlmostEqual(result['low'],-.1)
        self.assertAlmostEqual(paired_scene_ci(a,b)['high'],2.)

    def test_sparse_sampler_agrees_with_opencv_full_flow(self):
        rng=np.random.default_rng(9);indices=rng.integers(0,256,256);h,w=480,752
        y,x=np.mgrid[:16,:16];ids=indices.reshape(16,16)
        low=np.stack([(ids%16-x)*w/16,(ids//16-y)*h/16],-1).astype(np.float32)
        dense=cv2.resize(low,(w,h),interpolation=cv2.INTER_LINEAR)
        xy=np.concatenate([rng.integers([0,0],[w,h],size=(1000,2)),[[0,0],[w-1,h-1],[375,239]]])
        np.testing.assert_allclose(sparse_flow(indices,xy,[h,w]),dense[xy[:,1],xy[:,0]],atol=8e-4)

    def test_label_rasterization_preserves_float_displacement_and_last_duplicate(self):
        points=np.array([[4.1,5.8,1.1,2.8],[9.2,7.4,1.2,2.9],[6.,8.,4.,5.]],np.float32)
        xy,gt=sparse_truth(points,[10,10]);mapping={tuple(p):v for p,v in zip(xy,gt)}
        np.testing.assert_allclose(mapping[(1,3)],[8.,4.5],atol=1e-6)
        np.testing.assert_allclose(mapping[(4,5)],[2.,3.],atol=1e-6)

    def test_scene_weighting_does_not_favor_scenes_with_more_pairs(self):
        rows=[]
        for interval in [3,5,7,9,11,13,15]:
            for scene in range(10):
                for _ in range(scene+1):
                    rows.append(dict(interval=interval,scene=str(scene),aepe=float(scene),pck1=.1,pck3=.3,pck5=.5,points=10,correct1=1,correct3=3,correct5=5))
        summary=aggregate(rows)['summary'];self.assertEqual(summary['aepe'],4.5)
        self.assertAlmostEqual(summary['point_weighted_pck3'],.3)


if __name__=='__main__':unittest.main()
