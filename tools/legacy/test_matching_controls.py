import unittest
import numpy as np
from verify_matching_controls import compare


class MatchingControls(unittest.TestCase):
    def test_counts_changed_predictions_and_mutual_flags_separately(self):
        a={'pair1':(np.array([0,1,2,3]),np.ones(4,dtype=bool))}
        b={'pair1':(np.array([0,0,2,3]),np.array([True,False,False,True]))}
        result=compare(a,b)
        self.assertEqual(result['indices_changed'],1)
        self.assertEqual(result['mutual_changed'],2)
        self.assertEqual(result['changed_index_fraction'],.25)
        self.assertEqual(compare(a,a)['indices_changed'],0)

    def test_missing_pair_cannot_pass_as_equal(self):
        a={'pair1':(np.arange(4),np.ones(4,dtype=bool))}
        with self.assertRaisesRegex(AssertionError,'different pairs'):
            compare(a,{})


if __name__ == '__main__':
    unittest.main()
