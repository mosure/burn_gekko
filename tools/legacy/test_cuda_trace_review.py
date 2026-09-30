import unittest
from cuda_trace_review import union_seconds


class TraceIntervalsTests(unittest.TestCase):
    def test_overlap_and_window_clipping_do_not_double_count_gpu_time(self):
        ranges = [(0, 3_000_000_000), (2_000_000_000, 4_000_000_000),
                  (6_000_000_000, 9_000_000_000)]
        self.assertEqual(union_seconds(ranges, 1_000_000_000, 7_000_000_000), 4)


if __name__ == '__main__':
    unittest.main()
