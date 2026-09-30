import unittest
from gpu_efficiency import integrate


class EnergyTests(unittest.TestCase):
    def test_irregular_samples_and_boundary_accounting(self):
        rows = [dict(elapsed_seconds=t, device_power_w=w, device_gpu_percent=90)
                for t, w in [(1, 100), (2, 200), (4, 100)]]
        result = integrate(rows, duration=5)
        self.assertEqual(result['observed_board_joules'], 650)
        self.assertEqual(result['coverage_fraction'], 1)
        self.assertEqual(result['time_weighted_board_power_w'], 130)

    def test_missing_power_does_not_bridge_a_long_gap(self):
        result = integrate([dict(elapsed_seconds=0, device_power_w=100),
                            dict(elapsed_seconds=2),
                            dict(elapsed_seconds=5, device_power_w=200),
                            dict(elapsed_seconds=6, device_power_w=200)], duration=6)
        self.assertEqual(result['observed_board_joules'], 200)
        self.assertEqual(result['unknown_seconds'], 5)
        self.assertAlmostEqual(result['coverage_fraction'], 1/6)


if __name__ == '__main__':
    unittest.main()
