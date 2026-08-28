import tempfile
import unittest
from pathlib import Path

import numpy as np

from analyze_simple_shi import (
    BINARY_DTYPE,
    RunningStats,
    calculate_shi,
    estimate_fs,
    make_plan,
)


class ShiCalculationTests(unittest.TestCase):
    def test_sampling_rate_uses_positive_timestamp_deltas(self):
        self.assertEqual(estimate_fs(np.array([0, 1, 1, 2, 3])), 1000.0)

    def test_running_scale_matches_streamed_standard_deviation(self):
        values = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]])
        stats = RunningStats()
        stats.update(values[:2])
        stats.update(values[2:])
        expected = np.maximum(np.std(values, axis=0, ddof=1), 0.1 * np.mean(np.abs(values), axis=0))
        np.testing.assert_allclose(stats.scale(0.1), expected)

    def test_exact_clean_window_is_fully_healthy(self):
        clean = np.array([[1.0, 2.0], [2.0, 4.0]])
        fractions = np.zeros(2)
        health, distance = calculate_shi(clean, clean.copy(), np.ones(2), fractions)
        np.testing.assert_allclose(health, 100.0)
        np.testing.assert_allclose(distance, 0.0)

    def test_feature_deviation_reduces_health(self):
        clean = np.ones((2, 2))
        observed = np.full((2, 2), 2.0)
        health, _ = calculate_shi(clean, observed, np.ones(2), np.ones(2))
        self.assertTrue(np.all(health < 100.0))
        self.assertTrue(np.all(health >= 0.0))

    def test_binary_record_is_fixed_width(self):
        self.assertEqual(BINARY_DTYPE.itemsize, 44)
        data = np.zeros(3, dtype=BINARY_DTYPE)
        self.assertEqual(len(data.tobytes()), 132)

    def test_plan_is_deterministic_and_balanced(self):
        with tempfile.TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            jobs = []
            for index, records in enumerate((100, 80, 40, 20)):
                relative = Path(f"RUN_{index:03d}/sensor.bin")
                paths = [dataset / "ground_truth" / relative]
                paths += [
                    dataset / "noise_injection" / relative.parts[0] / variant / relative.name
                    for variant in ("random_0_25", "random_0_5")
                ]
                for path in paths:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b"input")
                jobs.append({"job_id": str(index), "clean_relative": relative.as_posix(), "records": records})
            first = make_plan(dataset, jobs, 2)
            second = make_plan(dataset, list(reversed(jobs)), 2)
            self.assertEqual(first, second)
            self.assertEqual([batch["records"] for batch in first["batches"]], [120, 120])


if __name__ == "__main__":
    unittest.main()
