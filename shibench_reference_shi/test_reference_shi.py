import unittest

import numpy as np

from shibench_reference_shi.core import (
    FLAG_CONTINUOUS_ALARM, FLAG_LIVENESS_ALARM, REFERENCE_DTYPE,
    ReconstructionConfig, ReferenceSHI, extract_reference_features,
)


class ReferenceSHITests(unittest.TestCase):
    def make_windows(self, count=240, size=256):
        rng = np.random.default_rng(123)
        phase = np.linspace(0, 8 * np.pi, size)
        return np.asarray([
            (np.sin(phase + index * 0.03) + rng.normal(0, 0.03, size))[:, None]
            for index in range(count)
        ])

    def test_binary_dtype_is_fixed_width(self):
        self.assertEqual(REFERENCE_DTYPE.itemsize, 48)

    def test_calibration_targets_five_percent_continuous_false_positives(self):
        windows = self.make_windows()
        timestamps = np.arange(len(windows), dtype=np.uint64) * 640
        features = extract_reference_features(windows, 100.0)
        model = ReferenceSHI(ReconstructionConfig(isolation_estimators=16)).fit(
            features, windows, timestamps, "temperature",
        )
        model.reset()
        scores = model.score(features, windows, timestamps, windows[:, -1])
        rate = np.mean(scores["shi"] < 0.5)
        self.assertLess(abs(rate - 0.05), 0.04)

    def test_stuck_signal_triggers_health_or_liveness_alarm(self):
        calibration = self.make_windows()
        timestamps = np.arange(len(calibration), dtype=np.uint64) * 640
        model = ReferenceSHI(ReconstructionConfig(isolation_estimators=16)).fit(
            extract_reference_features(calibration, 100.0), calibration, timestamps,
            "accelerometer",
        )
        stuck = np.full((30, 256, 1), 4.0)
        stuck_timestamps = np.arange(30, dtype=np.uint64) * 640 + timestamps[-1] + 640
        result = model.score(
            extract_reference_features(stuck, 100.0), stuck, stuck_timestamps, stuck[:, -1],
        )
        alarm = (result["flags"] & (FLAG_CONTINUOUS_ALARM | FLAG_LIVENESS_ALARM)) != 0
        self.assertGreater(np.mean(alarm), 0.9)

    def test_auto_event_branch_is_enabled_for_binary_vibration(self):
        windows = np.zeros((80, 64, 1), dtype=float)
        windows[::4, 20:22, 0] = 1.0
        timestamps = np.arange(len(windows), dtype=np.uint64) * 640
        model = ReferenceSHI(
            ReconstructionConfig(isolation_estimators=8, minimum_calibration_windows=20)
        ).fit(extract_reference_features(windows, 100.0), windows, timestamps, "vibration")
        self.assertTrue(model.event_enabled)
        model.reset()
        result = model.score(
            extract_reference_features(windows, 100.0), windows, timestamps, windows[:, -1],
        )
        self.assertGreater(float(np.median(result["event_health"])), 0.8)


if __name__ == "__main__":
    unittest.main()
