from pathlib import Path
import importlib.util
import unittest

import numpy as np

from common import PREDICTION_DTYPE, deterministic_sample, feature_dtype, labels_with_prediction_horizon
from model_features import extract_feature_batch, get_pipeline_feature_names


class ModelShiTests(unittest.TestCase):
    def test_prediction_record_format(self):
        self.assertEqual(PREDICTION_DTYPE.itemsize, 32)
        self.assertEqual(feature_dtype(26).itemsize, 116)
        self.assertEqual(feature_dtype(78).itemsize, 324)

    def test_multiple_injection_intervals_get_prefault_labels(self):
        timestamps = np.arange(10, dtype=np.int64) * 1000
        active = np.array([0, 0, 0, 1, 1, 0, 0, 1, 1, 0], dtype=bool)
        labels = labels_with_prediction_horizon(timestamps, active, 2.0)
        np.testing.assert_array_equal(labels, [0, 1, 1, 2, 2, 1, 1, 2, 2, 0])

    def test_sampling_is_stable_and_retains_classes(self):
        labels = np.array([0] * 90 + [1] * 5 + [2] * 5)
        first = deterministic_sample(labels, 20)
        second = deterministic_sample(labels, 20)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(set(labels[first]), {0, 1, 2})

    def test_optimized_features_match_vendored_upstream(self):
        upstream_path = Path(__file__).parent / "upstream" / "features.py"
        spec = importlib.util.spec_from_file_location("test_upstream_features", upstream_path)
        upstream = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(upstream)
        windows = np.random.default_rng(42).normal(size=(6, 256))
        expected = upstream.build_feature_pipeline(1000.0).transform(windows)
        actual = extract_feature_batch(windows, 1000.0, include_ar=True)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10)
        self.assertEqual(len(get_pipeline_feature_names(False)), 26)
        self.assertEqual(len(get_pipeline_feature_names(True)), 78)

        without_ar = extract_feature_batch(windows, 1000.0, include_ar=False)
        np.testing.assert_allclose(without_ar, expected[:, :22], rtol=1e-10, atol=1e-10)
        self.assertEqual(without_ar.shape, (6, 22))
        self.assertEqual(len(get_pipeline_feature_names(False, include_ar=False)), 22)
        self.assertEqual(len(get_pipeline_feature_names(True, include_ar=False)), 66)


if __name__ == "__main__":
    unittest.main()
