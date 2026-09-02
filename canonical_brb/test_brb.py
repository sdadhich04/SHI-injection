from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from canonical_brb.core import (
    BRBModel, fit_binary_event, fit_brb, score_binary_event, score_brb,
)
from canonical_brb.features import extract_batch
from canonical_brb.runtime import Reservoir, quarantine_incomplete


class CanonicalBRBTests(unittest.TestCase):
    def test_health_is_bounded_and_noise_lowers_it(self):
        rng = np.random.default_rng(42)
        time = np.arange(128) / 100.0
        healthy = np.stack([np.sin(2 * np.pi * 5 * time) + rng.normal(0, 0.01, 128) for _ in range(24)])
        names, healthy_features = extract_batch(healthy, 100.0, wavelet_level=3, include_adev=False)
        model = fit_brb(names, healthy_features, healthy_quantile=0.9)
        healthy_shi, _, _ = score_brb(model, names, healthy_features)
        noisy_names, noisy_features = extract_batch(
            healthy + rng.normal(0, 0.8, healthy.shape), 100.0, wavelet_level=3, include_adev=False,
        )
        noisy_shi, _, _ = score_brb(model, noisy_names, noisy_features)
        self.assertTrue(np.all((healthy_shi >= 0) & (healthy_shi <= 1)))
        self.assertLess(float(np.mean(noisy_shi)), float(np.mean(healthy_shi)))

    def test_model_round_trip(self):
        rng = np.random.default_rng(8)
        names = ["axis0_sq_snr", "axis0_variance", "axis0_mean"]
        model = fit_brb(names, rng.normal(size=(20, 3)), healthy_quantile=0.9)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            model.to_json(path)
            self.assertEqual(BRBModel.from_json(path), model)

    def test_reservoir_is_bounded_and_reproducible(self):
        values = np.arange(200.0).reshape(100, 2)
        first, second = Reservoir(12, 9), Reservoir(12, 9)
        first.update(["a", "b"], values)
        second.update(["a", "b"], values)
        self.assertEqual(first.seen, 100)
        self.assertEqual(len(first.rows), 12)
        self.assertTrue(np.array_equal(first.matrix(), second.matrix()))

    def test_interrupted_output_is_preserved_for_retry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "archive" / "job"
            destination.mkdir(parents=True)
            (destination / "predictions.bin.tmp").write_bytes(b"partial")
            quarantine = quarantine_incomplete(destination, root)
            self.assertIsNotNone(quarantine)
            self.assertEqual((quarantine / "predictions.bin.tmp").read_bytes(), b"partial")
            self.assertFalse(destination.exists())

    def test_constant_binary_baseline_uses_event_health(self):
        baseline = np.zeros((12, 256, 1))
        model = fit_binary_event(baseline, healthy_quantile=0.95)
        observed = baseline[:2].copy()
        observed[1, 10, 0] = 1.0
        health, event_fraction, degradation = score_binary_event(model, observed)
        self.assertEqual(model.model_type, "binary_event_health")
        self.assertEqual(float(health[0]), 1.0)
        self.assertAlmostEqual(float(health[1]), 0.5, places=6)
        self.assertTrue(np.array_equal(event_fraction, degradation))


if __name__ == "__main__":
    unittest.main()
