"""Focused unit and end-to-end tests for the Tier-2 fused SHI adapter."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from tier2_fused_shi.core import (
    FLAG_EVENT_ALARM,
    TIER2_DTYPE,
    EventRateModel,
    FusedConfig,
    FusedSHI,
    event_rates,
    extract_features,
    feature_names,
)
from tier2_fused_shi.plot_hardware import plot_one
from tier2_fused_shi.run_hardware import process_job


class CoreTests(unittest.TestCase):
    def test_feature_schema_and_score_range(self) -> None:
        rng = np.random.default_rng(12)
        time = np.linspace(0.0, 3.0, 96, endpoint=False)
        windows = np.stack([
            np.sin(2 * np.pi * (2.0 + index / 50.0) * time)
            + 0.03 * rng.normal(size=len(time))
            for index in range(30)
        ])[:, :, None]
        features = extract_features(windows)
        self.assertEqual(features.shape, (30, 20))
        self.assertTrue(np.all(np.isfinite(features)))

        model = FusedSHI(FusedConfig(isolation_estimators=8, minimum_calibration_windows=10))
        model.fit(features[:20], feature_names(1))
        scored = model.score(features[20:])
        for name in ("shi", "isolation_health", "mahalanobis_health", "ewma_health"):
            self.assertTrue(np.all((scored[name] >= 0.0) & (scored[name] <= 1.0)))
        self.assertEqual(model.metadata()["parameters"]["fusion_weights"], [1 / 3] * 3)

    def test_event_rate_detects_clipped_zero_health(self) -> None:
        healthy = np.zeros((24, 40, 1), dtype=np.float64)
        healthy[:, ::10, 0] = 1.0
        model = EventRateModel().fit(event_rates(healthy))
        faulty = np.indices((1, 40, 1))[1] % 2
        health = model.score_rates(event_rates(faulty))
        self.assertEqual(float(health[0]), 0.0)
        self.assertLessEqual(float(health[0]), model.alarm_threshold)

    def test_binary_record_size_is_stable(self) -> None:
        self.assertEqual(TIER2_DTYPE.itemsize, 48)


class EndToEndTests(unittest.TestCase):
    def test_zip_to_binary_to_png(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_text:
            temporary = Path(temporary_text)
            input_root = temporary / "input"
            output_root = temporary / "output"
            plot_root = temporary / "plots"
            input_root.mkdir()
            archive = input_root / "synthetic.zip"
            member = "Experiment/Converted/UNIT_0001_RUN_001/temperature.csv"

            csv_path = temporary / "temperature.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(("timestamp_ms", "value"))
                for index in range(3000):
                    seconds = index / 30.0
                    value = (
                        20.0 + 0.1 * np.sin(seconds / 3.0)
                        + 0.02 * np.sin(seconds * (1.0 + seconds / 500.0))
                    )
                    if seconds >= 40.0:
                        value += 1.5
                    writer.writerow((index * 33, value))
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.write(csv_path, member)
                info = bundle.getinfo(member)

            job = {
                "job_id": "synthetic-temperature",
                "archive": archive.name,
                "member": member,
                "sensor": "temperature",
                "axes": 1,
                "source_format": "csv",
                "uncompressed_bytes": info.file_size,
                "crc32": f"{info.CRC:08x}",
                "protocol": {
                    "fault_offset_s": 40.0,
                    "baseline": "pre_fault",
                    "baseline_seconds": 40.0,
                    "confidence": "test",
                    "reason": "synthetic integration test",
                },
            }
            fused = FusedConfig(isolation_estimators=8, minimum_calibration_windows=10)
            settings = {
                "window_seconds": 3.0,
                "overlap": 0.5,
                "feature_batch_windows": 16,
                "fused_config": fused.__dict__,
                "event_branch": "auto",
                "save_model": True,
                "record_bytes": TIER2_DTYPE.itemsize,
            }
            result = process_job(job, {
                "input": str(input_root),
                "output": str(output_root),
                "window_seconds": 3.0,
                "overlap": 0.5,
                "feature_batch_windows": 16,
                "fused_config": fused.__dict__,
                "event_branch": "auto",
                "save_model": True,
                "processing_config": settings,
                "provenance": {"test": True},
            })
            self.assertEqual(result["status"], "success")
            destination = output_root / archive.stem / job["job_id"]
            predictions = destination / "predictions.bin"
            records = np.memmap(predictions, dtype=TIER2_DTYPE, mode="r")
            self.assertGreater(len(records), 20)
            self.assertTrue(np.all((records["shi"] >= 0) & (records["shi"] <= 1)))
            marker = json.loads((destination / "job_report.json").read_text())
            self.assertEqual(marker["baseline_strategy"], "pre_fault")

            plotted = plot_one(
                str(predictions), str(output_root), str(plot_root), 500, None, False,
            )
            self.assertEqual(plotted["status"], "success")
            self.assertGreater(Path(plotted["output"]).stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
