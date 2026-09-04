"""Verification tests for the Tier-2 aggregation layer."""

from __future__ import annotations

import json
import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.tier2_metrics import (
    FLAG_CONTINUOUS_BREACH,
    FLAG_CONTINUOUS_PERSISTENT,
    FLAG_EVENT_ALARM,
    OutputConsistencyError,
    TIER2_DTYPE,
    analyse_job,
    build_metrics,
    clopper_pearson,
    default_paths,
    validate_format,
    walk_jobs,
)


def make_synthetic_job(directory: Path) -> None:
    directory.mkdir(parents=True)
    records = np.zeros(6, dtype=TIER2_DTYPE)
    records["timestamp_ms"] = [1000, 2000, 3000, 4000, 5000, 6000]
    records["shi"] = [0.9, 0.8, 0.7, 0.6, 0.4, 0.3]
    records["flags"] = [0, 0, FLAG_CONTINUOUS_BREACH, 0,
                         FLAG_CONTINUOUS_BREACH,
                         FLAG_CONTINUOUS_BREACH | FLAG_CONTINUOUS_PERSISTENT | FLAG_EVENT_ALARM]
    records.tofile(directory / "predictions.bin")
    breach = (records["flags"] & FLAG_CONTINUOUS_BREACH) != 0
    persistent = (records["flags"] & FLAG_CONTINUOUS_PERSISTENT) != 0
    event = (records["flags"] & FLAG_EVENT_ALARM) != 0
    report = {
        "status": "success",
        "pipeline_version": 2,
        "job": {
            "job_id": "synthetic",
            "archive": "01_Experiment_A1_Orbital_Shaker.zip",
            "member": "run/accelerometer.csv",
            "sensor": "accelerometer",
            "axes": 3,
            "protocol": {"fault_offset_s": 4.0},
        },
        "model_type": "fused_plus_event",
        "sampling_rate_hz": 1.0,
        "window_samples": 3,
        "stride_samples": 1,
        "fault_start_ms": 5000,
        "fault_timestamp_confidence": "protocol",
        "baseline_strategy": "pre_fault",
        "calibration_start_ms": 1000,
        "calibration_end_ms": 3000,
        "output_windows": len(records),
        "alarm_rates": {
            "continuous_breach": float(breach.mean()),
            "continuous_persistent": float(persistent.mean()),
            "event": float(event.mean()),
        },
    }
    (directory / "job_report.json").write_text(json.dumps(report), encoding="utf-8")


class SyntheticMetricsTests(unittest.TestCase):
    def test_synthetic_round_trip_and_partitions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "job"
            make_synthetic_job(directory)
            row = analyse_job(directory)
        self.assertEqual(row["n_calibration"], 2)
        self.assertEqual(row["n_pre_fault"], 2)
        self.assertEqual(row["n_post_fault"], 2)
        self.assertEqual(row["n_windows"], 6)
        self.assertEqual(row["pre_fault_breach_count"], 1)
        self.assertEqual(row["pre_fault_breach_rate"], 0.5)
        self.assertEqual(row["post_fault_persistent_rate"], 0.5)
        self.assertTrue(row["detected"])
        self.assertEqual(row["detection_latency_s"], 1.0)
        self.assertTrue(row["event_detected"])

    def test_record_size_violation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "job"
            make_synthetic_job(directory)
            with (directory / "predictions.bin").open("ab") as handle:
                handle.write(b"x")
            with self.assertRaisesRegex(OutputConsistencyError, "record-size violation"):
                analyse_job(directory)

    def test_aggregate_csv_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "hardware"
            directory = root / "archive" / "synthetic"
            make_synthetic_job(directory)
            (root / "format.json").write_text(json.dumps({
                "record_bytes": 48,
                "numpy_dtype": TIER2_DTYPE.descr,
            }))
            out = Path(temporary) / "summaries"
            rows, failures = build_metrics(root, out)
            with (out / "tier2_event_branch.csv").open() as handle:
                event_rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        self.assertEqual(failures, [])
        self.assertEqual(event_rows[0]["calibration_event_count"], "0")
        self.assertEqual(event_rows[0]["pre_fault_event_count"], "0")
        self.assertEqual(event_rows[0]["post_fault_event_count"], "1")

    def test_whole_recording_rate_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "job"
            make_synthetic_job(directory)
            path = directory / "job_report.json"
            report = json.loads(path.read_text())
            report["alarm_rates"]["event"] = 0.0
            path.write_text(json.dumps(report))
            with self.assertRaisesRegex(OutputConsistencyError, "event rate mismatch"):
                analyse_job(directory)

    def test_clopper_pearson_sanity(self) -> None:
        for successes, trials in ((0, 10), (1, 10), (5, 10), (10, 10)):
            lower, upper = clopper_pearson(successes, trials)
            estimate = successes / trials
            self.assertLessEqual(lower, estimate)
            self.assertGreaterEqual(upper, estimate)
        self.assertEqual(clopper_pearson(0, 10)[0], 0.0)


class DepositedOutputIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root, _, cls.source_root = default_paths()
        if not cls.root.exists():
            raise unittest.SkipTest("deposited Tier-2 hardware output is unavailable")

    def test_format_and_every_real_job(self) -> None:
        validate_format(self.root)
        directories = list(walk_jobs(self.root))
        self.assertGreater(len(directories), 0)
        errors = []
        for directory in directories:
            try:
                row = analyse_job(directory, source_root=self.source_root)
                self.assertEqual(
                    row["n_calibration"] + row["n_pre_fault"] + row["n_post_fault"],
                    row["n_windows"],
                )
            except Exception as error:  # Collect all paths in one actionable failure.
                errors.append(f"{directory}: {type(error).__name__}: {error}")
        self.assertEqual(errors, [], "\n" + "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
