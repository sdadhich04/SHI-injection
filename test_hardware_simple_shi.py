from pathlib import Path
import json
import tempfile
import unittest
import zipfile

import numpy as np

from hardware_simple_shi import HARDWARE_DTYPE, discover_jobs, process_job, protocol_for


class HardwareShiTests(unittest.TestCase):
    def test_binary_record_is_fixed_width(self):
        self.assertEqual(HARDWARE_DTYPE.itemsize, 28)

    def test_protocol_families(self):
        self.assertEqual(protocol_for("01_Experiment_A1/x")["fault_offset_s"], 300.0)
        self.assertEqual(protocol_for("02_Experiment_A2/x")["baseline"], "tail_recovery")
        self.assertEqual(protocol_for("04/IMU bias injection/x")["confidence"], "limited")
        self.assertEqual(protocol_for("04/Supply-voltage-droop/x")["fault_offset_s"], 0.0)

    def test_discovers_and_processes_csv_inside_zip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs, outputs = root / "input", root / "output"
            inputs.mkdir()
            rows = ["timestamp_ms,value"]
            for index in range(800):
                value = np.sin(index / 20) + (5.0 if index >= 500 else 0.0)
                rows.append(f"{index * 10},{value}")
            archive = inputs / "sample.zip"
            member = "01_Experiment_A1/Test/Converted/UNIT_0001_RUN_001/temperature.csv"
            with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr(member, "\n".join(rows) + "\n")
            jobs = discover_jobs(inputs, None, None, 5.0)
            self.assertEqual(len(jobs), 1)
            config = {
                "input_dir": str(inputs), "output_dir": str(outputs),
                "window_size": 64, "stride": 16, "feature_batch_windows": 32,
                "relative_tolerance": 0.1, "include_ar": False,
                "processing_config": {"window_size": 64},
            }
            result = process_job(jobs[0], config)
            self.assertEqual(result["status"], "success")
            report = result["report"]
            self.assertEqual(report["fault_start_ms"], 5000)
            data = np.memmap(report["output_path"], dtype=HARDWARE_DTYPE, mode="r")
            self.assertGreater(len(data), 1)
            self.assertTrue(np.all(np.isfinite(data["shi"])))
            self.assertLess(float(np.mean(data["shi"][-5:])), float(np.mean(data["shi"][:5])))


if __name__ == "__main__":
    unittest.main()
