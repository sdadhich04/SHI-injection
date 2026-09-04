"""Measure calibration-region Tier-2 SHI health without rerunning the detector."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from tier2_metrics import analyse_job, default_paths, walk_jobs


COLUMNS = [
    "job_id", "archive", "member", "sensor", "experiment_family",
    "fault_subtype", "fault_target", "onset_confidence", "baseline_strategy",
    "continuous_branch_enabled", "calibration_window_count",
    "calibration_region_mean_shi", "calibration_mean_finite", "below_0_7",
]


def build(root: Path, output: Path) -> list[dict]:
    rows = []
    for directory in walk_jobs(root):
        row = analyse_job(directory)
        mean = float(row["calibration_shi_mean"])
        finite = bool(row["continuous_branch_enabled"]) and math.isfinite(mean)
        rows.append({
            "job_id": row["job_id"],
            "archive": row["archive"],
            "member": row["member"],
            "sensor": row["sensor"],
            "experiment_family": row["experiment_family"],
            "fault_subtype": row["fault_subtype"],
            "fault_target": row["fault_target"],
            "onset_confidence": row["onset_confidence"],
            "baseline_strategy": row["baseline_strategy"],
            "continuous_branch_enabled": row["continuous_branch_enabled"],
            "calibration_window_count": row["n_calibration"],
            "calibration_region_mean_shi": mean if finite else "",
            "calibration_mean_finite": finite,
            "below_0_7": finite and mean < 0.7,
        })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    default_root, default_out, _ = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--out", type=Path, default=default_out / "tier2_calibration_health.csv")
    args = parser.parse_args()
    rows = build(args.root.resolve(), args.out.resolve())
    eligible = [row for row in rows if row["calibration_mean_finite"]]
    flagged = [row for row in eligible if row["below_0_7"]]
    experiment_b = [row for row in eligible if row["experiment_family"] == "B_faulty_hardware"]
    flagged_b = [row for row in experiment_b if row["below_0_7"]]
    print(
        f"Analysed {len(rows)} jobs; {len(eligible)} have continuous SHI and "
        f"{len(flagged)} of those have calibration mean SHI < 0.7."
    )
    print(f"Event-rate-only jobs (continuous calibration SHI not applicable): {len(rows) - len(eligible)}.")
    print(f"Experiment B: {len(flagged_b)}/{len(experiment_b)} flagged.")
    print(args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
