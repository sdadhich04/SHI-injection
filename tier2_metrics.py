"""Fault-aligned metrics and aggregation for the Tier-2 fused SHI outputs.

The per-job reports record ``first_detection`` as the first threshold breach
anywhere in the recording and ``alarm_rates`` as whole-recording rates. Neither
is aligned to the documented fault onset, so neither is a detection latency or a
true-detection rate. This module derives the aligned quantities from
``predictions.bin`` and writes the aggregate tables that Section 5 needs.

Window partition, per recording:

    calibration   calibration_start_ms <= t <  calibration_end_ms
    pre-fault     t <  fault_start_ms  and  t >= calibration_end_ms
    post-fault    t >= fault_start_ms

The pre-fault partition is the only out-of-sample healthy evidence a
within-recording job provides, because the calibration partition is the data the
threshold was fitted to and a 5% breach rate there merely restates the
calibration. Jobs whose protocol sets ``fault_offset_s = 0`` have no pre-fault
partition at all; they are reported separately and must not be pooled with the
protocol-confident jobs.

Usage:

    python tier2_metrics.py \
        --root outputs/tier2_fused_shi/hardware \
        --out  outputs/tier2_fused_shi/summaries

Writes ``tier2_jobs.csv`` (one row per job), ``tier2_summary.json`` (pooled
figures with binomial confidence intervals) and ``tier2_failures.csv``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

try:
    from scipy.stats import beta as _beta
except ImportError:  # pragma: no cover
    _beta = None


TIER2_DTYPE = np.dtype(
    [
        ("timestamp_ms", "<u8"),
        ("shi", "<f4"),
        ("isolation_health", "<f4"),
        ("mahalanobis_health", "<f4"),
        ("ewma_health", "<f4"),
        ("event_health", "<f4"),
        ("alarm_threshold", "<f4"),
        ("flags", "u1"),
        ("reserved", "u1", (3,)),
        ("signal", "<f4", (3,)),
    ],
    align=False,
)

FLAG_CONTINUOUS_BREACH = 1 << 0
FLAG_CONTINUOUS_PERSISTENT = 1 << 1
FLAG_EVENT_ALARM = 1 << 2

CSV_COLUMNS = [
    "job_id", "archive", "member", "sensor", "axes", "experiment_family",
    "pipeline_version", "sampling_rate_hz", "window_samples", "stride_samples",
    "onset_confidence", "baseline_strategy", "calibration_warning",
    "n_windows", "n_calibration", "n_pre_fault", "n_post_fault",
    "calibration_shi_mean", "pre_fault_shi_mean", "post_fault_shi_mean",
    "calibration_breach_rate", "pre_fault_breach_rate", "pre_fault_persistent_rate",
    "post_fault_breach_rate", "post_fault_persistent_rate",
    "detected", "detection_latency_s", "event_detected", "event_latency_s",
]


def clopper_pearson(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial interval. Returns (lo, hi); (nan, nan) without scipy."""
    if trials == 0:
        return (float("nan"), float("nan"))
    if _beta is None:
        return (float("nan"), float("nan"))
    lo = 0.0 if successes == 0 else float(_beta.ppf(alpha / 2, successes, trials - successes + 1))
    hi = 1.0 if successes == trials else float(
        _beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
    )
    return (lo, hi)


def experiment_family(archive: str) -> str:
    """Coarse family label derived from the source archive name."""
    name = archive.lower()
    if "a1" in name or "shaker" in name:
        return "A1_orbital_shaker"
    if "a2" in name or "thermal" in name or "fridge" in name:
        return "A2_thermal"
    if "a3" in name or "emi" in name:
        return "A3_emi"
    if "faulty_hardware" in name or "_b_" in name:
        return "B_faulty_hardware"
    return "other"


def fault_subtype(member: str) -> str:
    """Sub-experiment label from the member path inside the archive.

    Experiment B holds several distinct physical faults in one archive; without
    this the connector, bias, occlusion, aging and droop runs are pooled.
    """
    lowered = member.lower()
    for needle, label in (
        ("loose connector", "loose_connector"),
        ("imu bias injection", "imu_bias"),
        ("partial-occlusion", "partial_occlusion"),
        ("accelerated aging", "accelerated_aging"),
        ("supply-voltage-droop", "supply_voltage_droop"),
    ):
        if needle in lowered:
            return label
    return ""


def analyse_job(job_directory: Path) -> dict | None:
    report_path = job_directory / "job_report.json"
    predictions_path = job_directory / "predictions.bin"
    if not report_path.exists() or not predictions_path.exists():
        return None

    report = json.loads(report_path.read_text())
    if report.get("status") != "success":
        return None

    records = np.fromfile(predictions_path, dtype=TIER2_DTYPE)
    if not len(records):
        return None

    timestamps = records["timestamp_ms"].astype(np.int64)
    shi = records["shi"].astype(np.float64)
    flags = records["flags"]
    breach = (flags & FLAG_CONTINUOUS_BREACH) != 0
    persistent = (flags & FLAG_CONTINUOUS_PERSISTENT) != 0
    event = (flags & FLAG_EVENT_ALARM) != 0

    fault_start = int(report["fault_start_ms"])
    calibration_start = int(report["calibration_start_ms"])
    calibration_end = int(report["calibration_end_ms"])

    is_calibration = (timestamps >= calibration_start) & (timestamps < calibration_end)
    is_post = timestamps >= fault_start
    is_pre = (~is_post) & (~is_calibration)

    def rate(mask: np.ndarray, values: np.ndarray) -> float:
        return float(values[mask].mean()) if mask.any() else float("nan")

    # Detection: first persistent breach at or after the documented onset.
    detected = bool(persistent[is_post].any()) if is_post.any() else False
    latency = float("nan")
    if detected:
        first = timestamps[is_post & persistent].min()
        latency = (first - fault_start) / 1000.0

    event_detected = bool(event[is_post].any()) if is_post.any() else False
    event_latency = float("nan")
    if event_detected:
        first_event = timestamps[is_post & event].min()
        event_latency = (first_event - fault_start) / 1000.0

    job = report["job"]
    member = job.get("member", "")
    subtype = fault_subtype(member)
    family = experiment_family(job.get("archive", ""))
    if subtype:
        family = f"{family}/{subtype}"

    return {
        "job_id": job.get("job_id", ""),
        "archive": job.get("archive", ""),
        "member": member,
        "sensor": job.get("sensor", ""),
        "axes": job.get("axes", ""),
        "experiment_family": family,
        "pipeline_version": report.get("pipeline_version", ""),
        "sampling_rate_hz": report.get("sampling_rate_hz", ""),
        "window_samples": report.get("window_samples", ""),
        "stride_samples": report.get("stride_samples", ""),
        "onset_confidence": report.get("fault_timestamp_confidence", ""),
        "baseline_strategy": report.get("baseline_strategy", ""),
        "calibration_warning": report.get("calibration_warning") or "",
        "n_windows": int(len(records)),
        "n_calibration": int(is_calibration.sum()),
        "n_pre_fault": int(is_pre.sum()),
        "n_post_fault": int(is_post.sum()),
        "calibration_shi_mean": rate(is_calibration, shi),
        "pre_fault_shi_mean": rate(is_pre, shi),
        "post_fault_shi_mean": rate(is_post, shi),
        "calibration_breach_rate": rate(is_calibration, breach.astype(float)),
        "pre_fault_breach_rate": rate(is_pre, breach.astype(float)),
        "pre_fault_persistent_rate": rate(is_pre, persistent.astype(float)),
        "post_fault_breach_rate": rate(is_post, breach.astype(float)),
        "post_fault_persistent_rate": rate(is_post, persistent.astype(float)),
        "detected": detected,
        "detection_latency_s": latency,
        "event_detected": event_detected,
        "event_latency_s": event_latency,
        # Retained for pooling but not written to the per-job CSV.
        "_pre_breach_count": int(breach[is_pre].sum()) if is_pre.any() else 0,
        "_pre_count": int(is_pre.sum()),
        "_cal_breach_count": int(breach[is_calibration].sum()) if is_calibration.any() else 0,
        "_cal_count": int(is_calibration.sum()),
    }


def walk_jobs(root: Path):
    for report_path in sorted(root.rglob("job_report.json")):
        yield report_path.parent


def summarise(rows: list[dict]) -> dict:
    protocol = [r for r in rows if r["onset_confidence"] == "protocol"]
    limited = [r for r in rows if r["onset_confidence"] != "protocol"]

    def pooled_pre_fault(subset: list[dict]) -> dict:
        trials = sum(r["_pre_count"] for r in subset)
        successes = sum(r["_pre_breach_count"] for r in subset)
        lo, hi = clopper_pearson(successes, trials)
        return {
            "n_jobs": len(subset),
            "n_windows": trials,
            "n_flagged": successes,
            "rate": (successes / trials) if trials else float("nan"),
            "ci95_lo": lo,
            "ci95_hi": hi,
            "target": 0.05,
        }

    def detection(subset: list[dict]) -> dict:
        usable = [r for r in subset if r["n_post_fault"] > 0]
        detected = [r for r in usable if r["detected"]]
        lo, hi = clopper_pearson(len(detected), len(usable))
        latencies = [r["detection_latency_s"] for r in detected if not math.isnan(r["detection_latency_s"])]
        return {
            "n_jobs": len(usable),
            "n_detected": len(detected),
            "rate": (len(detected) / len(usable)) if usable else float("nan"),
            "ci95_lo": lo,
            "ci95_hi": hi,
            "median_latency_s": float(np.median(latencies)) if latencies else float("nan"),
            "iqr_latency_s": (
                [float(np.percentile(latencies, 25)), float(np.percentile(latencies, 75))]
                if len(latencies) >= 4 else None
            ),
        }

    by_family: dict[str, dict] = {}
    for row in rows:
        by_family.setdefault(row["experiment_family"], []).append(row)

    return {
        "n_jobs_analysed": len(rows),
        "protocol_confidence": {
            "pre_fault_false_positive": pooled_pre_fault(protocol),
            "detection": detection(protocol),
        },
        "limited_confidence": {
            "note": (
                "calibration may contain the physical fault; these jobs are "
                "exploratory and are not pooled with protocol-confidence jobs"
            ),
            "pre_fault_false_positive": pooled_pre_fault(limited),
            "detection": detection(limited),
        },
        "by_experiment_family": {
            family: {
                "n_jobs": len(subset),
                "sensors": sorted({r["sensor"] for r in subset}),
                "detection": detection(subset),
                "pre_fault_false_positive": pooled_pre_fault(subset),
            }
            for family, subset in sorted(by_family.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    arguments = parser.parse_args()
    arguments.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    skipped: list[str] = []
    for job_directory in walk_jobs(arguments.root):
        try:
            row = analyse_job(job_directory)
        except Exception as error:  # noqa: BLE001 — record and continue
            skipped.append(f"{job_directory},{type(error).__name__},{error}")
            continue
        if row is None:
            skipped.append(f"{job_directory},skipped,missing or unsuccessful job")
        else:
            rows.append(row)

    csv_path = arguments.out / "tier2_jobs.csv"
    with csv_path.open("w", encoding="utf-8") as handle:
        handle.write(",".join(CSV_COLUMNS) + "\n")
        for row in rows:
            handle.write(
                ",".join(f"\"{row[column]}\"" if isinstance(row[column], str) else str(row[column])
                         for column in CSV_COLUMNS) + "\n"
            )

    summary_path = arguments.out / "tier2_summary.json"
    summary_path.write_text(json.dumps(summarise(rows), indent=2))

    failures_path = arguments.out / "tier2_failures.csv"
    failures_path.write_text("job_directory,error_type,message\n" + "\n".join(skipped) + "\n")

    print(f"analysed {len(rows)} jobs, skipped {len(skipped)}")
    print(f"  {csv_path}")
    print(f"  {summary_path}")
    print(f"  {failures_path}")


if __name__ == "__main__":
    main()
