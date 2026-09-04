"""Fault-aligned aggregation for existing Tier-2 fused-SHI predictions.

This is a hardened version of the recovered ``tier2_metrics.py`` deposited at
the repository root.  It only reads existing detector artifacts; it does not
extract features, fit a model, or rerun the detector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import beta


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

JOB_COLUMNS = [
    "job_id", "archive", "source_archive_path", "source_archive_exists",
    "member", "sensor", "axes", "experiment_family", "fault_subtype",
    "fault_target", "is_fault_target_sensor", "pipeline_version", "model_type",
    "continuous_branch_enabled", "sampling_rate_hz", "window_samples",
    "stride_samples", "onset_confidence", "baseline_strategy",
    "fault_offset_s", "calibration_warning", "fault_start_ms",
    "calibration_start_ms", "calibration_end_ms", "n_windows",
    "n_calibration", "n_pre_fault", "n_post_fault", "has_pre_fault",
    "calibration_shi_mean", "pre_fault_shi_mean", "post_fault_shi_mean",
    "calibration_breach_count", "calibration_breach_rate",
    "pre_fault_breach_count", "pre_fault_breach_rate",
    "pre_fault_persistent_count", "pre_fault_persistent_rate",
    "post_fault_breach_count", "post_fault_breach_rate",
    "post_fault_persistent_count", "post_fault_persistent_rate", "detected",
    "detection_latency_s", "whole_breach_count", "whole_breach_rate",
    "whole_persistent_count", "whole_persistent_rate", "event_branch_enabled",
    "whole_event_count", "whole_event_rate", "post_event_count",
    "event_detected", "event_latency_s", "combined_detected", "combined_latency_s",
]

EVENT_COLUMNS = [
    "job_id", "archive", "member", "sensor", "experiment_family",
    "fault_subtype", "fault_target", "onset_confidence",
    "is_fault_target_sensor", "event_branch_enabled", "n_windows", "calibration_event_count",
    "pre_fault_event_count", "post_fault_event_count", "whole_event_count",
    "whole_event_rate", "event_detected", "event_latency_s",
]

FAILURE_COLUMNS = [
    "source", "severity", "job_id", "archive", "member", "sensor",
    "job_directory", "error_type", "message",
]


class OutputConsistencyError(ValueError):
    """An artifact cannot safely be used for published aggregation."""


def clopper_pearson(successes: int, trials: int, alpha: float = 0.05) -> tuple[float, float]:
    """Return a two-sided exact Clopper-Pearson binomial interval."""
    if trials < 0 or successes < 0 or successes > trials:
        raise ValueError("require 0 <= successes <= trials")
    if trials == 0:
        return (math.nan, math.nan)
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    upper = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return lower, upper


def experiment_family(archive: str) -> str:
    value = archive.casefold()
    if "a1" in value or "shaker" in value:
        return "A1_orbital_shaker"
    if "a2" in value or "thermal" in value or "fridge" in value:
        return "A2_thermal_fridge"
    if "a3" in value or "emi" in value:
        return "A3_smartphone_emi"
    if "faulty_hardware" in value or "experiment_b" in value:
        return "B_faulty_hardware"
    return "unclassified"


def fault_context(member: str, family: str) -> tuple[str, str]:
    """Return (fault subtype, experimental target) from the member path."""
    lowered = member.casefold()
    choices = (
        ("loose connector", "loose_connector"),
        ("imu bias injection", "imu_bias_injection"),
        ("partial-occlusion", "partial_occlusion"),
        ("partial occlusion", "partial_occlusion"),
        ("accelerated aging", "accelerated_aging"),
        ("supply-voltage-droop", "supply_voltage_droop"),
        ("supply voltage droop", "supply_voltage_droop"),
    )
    subtype = next((label for needle, label in choices if needle in lowered), "")
    if not subtype:
        subtype = {
            "A1_orbital_shaker": "orbital_shaker",
            "A2_thermal_fridge": "thermal_fridge",
            "A3_smartphone_emi": "smartphone_emi",
        }.get(family, "unclassified")

    target = ""
    if family == "B_faulty_hardware":
        parts = [part.strip() for part in member.replace("\\", "/").split("/")]
        needles = {
            "loose connector", "imu bias injection", "partial-occlusion",
            "partial occlusion", "accelerated aging", "supply-voltage-droop",
            "supply voltage droop",
        }
        for index, part in enumerate(parts[:-1]):
            if part.casefold() in needles:
                target = parts[index + 1]
                break
    return subtype, target


def is_fault_target_sensor(target: str, sensor: str) -> bool:
    target_key = target.casefold().strip()
    sensor_key = sensor.casefold().strip()
    if not target_key:
        return False
    if target_key == "imu":
        return sensor_key in {"accelerometer", "gyroscope", "magnetometer"}
    return target_key == sensor_key


def validate_format(root: Path) -> dict[str, Any]:
    path = root / "format.json"
    if not path.is_file():
        raise OutputConsistencyError(f"missing format file: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("record_bytes") != TIER2_DTYPE.itemsize or TIER2_DTYPE.itemsize != 48:
        raise OutputConsistencyError(
            f"format record size is {data.get('record_bytes')!r}; expected {TIER2_DTYPE.itemsize}"
        )
    expected_names = list(TIER2_DTYPE.names or ())
    actual_names = [entry[0] for entry in data.get("numpy_dtype", [])]
    if actual_names != expected_names:
        raise OutputConsistencyError(f"format dtype fields differ: {actual_names!r}")
    return data


def walk_jobs(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("job_report.json")):
        if "_interrupted" not in path.parts:
            yield path.parent


def _rate(values: np.ndarray, mask: np.ndarray) -> float:
    return float(values[mask].mean()) if bool(mask.any()) else math.nan


def _count(values: np.ndarray, mask: np.ndarray) -> int:
    return int(np.count_nonzero(values[mask])) if bool(mask.any()) else 0


def _check_report_rate(name: str, derived: float, report: dict[str, Any], tolerance: float) -> None:
    expected = report.get("alarm_rates", {}).get(name)
    if expected is None or not math.isclose(derived, float(expected), rel_tol=tolerance, abs_tol=tolerance):
        raise OutputConsistencyError(
            f"whole-recording {name} rate mismatch: binary={derived:.12g}, report={expected!r}"
        )


def analyse_job(
    job_directory: Path,
    source_root: Path | None = None,
    rate_tolerance: float = 1e-7,
) -> dict[str, Any]:
    """Parse and validate one successful output job."""
    report_path = job_directory / "job_report.json"
    predictions_path = job_directory / "predictions.bin"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "success":
        raise OutputConsistencyError(f"job report status is {report.get('status')!r}")
    if not predictions_path.is_file():
        raise OutputConsistencyError("predictions.bin is missing")
    byte_count = predictions_path.stat().st_size
    if byte_count % TIER2_DTYPE.itemsize:
        raise OutputConsistencyError(
            f"record-size violation: {byte_count} bytes is not a multiple of {TIER2_DTYPE.itemsize}"
        )
    records = np.fromfile(predictions_path, dtype=TIER2_DTYPE)
    if not len(records):
        raise OutputConsistencyError("predictions.bin is empty")
    if int(report.get("output_windows", -1)) != len(records):
        raise OutputConsistencyError(
            f"window count mismatch: binary={len(records)}, report={report.get('output_windows')!r}"
        )

    timestamps = records["timestamp_ms"].astype(np.int64)
    shi = records["shi"].astype(np.float64)
    flags = records["flags"]
    breach = (flags & FLAG_CONTINUOUS_BREACH) != 0
    persistent = (flags & FLAG_CONTINUOUS_PERSISTENT) != 0
    event = (flags & FLAG_EVENT_ALARM) != 0

    fault_start = int(report["fault_start_ms"])
    calibration_start = int(report["calibration_start_ms"])
    calibration_end = int(report["calibration_end_ms"])
    calibration = (timestamps >= calibration_start) & (timestamps < calibration_end)
    # Calibration takes precedence. This makes the requested partitions disjoint
    # for initial-proxy and tail-recovery jobs whose fault time overlaps it.
    pre_fault = (~calibration) & (timestamps >= calibration_end) & (timestamps < fault_start)
    post_fault = (~calibration) & (timestamps >= fault_start)
    assigned = calibration.astype(np.uint8) + pre_fault.astype(np.uint8) + post_fault.astype(np.uint8)
    if np.any(assigned != 1):
        missing = int(np.count_nonzero(assigned == 0))
        overlap = int(np.count_nonzero(assigned > 1))
        raise OutputConsistencyError(
            f"partition invariant failed: calibration+pre+post != windows "
            f"(unassigned={missing}, overlapping={overlap})"
        )

    whole_breach_rate = float(breach.mean())
    whole_persistent_rate = float(persistent.mean())
    whole_event_rate = float(event.mean())
    _check_report_rate("continuous_breach", whole_breach_rate, report, rate_tolerance)
    _check_report_rate("continuous_persistent", whole_persistent_rate, report, rate_tolerance)
    _check_report_rate("event", whole_event_rate, report, rate_tolerance)

    def latency(mask: np.ndarray, values: np.ndarray) -> float:
        selected = timestamps[mask & values]
        return float((selected.min() - fault_start) / 1000.0) if len(selected) else math.nan

    job = report.get("job", {})
    archive = str(job.get("archive", ""))
    member = str(job.get("member", ""))
    family = experiment_family(archive)
    subtype, target = fault_context(member, family)
    source_path = (source_root / archive).resolve() if source_root else None
    model_type = str(report.get("model_type", ""))
    continuous_branch_enabled = model_type != "event_rate_only"
    event_branch_enabled = "event" in model_type.casefold()
    protocol = job.get("protocol", {})
    detected = bool(np.any(persistent & post_fault))
    event_detected = bool(np.any(event & post_fault))

    row: dict[str, Any] = {
        "job_id": job.get("job_id", job_directory.name),
        "archive": archive,
        "source_archive_path": str(source_path) if source_path else "",
        "source_archive_exists": bool(source_path and source_path.is_file()),
        "member": member,
        "sensor": job.get("sensor", ""),
        "axes": job.get("axes", ""),
        "experiment_family": family,
        "fault_subtype": subtype,
        "fault_target": target,
        "is_fault_target_sensor": is_fault_target_sensor(target, str(job.get("sensor", ""))),
        "pipeline_version": report.get("pipeline_version", ""),
        "model_type": model_type,
        "continuous_branch_enabled": continuous_branch_enabled,
        "sampling_rate_hz": report.get("sampling_rate_hz", ""),
        "window_samples": report.get("window_samples", ""),
        "stride_samples": report.get("stride_samples", ""),
        "onset_confidence": report.get("fault_timestamp_confidence", ""),
        "baseline_strategy": report.get("baseline_strategy", ""),
        "fault_offset_s": protocol.get("fault_offset_s", ""),
        "calibration_warning": report.get("calibration_warning") or "",
        "fault_start_ms": fault_start,
        "calibration_start_ms": calibration_start,
        "calibration_end_ms": calibration_end,
        "n_windows": int(len(records)),
        "n_calibration": int(calibration.sum()),
        "n_pre_fault": int(pre_fault.sum()),
        "n_post_fault": int(post_fault.sum()),
        "has_pre_fault": bool(pre_fault.any()),
        "calibration_shi_mean": _rate(shi, calibration),
        "pre_fault_shi_mean": _rate(shi, pre_fault),
        "post_fault_shi_mean": _rate(shi, post_fault),
        "calibration_breach_count": _count(breach, calibration),
        "calibration_breach_rate": _rate(breach, calibration),
        "pre_fault_breach_count": _count(breach, pre_fault),
        "pre_fault_breach_rate": _rate(breach, pre_fault),
        "pre_fault_persistent_count": _count(persistent, pre_fault),
        "pre_fault_persistent_rate": _rate(persistent, pre_fault),
        "post_fault_breach_count": _count(breach, post_fault),
        "post_fault_breach_rate": _rate(breach, post_fault),
        "post_fault_persistent_count": _count(persistent, post_fault),
        "post_fault_persistent_rate": _rate(persistent, post_fault),
        "detected": detected,
        "detection_latency_s": latency(post_fault, persistent),
        "whole_breach_count": int(breach.sum()),
        "whole_breach_rate": whole_breach_rate,
        "whole_persistent_count": int(persistent.sum()),
        "whole_persistent_rate": whole_persistent_rate,
        "event_branch_enabled": event_branch_enabled,
        "whole_event_count": int(event.sum()),
        "whole_event_rate": whole_event_rate,
        "post_event_count": _count(event, post_fault),
        "event_detected": event_detected,
        "event_latency_s": latency(post_fault, event),
        "_calibration_event_count": _count(event, calibration),
        "_pre_fault_event_count": _count(event, pre_fault),
    }
    row["combined_detected"] = detected if continuous_branch_enabled else event_detected
    row["combined_latency_s"] = (
        row["detection_latency_s"] if continuous_branch_enabled else row["event_latency_s"]
    )
    if row["n_calibration"] + row["n_pre_fault"] + row["n_post_fault"] != row["n_windows"]:
        raise OutputConsistencyError("partition count invariant failed after aggregation")
    return row


def _proportion(successes: int, trials: int) -> dict[str, Any]:
    lo, hi = clopper_pearson(successes, trials)
    return {
        "successes": successes,
        "trials": trials,
        "estimate": successes / trials if trials else None,
        "ci95": [lo, hi] if trials else [None, None],
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    usable = [row for row in rows if row["n_post_fault"] > 0 and row["continuous_branch_enabled"]]
    detected = [row for row in usable if row["detected"]]
    latencies = [float(row["detection_latency_s"]) for row in detected]
    pre_rows = [row for row in rows if row["n_pre_fault"] > 0]
    pre_trials = sum(int(row["n_pre_fault"]) for row in pre_rows)
    pre_flags = sum(int(row["pre_fault_breach_count"]) for row in pre_rows)
    event_usable = [row for row in rows if row["n_post_fault"] > 0 and row["event_branch_enabled"]]
    event_detected = [row for row in event_usable if row["event_detected"]]
    event_latencies = [float(row["event_latency_s"]) for row in event_detected]
    combined_usable = [row for row in rows if row["n_post_fault"] > 0]
    combined_detected = [row for row in combined_usable if row["combined_detected"]]
    return {
        "jobs": len(rows),
        "continuous": {
            "jobs_with_post_fault_evaluation": len(usable),
            "trace_detection": _proportion(len(detected), len(usable)),
            "detected_latency_s": {
            "median": float(np.median(latencies)) if latencies else None,
            "q1": float(np.percentile(latencies, 25)) if latencies else None,
            "q3": float(np.percentile(latencies, 75)) if latencies else None,
            },
        },
        "event_rate": {
            "jobs_with_post_fault_evaluation": len(event_usable),
            "trace_detection": _proportion(len(event_detected), len(event_usable)),
            "detected_latency_s": {
                "median": float(np.median(event_latencies)) if event_latencies else None,
                "q1": float(np.percentile(event_latencies, 25)) if event_latencies else None,
                "q3": float(np.percentile(event_latencies, 75)) if event_latencies else None,
            },
        },
        "branch_appropriate_combined": {
            "jobs_with_post_fault_evaluation": len(combined_usable),
            "trace_detection": _proportion(len(combined_detected), len(combined_usable)),
        },
        "jobs_with_out_of_sample_pre_fault": len(pre_rows),
        "pre_fault_window_breach": _proportion(pre_flags, pre_trials),
        "jobs_without_pre_fault": len(rows) - len(pre_rows),
    }


def summarise(rows: list[dict[str, Any]], source_root: Path | None = None) -> dict[str, Any]:
    confidence_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    subtype_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    target_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        confidence_groups[str(row["onset_confidence"])].append(row)
        subtype_groups[str(row["fault_subtype"])].append(row)
        family_groups[str(row["experiment_family"])].append(row)
        if row["fault_target"]:
            target_groups[f"{row['fault_subtype']}/{row['fault_target']}"] .append(row)

    def confidence_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        offset_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group:
            label = "onset_at_recording_start" if float(row["fault_offset_s"]) == 0 else "delayed_onset"
            offset_groups[label].append(row)
        return {
            "jobs": len(group),
            "by_fault_offset_class": {
                key: _aggregate(value) for key, value in sorted(offset_groups.items())
            },
        }

    missing_expected_subtypes = sorted(
        {"loose_connector", "imu_bias_injection", "partial_occlusion",
         "accelerated_aging", "supply_voltage_droop"}
        - set(subtype_groups)
    )
    return {
        "method": "fault-aligned aggregation of deposited Tier-2 fused-SHI outputs",
        "partition_precedence": "calibration, then pre-fault, then post-fault",
        "detection_definition": "first persistent continuous breach after onset and outside calibration",
        "source_dataset_root": str(source_root.resolve()) if source_root else None,
        "jobs_analysed": len(rows),
        "confidence_counts": dict(sorted(Counter(str(row["onset_confidence"]) for row in rows).items())),
        "baseline_strategy_counts": dict(sorted(Counter(str(row["baseline_strategy"]) for row in rows).items())),
        "event_branch": {
            "jobs_enabled": sum(bool(row["event_branch_enabled"]) for row in rows),
            "jobs_with_any_event_alarm": sum(int(row["whole_event_count"]) > 0 for row in rows),
            "total_event_alarms": sum(int(row["whole_event_count"]) for row in rows),
            "jobs_with_post_onset_event_alarm": sum(bool(row["event_detected"]) for row in rows),
        },
        "publication_guardrail": (
            "No overall pooled performance estimate is emitted: onset-at-start jobs, "
            "delayed-onset jobs, confidence classes, and fault subtypes must remain separate."
        ),
        "jobs_with_out_of_sample_pre_fault": sum(row["n_pre_fault"] > 0 for row in rows),
        "missing_expected_experiment_b_subtypes": missing_expected_subtypes,
        "by_fault_timestamp_confidence": {
            key: confidence_summary(value) for key, value in sorted(confidence_groups.items())
        },
        "by_fault_subtype": {
            key: _aggregate(value) for key, value in sorted(subtype_groups.items())
        },
        "by_experiment_family": {
            key: _aggregate(value) for key, value in sorted(family_groups.items())
        },
        "by_fault_subtype_and_target": {
            key: {
                "all_recorded_sensor_streams": _aggregate(value),
                "fault_target_sensor_streams": _aggregate(
                    [row for row in value if row["is_fault_target_sensor"]]
                ),
            }
            for key, value in sorted(target_groups.items())
        },
    }


def _pipeline_failures(root: Path) -> list[dict[str, Any]]:
    path = root / "latest_run_summary.json"
    if not path.is_file():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for failure in data.get("failures", []):
        job = failure.get("job", {})
        result.append({
            "source": "detector_run",
            "severity": "recorded_failure",
            "job_id": job.get("job_id", ""),
            "archive": job.get("archive", ""),
            "member": job.get("member", ""),
            "sensor": job.get("sensor", ""),
            "job_directory": "",
            "error_type": failure.get("error_type", ""),
            "message": failure.get("message", ""),
        })
    return result


def _write_csv(path: Path, columns: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_metrics(root: Path, out: Path, source_root: Path | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_format(root)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    failures = _pipeline_failures(root)
    consistency_failures: list[dict[str, Any]] = []
    for directory in walk_jobs(root):
        try:
            rows.append(analyse_job(directory, source_root=source_root))
        except Exception as error:  # Continue audit, but do not publish a partial summary.
            report: dict[str, Any] = {}
            try:
                report = json.loads((directory / "job_report.json").read_text(encoding="utf-8"))
            except Exception:
                pass
            job = report.get("job", {})
            consistency_failures.append({
                "source": "aggregation_audit",
                "severity": "fatal_consistency_error",
                "job_id": job.get("job_id", directory.name),
                "archive": job.get("archive", ""),
                "member": job.get("member", ""),
                "sensor": job.get("sensor", ""),
                "job_directory": str(directory),
                "error_type": type(error).__name__,
                "message": str(error),
            })
    failures.extend(consistency_failures)
    _write_csv(out / "tier2_failures.csv", FAILURE_COLUMNS, failures)
    if consistency_failures:
        raise OutputConsistencyError(
            f"{len(consistency_failures)} consistency failure(s); refusing to publish aggregate tables"
        )

    _write_csv(out / "tier2_jobs.csv", JOB_COLUMNS, rows)
    event_rows = [
        {
            **row,
            "calibration_event_count": row["_calibration_event_count"],
            "pre_fault_event_count": row["_pre_fault_event_count"],
            "post_fault_event_count": row["post_event_count"],
        }
        for row in rows
    ]
    _write_csv(out / "tier2_event_branch.csv", EVENT_COLUMNS, event_rows)
    summary = summarise(rows, source_root)
    summary["detector_run_failures"] = len(_pipeline_failures(root))
    (out / "tier2_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return rows, failures


def default_paths() -> tuple[Path, Path, Path]:
    repo = Path(__file__).resolve().parents[1]
    workspace = repo.parents[1]
    root = repo / "outputs" / "tier2_fused_shi" / "hardware"
    return root, repo / "outputs" / "tier2_fused_shi" / "summaries", workspace / "Data" / "hardware_injection"


def main() -> int:
    default_root, default_out, default_source = default_paths()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--out", type=Path, default=default_out)
    parser.add_argument("--source-root", type=Path, default=default_source)
    args = parser.parse_args()
    try:
        rows, failures = build_metrics(args.root.resolve(), args.out.resolve(), args.source_root.resolve())
    except OutputConsistencyError as error:
        print(f"FATAL: {error}")
        return 2
    print(f"Validated and aggregated {len(rows)} successful jobs.")
    print(f"Recorded detector-run failures: {len(failures)}")
    for name in ("tier2_jobs.csv", "tier2_summary.json", "tier2_failures.csv", "tier2_event_branch.csv"):
        print(args.out.resolve() / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
