#!/usr/bin/env python3
"""Run the reconstructed SHIBench reference SHI on hardware-injection ZIPs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np

from hardware_simple_shi import (
    discover_jobs, fingerprint, iter_window_batches, timestamp_bounds,
)
from model_shi.common import write_json_atomic
from shibench_reference_shi.core import (
    FLAG_CONTINUOUS_ALARM, FLAG_EVENT_ALARM, FLAG_LIVENESS_ALARM,
    FLAG_PLAUSIBILITY_ALARM, REFERENCE_DTYPE, ReconstructionConfig,
    ReferenceSHI, extract_reference_features,
)


PIPELINE_VERSION = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("../Data/hardware_injection"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/shibench_reference_shi/hardware"),
    )
    parser.add_argument("--batch", type=int)
    parser.add_argument("--total-batches", type=int, default=10)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, min(2, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--feature-batch-windows", type=int, default=128)
    parser.add_argument("--archive-contains")
    parser.add_argument("--member-contains")
    parser.add_argument("--sensor", action="append")
    parser.add_argument("--fault-start-seconds", type=float)
    parser.add_argument("--calibration-fpr", type=float, default=0.05)
    parser.add_argument("--ewma-alpha", type=float, default=0.15)
    parser.add_argument("--isolation-estimators", type=int, default=64)
    parser.add_argument("--minimum-calibration-windows", type=int, default=20)
    parser.add_argument("--event-branch", choices=("auto", "all", "none"), default="auto")
    parser.add_argument("--plausibility-mad-multiplier", type=float, default=20.0)
    args = parser.parse_args()
    if args.total_batches < 1 or args.workers < 1 or args.stride < 1:
        parser.error("total-batches, workers, and stride must be positive")
    if args.window_size < 16 or args.feature_batch_windows < 1:
        parser.error("invalid window or feature-batch setting")
    if not args.plan_only and args.batch is None:
        parser.error("--batch is required unless --plan-only is used")
    if args.batch is not None and not 1 <= args.batch <= args.total_batches:
        parser.error("--batch must be between 1 and --total-batches")
    if args.fault_start_seconds is not None and args.fault_start_seconds < 0:
        parser.error("fault-start-seconds cannot be negative")
    return args


def reconstruction_config(args: argparse.Namespace) -> ReconstructionConfig:
    config = ReconstructionConfig(
        calibration_false_positive_rate=args.calibration_fpr,
        ewma_alpha=args.ewma_alpha,
        isolation_estimators=args.isolation_estimators,
        minimum_calibration_windows=args.minimum_calibration_windows,
        event_branch=args.event_branch,
        plausibility_mad_multiplier=args.plausibility_mad_multiplier,
    )
    config.validate()
    return config


def make_plan(input_dir: Path, output: Path, args: argparse.Namespace) -> dict:
    jobs = discover_jobs(
        input_dir, args.archive_contains, args.member_contains, args.fault_start_seconds,
    )
    if args.sensor:
        wanted = set(args.sensor)
        jobs = [job for job in jobs if job["sensor"] in wanted]
    if not jobs:
        raise ValueError("no supported hardware sensor streams matched")
    batches = [{"batch": i + 1, "bytes": 0, "jobs": []} for i in range(args.total_batches)]
    for job in sorted(jobs, key=lambda item: (-item["uncompressed_bytes"], item["member"])):
        target = min(batches, key=lambda item: (item["bytes"], item["batch"]))
        target["jobs"].append(job)
        target["bytes"] += job["uncompressed_bytes"]
    for batch in batches:
        batch["jobs"].sort(key=lambda item: (item["archive"], item["member"], item["sensor"]))
    plan = {
        "pipeline_version": PIPELINE_VERSION,
        "method": "reconstructed SHIBench reference SHI",
        "exact_reproduction": False,
        "dataset_fingerprint": fingerprint(input_dir, jobs),
        "total_batches": args.total_batches,
        "total_jobs": len(jobs),
        "total_uncompressed_bytes": sum(job["uncompressed_bytes"] for job in jobs),
        "processing": {
            "window_size": args.window_size,
            "stride": args.stride,
            "feature_batch_windows": args.feature_batch_windows,
            "reconstruction": asdict(reconstruction_config(args)),
        },
        "filters": {
            "archive_contains": args.archive_contains,
            "member_contains": args.member_contains,
            "sensors": sorted(set(args.sensor or [])),
            "fault_start_seconds": args.fault_start_seconds,
        },
        "batches": batches,
    }
    path = output / "hardware_reference_shi_plan.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != plan:
            raise ValueError(f"existing plan differs: {path}; use a new output directory")
        return old
    write_json_atomic(path, plan)
    return plan


def output_paths(output: Path, job: dict) -> tuple[Path, Path, Path]:
    root = output / Path(job["archive"]).stem
    base = root / f"{job['job_id']}__{job['sensor']}"
    return base.with_suffix(".bin"), base.with_suffix(".json"), base.with_suffix(".calibration.joblib")


def calibration_bounds(first: int, last: int, job: dict) -> tuple[int, int]:
    protocol = job["protocol"]
    fault_start = first + round(protocol["fault_offset_s"] * 1000)
    baseline_ms = round(protocol["baseline_seconds"] * 1000)
    if protocol["baseline"] == "tail_recovery":
        return max(first, last - baseline_ms), last + 1
    if protocol["baseline"] == "pre_fault":
        return first, min(last + 1, fault_start)
    return first, min(last + 1, first + baseline_ms)


def process_job(job: dict, config: dict) -> dict:
    started = time.monotonic()
    input_dir, output = Path(config["input_dir"]), Path(config["output_dir"])
    archive = input_dir / job["archive"]
    final, marker, calibration_path = output_paths(output, job)
    if marker.is_file():
        old = json.loads(marker.read_text(encoding="utf-8"))
        if old.get("status") == "success" and old.get("processing_config") == config["processing_config"]:
            if (final.is_file() and final.stat().st_size == old.get("output_bytes")
                    and calibration_path.is_file()):
                return {"status": "skipped", "job": job, "report": old}
    temporary = final.with_suffix(".bin.tmp")
    calibration_temporary = calibration_path.with_suffix(".joblib.tmp")
    # Temporary files have no validity marker and can be left at any size by an
    # interrupted WSL session. They are always regenerated. A final artifact
    # without a marker is also safe to atomically replace; the marker is written
    # last and is the only completion authority.
    temporary.unlink(missing_ok=True)
    calibration_temporary.unlink(missing_ok=True)
    if marker.exists():
        raise FileExistsError(f"incomplete existing reference-SHI output for {job['job_id']}")

    first, last, input_records = timestamp_bounds(archive, job)
    calibration_start, calibration_end = calibration_bounds(first, last, job)
    calibration_features: list[np.ndarray] = []
    calibration_windows: list[np.ndarray] = []
    calibration_timestamps: list[np.ndarray] = []
    for windows, _ends, timestamps, fs in iter_window_batches(
        archive, job, config["window_size"], config["stride"], config["feature_batch_windows"],
    ):
        selected = (timestamps >= calibration_start) & (timestamps < calibration_end)
        if np.any(selected):
            calibration_windows.append(windows[selected])
            calibration_timestamps.append(timestamps[selected])
            calibration_features.append(extract_reference_features(windows[selected], fs))
    if not calibration_features:
        raise ValueError(f"no windows fall inside the calibration interval for {job['member']}")
    features_fit = np.concatenate(calibration_features)
    windows_fit = np.concatenate(calibration_windows)
    timestamps_fit = np.concatenate(calibration_timestamps)
    model = ReferenceSHI(ReconstructionConfig(**config["reconstruction_config"]))
    model.fit(features_fit, windows_fit, timestamps_fit, job["sensor"])
    model.reset()

    final.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    shi_sum = 0.0
    alarm_counts = {"continuous": 0, "event": 0, "liveness": 0, "plausibility": 0}
    try:
        joblib.dump(model, calibration_temporary)
        with temporary.open("xb") as stream:
            for windows, ends, timestamps, fs in iter_window_batches(
                archive, job, config["window_size"], config["stride"],
                config["feature_batch_windows"],
            ):
                features = extract_reference_features(windows, fs)
                scores = model.score(features, windows, timestamps, ends)
                result = np.zeros(len(features), dtype=REFERENCE_DTYPE)
                result["timestamp_ms"] = timestamps
                for field in (
                    "shi", "mahalanobis_health", "isolation_health", "ewma_health",
                    "event_health", "flags",
                ):
                    result[field] = scores[field]
                result["signal"][:, :job["axes"]] = ends[:, :job["axes"]]
                stream.write(result.tobytes())
                rows += len(result)
                shi_sum += float(np.sum(result["shi"]))
                flags = result["flags"]
                alarm_counts["continuous"] += int(np.count_nonzero(flags & FLAG_CONTINUOUS_ALARM))
                alarm_counts["event"] += int(np.count_nonzero(flags & FLAG_EVENT_ALARM))
                alarm_counts["liveness"] += int(np.count_nonzero(flags & FLAG_LIVENESS_ALARM))
                alarm_counts["plausibility"] += int(np.count_nonzero(flags & FLAG_PLAUSIBILITY_ALARM))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(calibration_temporary, calibration_path)
        os.replace(temporary, final)
    except Exception:
        temporary.unlink(missing_ok=True)
        calibration_temporary.unlink(missing_ok=True)
        raise

    protocol = job["protocol"]
    warning = None
    if protocol["baseline"] == "initial_proxy":
        warning = "calibration interval may already contain the physical fault"
    report = {
        "status": "success",
        "pipeline_version": PIPELINE_VERSION,
        "method": "reconstructed SHIBench reference SHI",
        "exact_reproduction": False,
        "job": job,
        "processing_config": config["processing_config"],
        "calibration": model.metadata(),
        "calibration_scope": "within-recording",
        "calibration_warning": warning,
        "first_timestamp_ms": first,
        "last_timestamp_ms": last,
        "fault_start_ms": first + round(protocol["fault_offset_s"] * 1000),
        "fault_start_seconds_from_signal": protocol["fault_offset_s"],
        "fault_timestamp_source": protocol["reason"],
        "fault_timestamp_confidence": protocol["confidence"],
        "baseline_strategy": protocol["baseline"],
        "calibration_start_ms": calibration_start,
        "calibration_end_ms": calibration_end,
        "input_records": input_records,
        "output_windows": rows,
        "output_path": str(final.resolve()),
        "calibration_path": str(calibration_path.resolve()),
        "output_bytes": final.stat().st_size,
        "record_bytes": REFERENCE_DTYPE.itemsize,
        "shi_mean": shi_sum / rows if rows else None,
        "alarm_counts": alarm_counts,
        "alarm_rates": {key: value / rows if rows else None for key, value in alarm_counts.items()},
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json_atomic(marker, report)
    return {"status": "success", "job": job, "report": report}


def next_report(output: Path, batch: int, total: int) -> Path:
    base = output / f"hardware_reference_report_batch_{batch:02d}_of_{total:02d}.json"
    if not base.exists():
        return base
    attempt = 2
    while True:
        candidate = base.with_name(f"{base.stem}_attempt_{attempt:02d}.json")
        if not candidate.exists():
            return candidate
        attempt += 1


def main() -> int:
    args = parse_args()
    input_dir, output = args.input_dir.resolve(), args.output_dir.resolve()
    plan = make_plan(input_dir, output, args)
    print(
        f"Plan: {plan['total_jobs']} hardware sensors, {plan['total_batches']} batches; "
        "reconstructed healthy-calibrated SHI",
        flush=True,
    )
    if args.plan_only:
        return 0
    processing_config = plan["processing"]
    config = {
        "input_dir": str(input_dir), "output_dir": str(output),
        "window_size": args.window_size, "stride": args.stride,
        "feature_batch_windows": args.feature_batch_windows,
        "reconstruction_config": processing_config["reconstruction"],
        "processing_config": processing_config,
    }
    jobs = plan["batches"][args.batch - 1]["jobs"]
    successes, skipped, failures = [], [], []
    started = time.monotonic()

    def accept(job: dict, result: dict | None = None, error: Exception | None = None) -> None:
        if error is not None:
            item = {"job": job, "error_type": type(error).__name__, "message": str(error)}
            failures.append(item)
            print(f"[failed] {job['member']} [{job['sensor']}]: {item['error_type']}: {item['message']}", flush=True)
        else:
            assert result is not None
            (skipped if result["status"] == "skipped" else successes).append(result)
            print(f"[{result['status']}] {job['member']} [{job['sensor']}]", flush=True)

    if args.workers == 1:
        for job in jobs:
            try:
                accept(job, result=process_job(job, config))
            except Exception as error:
                accept(job, error=error)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_job, job, config): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    accept(job, result=future.result())
                except Exception as error:
                    accept(job, error=error)
    report = {
        "pipeline_version": PIPELINE_VERSION,
        "batch": args.batch,
        "total_batches": args.total_batches,
        "workers": args.workers,
        "elapsed_seconds": time.monotonic() - started,
        "successful_jobs": successes,
        "skipped_jobs": skipped,
        "failed_jobs": failures,
    }
    path = next_report(output, args.batch, args.total_batches)
    write_json_atomic(path, report)
    print(
        f"Batch {args.batch}: {len(successes)} succeeded, {len(skipped)} skipped, "
        f"{len(failures)} failed. Report: {path}",
        flush=True,
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
