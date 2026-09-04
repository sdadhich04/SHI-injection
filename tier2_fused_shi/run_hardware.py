#!/usr/bin/env python3
"""Run the recovered fused SHI on Project SHIELD Tier-2 hardware archives."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import joblib
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parents[1]
NOISE_CODE = Path(
    os.environ.get("NOISE_INJECTION_CODE", WORKSPACE_ROOT / "noise_injection")
).resolve()
for candidate in (str(PACKAGE_ROOT), str(NOISE_CODE)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from hardware_simple_shi import discover_jobs, fingerprint, iter_records, iter_window_batches  # noqa: E402
from canonical_brb.runtime import (  # noqa: E402
    completed_report, distribute_jobs, quarantine_incomplete, write_json_atomic,
)
from tier2_fused_shi.core import (  # noqa: E402
    FLAG_CONTINUOUS_BREACH, FLAG_CONTINUOUS_PERSISTENT, FLAG_EVENT_ALARM,
    METHOD_NAME, PIPELINE_VERSION, TIER2_DTYPE, EventRateModel, FusedConfig,
    FusedSHI, event_rates, extract_features, feature_names,
)


DEFAULT_OUTPUT = PACKAGE_ROOT / "outputs/tier2_fused_shi/hardware"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=WORKSPACE_ROOT / "Data/hardware_injection")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--batch", type=int)
    group.add_argument("--all-batches", action="store_true")
    parser.add_argument("--total-batches", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--feature-batch-windows", type=int, default=32)
    parser.add_argument("--minimum-calibration-windows", type=int, default=20)
    parser.add_argument("--event-branch", choices=("auto", "all", "none"), default="auto")
    parser.add_argument("--no-save-model", action="store_true")
    parser.add_argument("--archive-contains")
    parser.add_argument("--member-contains")
    parser.add_argument("--sensor", action="append")
    parser.add_argument("--fault-start-seconds", type=float)
    args = parser.parse_args()
    if args.total_batches < 1 or args.workers < 1:
        parser.error("total-batches and workers must be positive")
    if args.window_seconds <= 0 or not 0 <= args.overlap < 1:
        parser.error("window-seconds must be positive and overlap must be in [0, 1)")
    if args.feature_batch_windows < 1 or args.minimum_calibration_windows < 2:
        parser.error("invalid feature batch or minimum calibration size")
    if args.fault_start_seconds is not None and args.fault_start_seconds < 0:
        parser.error("fault-start-seconds cannot be negative")
    if not args.plan_only and args.batch is None and not args.all_batches:
        parser.error("choose --batch N or --all-batches")
    if args.batch is not None and not 1 <= args.batch <= args.total_batches:
        parser.error("--batch must be between 1 and --total-batches")
    return args


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def reference_provenance() -> dict:
    reference = WORKSPACE_ROOT / "shield-tier1-shi"
    paper = WORKSPACE_ROOT / "alperen-paper-v3 (2).docx"
    files = [reference / "src/pipeline.py", reference / "src/v3_pipeline.py"]
    return {
        "recovered_reference": str(reference),
        "reference_files": [
            {"path": str(path), "sha256": _sha256(path)} for path in files
        ],
        "draft_paper": {"path": str(paper), "sha256": _sha256(paper)},
        "continuous_formula": "exact port of recovered _fit_one/_score and feature front end",
        "hardware_adapter": "sampling-rate inference, ZIP streaming, protocol baseline/onset, batching",
        "event_rate_formula": "transparent Tier-2 adaptation; exact equation absent from recovered source",
    }


def processing_settings(args: argparse.Namespace) -> dict:
    fused = FusedConfig(minimum_calibration_windows=args.minimum_calibration_windows)
    fused.validate()
    return {
        "window_seconds": args.window_seconds,
        "overlap": args.overlap,
        "feature_batch_windows": args.feature_batch_windows,
        "fused_config": asdict(fused),
        "event_branch": args.event_branch,
        "save_model": not args.no_save_model,
        "record_bytes": TIER2_DTYPE.itemsize,
    }


def make_plan(input_dir: Path, args: argparse.Namespace) -> dict:
    jobs = discover_jobs(
        input_dir, args.archive_contains, args.member_contains, args.fault_start_seconds,
    )
    if args.sensor:
        wanted = set(args.sensor)
        jobs = [job for job in jobs if job["sensor"] in wanted]
    if not jobs:
        raise ValueError("no supported Tier-2 hardware sensor streams matched")
    settings = processing_settings(args)
    return {
        "pipeline": "tier2_recovered_fused_shi",
        "pipeline_version": PIPELINE_VERSION,
        "method": METHOD_NAME,
        "dataset_fingerprint": fingerprint(input_dir, jobs),
        "total_batches": args.total_batches,
        "total_jobs": len(jobs),
        "total_uncompressed_bytes": sum(job["uncompressed_bytes"] for job in jobs),
        "settings": settings,
        "filters": {
            "archive_contains": args.archive_contains,
            "member_contains": args.member_contains,
            "sensors": sorted(set(args.sensor or [])),
            "fault_start_seconds": args.fault_start_seconds,
        },
        "provenance": reference_provenance(),
        "batches": distribute_jobs(jobs, args.total_batches, "uncompressed_bytes"),
    }


def _format_description() -> dict:
    return {
        "format": "fixed-width little-endian, headerless",
        "record_bytes": TIER2_DTYPE.itemsize,
        "numpy_dtype": TIER2_DTYPE.descr,
        "flags": {
            "bit_0": "continuous SHI below sensor-specific threshold",
            "bit_1": "continuous threshold breached for at least two consecutive windows",
            "bit_2": "event-rate health at or below its calibration threshold",
        },
        "component_fields": "health values: 1 is calibration-like, 0 is anomalous",
        "continuous_shi": "1 - equal-weight mean of normalized branch anomaly scores",
    }


def ensure_plan(input_dir: Path, output: Path, args: argparse.Namespace) -> dict:
    expected = make_plan(input_dir, args)
    path = output / "batch_plan.json"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != expected:
            raise ValueError(
                f"existing plan differs from data/settings: {path}; use a new output directory"
            )
    else:
        write_json_atomic(path, expected)

    jobs = [job for batch in expected["batches"] for job in batch["jobs"]]
    baseline_counts = Counter(job["protocol"]["baseline"] for job in jobs)
    confidence_counts = Counter(job["protocol"]["confidence"] for job in jobs)
    write_json_atomic(output / "format.json", _format_description())
    write_json_atomic(output / "manifest.json", {
        "method": METHOD_NAME,
        "dataset_fingerprint": expected["dataset_fingerprint"],
        "job_count": len(jobs),
        "baseline_strategy_counts": dict(sorted(baseline_counts.items())),
        "onset_confidence_counts": dict(sorted(confidence_counts.items())),
        "jobs": jobs,
        "limitations": [
            "initial_proxy calibration may already contain the physical fault",
            "tail_recovery is retrospective and not online detection",
            "protocol-family onset is less certain than an operator event timestamp",
        ],
    })
    return expected


def inspect_stream(archive: Path, job: dict) -> tuple[int, int, int, float]:
    first = last = previous = None
    count = 0
    positive_differences: list[int] = []
    for timestamp, _value in iter_records(archive, job):
        if first is None:
            first = timestamp
        if previous is not None:
            if timestamp < previous:
                raise ValueError(
                    f"timestamps decrease in {job['member']}: {previous} -> {timestamp}"
                )
            if timestamp > previous and len(positive_differences) < 100_000:
                positive_differences.append(timestamp - previous)
        previous = timestamp
        last = timestamp
        count += 1
    if first is None or last is None or not positive_differences:
        raise ValueError(f"empty stream or no positive timestamp deltas: {job['member']}")
    sampling_rate = 1000.0 / float(np.median(np.asarray(positive_differences)))
    if not np.isfinite(sampling_rate) or sampling_rate <= 0:
        raise ValueError(f"invalid sampling rate for {job['member']}: {sampling_rate}")
    return int(first), int(last), count, sampling_rate


def calibration_bounds(first: int, last: int, job: dict) -> tuple[int, int]:
    protocol = job["protocol"]
    fault_start = first + round(float(protocol["fault_offset_s"]) * 1000.0)
    duration = round(float(protocol["baseline_seconds"]) * 1000.0)
    if protocol["baseline"] == "tail_recovery":
        return max(first, last - duration), last + 1
    if protocol["baseline"] == "pre_fault":
        return first, min(last + 1, fault_start)
    return first, min(last + 1, first + duration)


def _event_enabled(job: dict, mode: str) -> bool:
    return mode == "all" or (mode == "auto" and job["sensor"] == "vibration")


def output_directory(root: Path, job: dict) -> Path:
    return root / Path(job["archive"]).stem / job["job_id"]


def process_job(job: dict, config: dict) -> dict:
    started = time.monotonic()
    input_root, output_root = Path(config["input"]), Path(config["output"])
    archive = input_root / job["archive"]
    destination = output_directory(output_root, job)
    marker = destination / "job_report.json"
    processing_config = config["processing_config"]
    previous = completed_report(marker, processing_config)
    if previous:
        return {"status": "skipped", "job": job, "report": previous}
    recovered = quarantine_incomplete(destination, output_root)
    destination.mkdir(parents=True, exist_ok=True)

    archive_signature = [archive.stat().st_size, archive.stat().st_mtime_ns]
    first, last, input_records, sampling_rate = inspect_stream(archive, job)
    window_size = max(8, int(round(config["window_seconds"] * sampling_rate)))
    stride = max(1, int(round(window_size * (1.0 - config["overlap"]))))
    calibration_start, calibration_end = calibration_bounds(first, last, job)

    calibration_features: list[np.ndarray] = []
    calibration_event_rates: list[np.ndarray] = []
    for windows, _ends, timestamps, _observed_fs in iter_window_batches(
        archive, job, window_size, stride, config["feature_batch_windows"],
    ):
        selected = (timestamps >= calibration_start) & (timestamps < calibration_end)
        if np.any(selected):
            calibration_features.append(extract_features(windows[selected]))
            calibration_event_rates.append(event_rates(windows[selected]))
    if not calibration_features:
        raise ValueError(f"no windows fall inside calibration interval for {job['member']}")
    features_fit = np.concatenate(calibration_features)
    event_rates_fit = np.concatenate(calibration_event_rates)

    continuous: FusedSHI | None = None
    continuous_error = None
    try:
        continuous = FusedSHI(FusedConfig(**config["fused_config"]))
        continuous.fit(features_fit, feature_names(job["axes"]))
    except ValueError as error:
        if "no finite varying features" not in str(error) or job["sensor"] != "vibration":
            raise
        continuous = None
        continuous_error = str(error)

    enabled_event = _event_enabled(job, config["event_branch"])
    event_model = EventRateModel().fit(event_rates_fit) if enabled_event else None
    if continuous is None and event_model is None:
        raise ValueError("continuous SHI is undefined and event branch is disabled")
    if continuous is not None:
        continuous.reset()

    pending_model = destination / "model.joblib.pending"
    final_model = destination / "model.joblib"
    pending_metadata = destination / "model_metadata.json.pending"
    final_metadata = destination / "model_metadata.json"
    pending_predictions = destination / "predictions.bin.pending"
    final_predictions = destination / "predictions.bin"
    model_bundle = {"continuous": continuous, "event_rate": event_model}
    model_metadata = {
        "method": METHOD_NAME,
        "model_type": "event_rate_only" if continuous is None else (
            "fused_continuous_plus_event" if event_model is not None else "fused_continuous"
        ),
        "continuous": continuous.metadata() if continuous is not None else None,
        "continuous_unavailable_reason": continuous_error,
        "event_rate": event_model.metadata() if event_model is not None else None,
        "window": {
            "seconds": config["window_seconds"], "overlap": config["overlap"],
            "samples": window_size, "stride_samples": stride,
            "estimated_sampling_rate_hz": sampling_rate,
        },
        "calibration": {
            "strategy": job["protocol"]["baseline"],
            "start_ms": calibration_start, "end_ms": calibration_end,
            "window_count": len(features_fit),
        },
        "provenance": config["provenance"],
    }

    rows = 0
    shi_total = 0.0
    finite_shi_rows = 0
    alarm_counts = {"continuous_breach": 0, "continuous_persistent": 0, "event": 0}
    first_detection = {"continuous_breach_ms": None, "continuous_persistent_ms": None, "event_ms": None}
    try:
        if config["save_model"]:
            joblib.dump(model_bundle, pending_model, compress=3)
        write_json_atomic(pending_metadata, model_metadata)
        with pending_predictions.open("xb") as stream:
            for windows, ends, timestamps, _observed_fs in iter_window_batches(
                archive, job, window_size, stride, config["feature_batch_windows"],
            ):
                result = np.zeros(len(windows), dtype=TIER2_DTYPE)
                result["timestamp_ms"] = timestamps
                result["signal"][:, : job["axes"]] = ends[:, : job["axes"]]
                flags = np.zeros(len(windows), dtype=np.uint8)
                if continuous is not None:
                    scores = continuous.score(extract_features(windows))
                    for field in ("shi", "isolation_health", "mahalanobis_health", "ewma_health"):
                        result[field] = scores[field]
                    result["alarm_threshold"] = continuous.alarm_threshold
                    flags |= scores["continuous_breach"].astype(np.uint8) * FLAG_CONTINUOUS_BREACH
                    flags |= (
                        scores["continuous_persistent"].astype(np.uint8)
                        * FLAG_CONTINUOUS_PERSISTENT
                    )
                else:
                    for field in (
                        "shi", "isolation_health", "mahalanobis_health", "ewma_health",
                        "alarm_threshold",
                    ):
                        result[field] = np.nan
                if event_model is not None:
                    event_health = event_model.score_rates(event_rates(windows))
                    # Event health is clipped to zero and its calibration P5 can
                    # consequently equal zero.  Include equality so the most
                    # anomalous event-rate windows remain detectable.
                    event_alarm = event_health <= event_model.alarm_threshold
                    result["event_health"] = event_health
                    flags |= event_alarm.astype(np.uint8) * FLAG_EVENT_ALARM
                else:
                    result["event_health"] = 1.0
                result["flags"] = flags
                stream.write(result.tobytes())

                finite = np.isfinite(result["shi"])
                finite_shi_rows += int(np.count_nonzero(finite))
                shi_total += float(np.sum(result["shi"][finite]))
                rows += len(result)
                masks = {
                    "continuous_breach": (flags & FLAG_CONTINUOUS_BREACH) != 0,
                    "continuous_persistent": (flags & FLAG_CONTINUOUS_PERSISTENT) != 0,
                    "event": (flags & FLAG_EVENT_ALARM) != 0,
                }
                for name, mask in masks.items():
                    alarm_counts[name] += int(np.count_nonzero(mask))
                    if first_detection[f"{name}_ms"] is None and np.any(mask):
                        first_detection[f"{name}_ms"] = int(timestamps[np.flatnonzero(mask)[0]])
            stream.flush()
            os.fsync(stream.fileno())
        if config["save_model"]:
            os.replace(pending_model, final_model)
        os.replace(pending_metadata, final_metadata)
        os.replace(pending_predictions, final_predictions)
    except Exception:
        for path in (pending_model, pending_metadata, pending_predictions):
            path.unlink(missing_ok=True)
        raise

    if archive_signature != [archive.stat().st_size, archive.stat().st_mtime_ns]:
        raise RuntimeError("input archive changed during processing")
    protocol = job["protocol"]
    warning = None
    if protocol["baseline"] == "initial_proxy":
        warning = "calibration may already contain the physical fault; exploratory result"
    elif protocol["baseline"] == "tail_recovery":
        warning = "recovery-tail calibration is retrospective, not online detection"
    fault_timestamp = first + round(float(protocol["fault_offset_s"]) * 1000.0)
    outputs = [
        {"artifact": "predictions", "path": str(final_predictions), "bytes": final_predictions.stat().st_size},
        {"artifact": "model_metadata", "path": str(final_metadata), "bytes": final_metadata.stat().st_size},
    ]
    if config["save_model"]:
        outputs.append({"artifact": "fitted_model", "path": str(final_model), "bytes": final_model.stat().st_size})
    report = {
        "status": "success",
        "pipeline_version": PIPELINE_VERSION,
        "method": METHOD_NAME,
        "job": job,
        "processing_config": processing_config,
        "model_type": model_metadata["model_type"],
        "sampling_rate_hz": sampling_rate,
        "window_samples": window_size,
        "stride_samples": stride,
        "first_timestamp_ms": first,
        "last_timestamp_ms": last,
        "fault_start_ms": fault_timestamp,
        "fault_start_seconds_from_signal": protocol["fault_offset_s"],
        "fault_timestamp_source": protocol["reason"],
        "fault_timestamp_confidence": protocol["confidence"],
        "baseline_strategy": protocol["baseline"],
        "calibration_scope": "within-recording",
        "calibration_start_ms": calibration_start,
        "calibration_end_ms": calibration_end,
        "calibration_warning": warning,
        "input_records": input_records,
        "output_windows": rows,
        "record_bytes": TIER2_DTYPE.itemsize,
        "mean_shi": shi_total / finite_shi_rows if finite_shi_rows else None,
        "alarm_counts": alarm_counts,
        "alarm_rates": {name: count / rows if rows else None for name, count in alarm_counts.items()},
        "first_detection": first_detection,
        "recovered_interrupted_output": str(recovered) if recovered else None,
        "elapsed_seconds": time.monotonic() - started,
        "outputs": outputs,
    }
    write_json_atomic(marker, report)
    return {"status": "success", "job": job, "report": report}


def next_batch_report(output: Path, batch: int) -> Path:
    base = output / f"batch_{batch:03d}_report.json"
    if not base.exists():
        return base
    attempt = 2
    while True:
        candidate = output / f"batch_{batch:03d}_report_attempt_{attempt:03d}.json"
        if not candidate.exists():
            return candidate
        attempt += 1


def run_batch(batch: dict, plan: dict, input_dir: Path, output: Path, args: argparse.Namespace) -> dict:
    config = {
        "input": str(input_dir), "output": str(output),
        "window_seconds": plan["settings"]["window_seconds"],
        "overlap": plan["settings"]["overlap"],
        "feature_batch_windows": plan["settings"]["feature_batch_windows"],
        "fused_config": plan["settings"]["fused_config"],
        "event_branch": plan["settings"]["event_branch"],
        "save_model": plan["settings"]["save_model"],
        "processing_config": plan["settings"],
        "provenance": plan["provenance"],
    }
    successes: list[dict] = []
    skipped: list[dict] = []
    failures: list[dict] = []
    jobs = batch["jobs"]
    started = time.monotonic()
    print(f"Tier-2 batch {batch['batch']}/{plan['total_batches']}: {len(jobs)} jobs, workers={args.workers}")

    def accept(job: dict, result: dict | None = None, error: Exception | None = None) -> None:
        if error is not None:
            failure = {"job": job, "error_type": type(error).__name__, "message": str(error)}
            failures.append(failure)
            print(f"[ERROR] {job['member']} [{job['sensor']}]: {failure['message']}", flush=True)
            return
        assert result is not None
        (skipped if result["status"] == "skipped" else successes).append(result)
        print(f"[{result['status'].upper()}] {job['member']} [{job['sensor']}]", flush=True)

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
        "batch": batch["batch"], "total_batches": plan["total_batches"],
        "workers": args.workers, "elapsed_seconds": time.monotonic() - started,
        "successful_jobs": successes, "skipped_jobs": skipped, "failed_jobs": failures,
    }
    path = next_batch_report(output, batch["batch"])
    write_json_atomic(path, report)
    print(
        f"Batch {batch['batch']}: {len(successes)} success, {len(skipped)} skipped, "
        f"{len(failures)} failed. Report: {path}", flush=True,
    )
    return report


def main() -> int:
    args = parse_args()
    input_dir, output = args.input_dir.resolve(), args.output_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")
    if output == input_dir or input_dir in output.parents:
        raise ValueError("output must be outside the hardware input directory")
    plan = ensure_plan(input_dir, output, args)
    print(
        f"Plan: {plan['total_jobs']} hardware sensors, {plan['total_batches']} batches, "
        "recovered fused Tier-2 SHI", flush=True,
    )
    if args.plan_only:
        for batch in plan["batches"]:
            print(f"Batch {batch['batch']}: {len(batch['jobs'])} jobs, {batch['weight']/2**20:.1f} MiB")
        return 0

    selected = plan["batches"] if args.all_batches else [plan["batches"][args.batch - 1]]
    reports = [run_batch(batch, plan, input_dir, output, args) for batch in selected]
    failures = [failure for report in reports for failure in report["failed_jobs"]]
    successes = sum(len(report["successful_jobs"]) for report in reports)
    skipped = sum(len(report["skipped_jobs"]) for report in reports)
    summary = {
        "pipeline_version": PIPELINE_VERSION,
        "selected_batches": [report["batch"] for report in reports],
        "successful_jobs": successes, "skipped_jobs": skipped,
        "failed_jobs": len(failures), "failures": failures,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    write_json_atomic(output / "latest_run_summary.json", summary)
    print(f"Selected batches complete: {successes} success, {skipped} skipped, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
