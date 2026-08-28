#!/usr/bin/env python3
"""Run upstream-model P(healthy) SHI directly on hardware-injection ZIPs."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np

from common import (
    PIPELINE_VERSION, PREDICTION_DTYPE, UPSTREAM_COMMIT, healthy_probability,
    model_ar_setting, write_json_atomic,
)
from build_training_features import extract_multiaxis
from hardware_simple_shi import discover_jobs, fingerprint, iter_window_batches, timestamp_bounds


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("Data/hardware_injection"))
    parser.add_argument("--models-dir", type=Path, default=Path("model_shi/models"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("model_shi/outputs/hardware_predictions"),
    )
    parser.add_argument("--batch", type=int)
    parser.add_argument("--total-batches", type=int, default=10)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--feature-batch-windows", type=int, default=128)
    parser.add_argument("--archive-contains")
    parser.add_argument("--member-contains")
    parser.add_argument("--sensor", action="append")
    parser.add_argument("--fault-start-seconds", type=float)
    args = parser.parse_args()
    if args.total_batches < 1 or args.workers < 1 or args.stride < 1:
        parser.error("total-batches, workers, and stride must be positive")
    if args.window_size < 16 or args.feature_batch_windows < 1:
        parser.error("invalid window or feature-batch setting")
    if args.fault_start_seconds is not None and args.fault_start_seconds < 0:
        parser.error("fault-start-seconds cannot be negative")
    if not args.plan_only and args.batch is None:
        parser.error("--batch is required unless --plan-only is used")
    if args.batch is not None and not 1 <= args.batch <= args.total_batches:
        parser.error("--batch must be between 1 and --total-batches")
    return args


def available_sensors(models: Path) -> set[str]:
    return {path.parent.name for path in models.glob("*/model_metadata.json")}


def ensure_plan(input_dir: Path, models: Path, output: Path, args) -> dict:
    jobs = discover_jobs(
        input_dir, args.archive_contains, args.member_contains, args.fault_start_seconds,
    )
    sensors = available_sensors(models)
    if args.sensor: sensors &= set(args.sensor)
    jobs = [job for job in jobs if job["sensor"] in sensors]
    if not jobs:
        raise ValueError("no hardware jobs with trained sensor models matched")
    include_ar = model_ar_setting(models, sensors)
    batches = [{"batch": i + 1, "bytes": 0, "jobs": []} for i in range(args.total_batches)]
    for job in sorted(jobs, key=lambda item: (-item["uncompressed_bytes"], item["member"])):
        target = min(batches, key=lambda item: (item["bytes"], item["batch"]))
        target["jobs"].append(job); target["bytes"] += job["uncompressed_bytes"]
    for batch in batches: batch["jobs"].sort(key=lambda item: (item["archive"], item["member"], item["sensor"]))
    plan = {
        "pipeline_version": PIPELINE_VERSION, "upstream_commit": UPSTREAM_COMMIT,
        "dataset_fingerprint": fingerprint(input_dir, jobs), "total_batches": args.total_batches,
        "total_jobs": len(jobs), "total_uncompressed_bytes": sum(j["uncompressed_bytes"] for j in jobs),
        "settings": {
            "models_dir": str(models), "sensors": sorted(sensors),
            "window_size": args.window_size, "stride": args.stride,
            "include_ar": include_ar,
            "archive_contains": args.archive_contains, "member_contains": args.member_contains,
            "fault_start_seconds": args.fault_start_seconds,
        },
        "batches": batches,
    }
    path = output / "hardware_prediction_plan.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != plan:
            raise ValueError(f"existing prediction plan differs: {path}; use a new output directory")
        return old
    write_json_atomic(path, plan); return plan


@lru_cache(maxsize=64)
def load_models(models_dir: str, sensor: str):
    root = Path(models_dir) / sensor
    metadata = json.loads((root / "model_metadata.json").read_text(encoding="utf-8"))
    return joblib.load(root / "scaler.joblib"), joblib.load(root / "rf.joblib"), joblib.load(root / "xgb.joblib"), metadata


def output_paths(output: Path, job: dict) -> tuple[Path, Path]:
    root = output / Path(job["archive"]).stem
    base = root / f"{job['job_id']}__{job['sensor']}"
    return base.with_suffix(".bin"), base.with_suffix(".json")


def process_job(job: dict, config: dict) -> dict:
    started = time.monotonic()
    input_dir, output = Path(config["input_dir"]), Path(config["output"])
    archive = input_dir / job["archive"]
    final, marker = output_paths(output, job)
    if marker.is_file():
        old = json.loads(marker.read_text(encoding="utf-8"))
        if old.get("status") == "success" and old.get("processing_config") == config["processing_config"]:
            if final.is_file() and final.stat().st_size == old.get("output_bytes"):
                return {"status": "skipped", "job": job, "report": old}
    temporary = final.with_suffix(".bin.tmp")
    # An interrupt can leave an empty file after it has been created but before
    # the first record is written. It has no recoverable output and must not
    # block a safe rerun. Non-empty partial files remain protected.
    if temporary.is_file() and temporary.stat().st_size == 0:
        temporary.unlink()
    if final.exists() or marker.exists() or temporary.exists():
        raise FileExistsError(f"incomplete existing hardware prediction output for {job['job_id']}")

    first, last, input_records = timestamp_bounds(archive, job)
    scaler, rf, xgb, metadata = load_models(config["models_dir"], job["sensor"])
    final.parent.mkdir(parents=True, exist_ok=True)
    rows = 0; rf_sum = xgb_sum = 0.0; rf_min = xgb_min = 1.0
    try:
        with temporary.open("xb") as stream:
            for windows, ends, timestamps, fs in iter_window_batches(
                archive, job, config["window_size"], config["stride"], config["feature_batch_windows"],
            ):
                features = extract_multiaxis(
                    windows, fs, include_ar=config["include_ar"],
                )
                if features.shape[1] != metadata["training_config"]["feature_count"]:
                    raise ValueError(f"feature width mismatch for {job['sensor']}")
                scaled = scaler.transform(features)
                p_rf, p_xgb = healthy_probability(rf, scaled), healthy_probability(xgb, scaled)
                result = np.zeros(len(features), dtype=PREDICTION_DTYPE)
                result["timestamp_ms"] = timestamps
                result["shi_rf"] = p_rf; result["shi_xgb"] = p_xgb
                result["ground_truth_label"] = 255
                result["scenario"] = 255
                result["signal"][:, : job["axes"]] = ends[:, : job["axes"]]
                stream.write(result.tobytes())
                rows += len(result); rf_sum += float(np.sum(p_rf)); xgb_sum += float(np.sum(p_xgb))
                rf_min = min(rf_min, float(np.min(p_rf))); xgb_min = min(xgb_min, float(np.min(p_xgb)))
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, final)
    except Exception:
        temporary.unlink(missing_ok=True); raise
    protocol = job["protocol"]
    report = {
        "status": "success", "job": job, "processing_config": config["processing_config"],
        "shi_definition": "P(healthy), range 0..1",
        "first_timestamp_ms": first, "last_timestamp_ms": last,
        "fault_start_ms": first + round(protocol["fault_offset_s"] * 1000),
        "fault_start_seconds_from_signal": protocol["fault_offset_s"],
        "fault_timestamp_source": protocol["reason"],
        "fault_timestamp_confidence": protocol["confidence"],
        "input_records": input_records, "rows": rows,
        "output_path": str(final.resolve()), "output_bytes": final.stat().st_size,
        "record_bytes": PREDICTION_DTYPE.itemsize,
        "shi_rf_mean": rf_sum / rows if rows else None, "shi_xgb_mean": xgb_sum / rows if rows else None,
        "shi_rf_min": rf_min if rows else None, "shi_xgb_min": xgb_min if rows else None,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json_atomic(marker, report)
    return {"status": "success", "job": job, "report": report}


def next_report(output: Path, batch: int, total: int) -> Path:
    base = output / f"hardware_report_batch_{batch:02d}_of_{total:02d}.json"
    if not base.exists(): return base
    attempt = 2
    while True:
        path = base.with_name(f"{base.stem}_attempt_{attempt:02d}.json")
        if not path.exists(): return path
        attempt += 1


def main() -> int:
    args = parse_args()
    input_dir, models, output = args.input_dir.resolve(), args.models_dir.resolve(), args.output_dir.resolve()
    plan = ensure_plan(input_dir, models, output, args)
    print(f"Plan: {plan['total_jobs']} hardware sensors, {plan['total_batches']} batches, P(healthy) 0..1")
    if args.plan_only: return 0
    processing_config = {
        "models_dir": str(models), "window_size": args.window_size, "stride": args.stride,
        "feature_batch_windows": args.feature_batch_windows,
        "include_ar": plan["settings"]["include_ar"],
    }
    config = {"input_dir": str(input_dir), "output": str(output), **processing_config,
              "processing_config": processing_config}
    jobs = plan["batches"][args.batch - 1]["jobs"]
    success, skipped, failed = [], [], []
    started = time.monotonic()
    def accept(job, result=None, error=None):
        if error:
            failed.append({"job": job, "error_type": type(error).__name__, "message": str(error)})
            print(f"[failed] {job['member']} [{job['sensor']}]: {type(error).__name__}: {error}", flush=True)
        else:
            (skipped if result["status"] == "skipped" else success).append(result)
            print(f"[{result['status']}] {job['member']} [{job['sensor']}]", flush=True)
    if args.workers == 1:
        for job in jobs:
            try: accept(job, result=process_job(job, config))
            except Exception as exc: accept(job, error=exc)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_job, job, config): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try: accept(job, result=future.result())
                except Exception as exc: accept(job, error=exc)
    report = {
        "pipeline_version": PIPELINE_VERSION, "batch": args.batch, "total_batches": args.total_batches,
        "elapsed_seconds": time.monotonic() - started, "successful_jobs": success,
        "skipped_jobs": skipped, "failed_jobs": failed,
    }
    path = next_report(output, args.batch, args.total_batches); write_json_atomic(path, report)
    print(f"Batch {args.batch}: {len(success)} succeeded, {len(skipped)} skipped, {len(failed)} failed. Report: {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
