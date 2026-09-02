#!/usr/bin/env python3
"""Run canonical BRB-r SHI directly on hardware-injection ZIP archives."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parents[1]
NOISE_CODE = Path(os.environ.get("NOISE_INJECTION_CODE", WORKSPACE_ROOT / "noise_injection")).resolve()
for candidate in (str(PACKAGE_ROOT), str(NOISE_CODE)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from hardware_simple_shi import (  # noqa: E402
    discover_jobs, fingerprint, iter_window_batches, timestamp_bounds,
)
from canonical_brb.core import (  # noqa: E402
    BinaryEventModel, fit_binary_event, fit_brb, score_binary_event, score_brb,
)
from canonical_brb.features import UPSTREAM_COMMIT, extract_batch  # noqa: E402
from canonical_brb.runtime import (  # noqa: E402
    PIPELINE_VERSION, Reservoir, completed_report, distribute_jobs, stable_seed,
    quarantine_incomplete, write_json_atomic,
)


HARDWARE_DTYPE = np.dtype([
    ("timestamp_ms", "<u8"), ("shi", "<f4"), ("raw_distance", "<f4"),
    ("degradation_score", "<f4"), ("signal", "<f4", (3,)),
], align=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=WORKSPACE_ROOT / "Data/hardware_injection")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "outputs/canonical_brb/hardware")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--batch", type=int)
    group.add_argument("--all-batches", action="store_true")
    parser.add_argument("--total-batches", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--feature-batch-windows", type=int, default=64)
    parser.add_argument("--max-calibration-windows", type=int, default=4096)
    parser.add_argument("--healthy-quantile", type=float, default=0.95)
    parser.add_argument("--wavelet-level", type=int, default=5)
    parser.add_argument("--no-adev", action="store_true")
    parser.add_argument("--archive-contains")
    parser.add_argument("--member-contains")
    parser.add_argument("--fault-start-seconds", type=float)
    args = parser.parse_args()
    if args.total_batches < 1 or args.workers < 1 or args.stride < 1:
        parser.error("batch counts, workers, and stride must be positive")
    if args.window_size < 16 or args.feature_batch_windows < 1 or args.max_calibration_windows < 4:
        parser.error("invalid window or calibration setting")
    if not 0.5 <= args.healthy_quantile < 1.0:
        parser.error("--healthy-quantile must be in [0.5, 1)")
    if not args.plan_only and args.batch is None and not args.all_batches:
        parser.error("choose --batch N or --all-batches")
    if args.batch is not None and not 1 <= args.batch <= args.total_batches:
        parser.error("--batch must be between 1 and --total-batches")
    return args


def make_plan(input_dir: Path, args: argparse.Namespace) -> dict:
    jobs = discover_jobs(input_dir, args.archive_contains, args.member_contains, args.fault_start_seconds)
    if not jobs:
        raise ValueError("no supported hardware sensor streams matched")
    settings = {
        "window_size": args.window_size, "stride": args.stride,
        "feature_batch_windows": args.feature_batch_windows,
        "max_calibration_windows": args.max_calibration_windows,
        "healthy_quantile": args.healthy_quantile, "wavelet_level": args.wavelet_level,
        "include_adev": not args.no_adev,
    }
    return {
        "pipeline": "canonical_brb_hardware", "pipeline_version": PIPELINE_VERSION,
        "upstream_commit": UPSTREAM_COMMIT, "dataset_fingerprint": fingerprint(input_dir, jobs),
        "total_batches": args.total_batches, "total_jobs": len(jobs), "settings": settings,
        "selection": {"archive_contains": args.archive_contains, "member_contains": args.member_contains,
                      "fault_start_seconds": args.fault_start_seconds},
        "batches": distribute_jobs(jobs, args.total_batches, "uncompressed_bytes"),
    }


def ensure_plan(input_dir: Path, output: Path, args: argparse.Namespace) -> dict:
    expected = make_plan(input_dir, args)
    path = output / "batch_plan.json"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != expected:
            raise ValueError(f"existing plan differs from inputs/settings: {path}; use a new output directory")
        return current
    write_json_atomic(path, expected)
    return expected


def _in_baseline(timestamp: int, first: int, last: int, protocol: dict) -> bool:
    duration_ms = int(float(protocol["baseline_seconds"]) * 1000.0)
    strategy = protocol["baseline"]
    if strategy == "tail_recovery":
        return timestamp >= last - duration_ms
    return timestamp <= first + duration_ms


def process_job(job: dict, config: dict) -> dict:
    started = time.monotonic()
    input_dir, output_root = Path(config["input"]), Path(config["output"])
    archive = input_dir / job["archive"]
    archive_stat = [archive.stat().st_size, archive.stat().st_mtime_ns]
    destination = output_root / archive.stem / job["job_id"]
    marker = destination / "job_report.json"
    processing_config = {key: value for key, value in config.items() if key not in {"input", "output"}}
    previous = completed_report(marker, processing_config)
    if previous:
        return {"status": "skipped", "job": job, "report": previous}
    recovered = quarantine_incomplete(destination, output_root)
    destination.mkdir(parents=True, exist_ok=True)
    first, last, sample_count = timestamp_bounds(archive, job)
    reservoir = Reservoir(config["max_calibration_windows"], stable_seed(job["job_id"], job["sensor"]))
    raw_reservoir = (
        Reservoir(config["max_calibration_windows"], stable_seed(job["job_id"], job["sensor"], "raw"))
        if job["sensor"] == "vibration" else None
    )
    for windows, _signal, timestamps, fs in iter_window_batches(
        archive, job, config["window_size"], config["stride"], config["feature_batch_windows"],
    ):
        mask = np.asarray([_in_baseline(int(timestamp), first, last, job["protocol"]) for timestamp in timestamps])
        if not np.any(mask):
            continue
        if raw_reservoir is not None:
            raw_reservoir.update(
                [f"sample_{index}" for index in range(windows.shape[1])],
                windows[mask, :, 0],
            )
        names, features = extract_batch(
            windows[mask], fs, wavelet_level=config["wavelet_level"], include_adev=config["include_adev"],
        )
        reservoir.update(names, features)
    try:
        model = fit_brb(
            reservoir.names or [], reservoir.matrix(), healthy_quantile=config["healthy_quantile"],
            upstream_commit=UPSTREAM_COMMIT,
        )
    except ValueError as error:
        if "no finite varying features" not in str(error) or raw_reservoir is None:
            raise
        model = fit_binary_event(
            raw_reservoir.matrix(), healthy_quantile=config["healthy_quantile"],
            upstream_commit=UPSTREAM_COMMIT,
        )
    final_model = destination / "model.json"
    pending_model = destination / "model.json.pending"
    model.to_json(pending_model)

    final = destination / "predictions.bin"
    temporary = final.with_name(final.name + ".tmp")
    count, minimum, total = 0, 1.0, 0.0
    try:
        with temporary.open("wb") as stream:
            for windows, signal, timestamps, fs in iter_window_batches(
                archive, job, config["window_size"], config["stride"], config["feature_batch_windows"],
            ):
                if isinstance(model, BinaryEventModel):
                    health, raw, degradation = score_binary_event(model, windows)
                else:
                    names, features = extract_batch(
                        windows, fs, wavelet_level=config["wavelet_level"], include_adev=config["include_adev"],
                    )
                    health, raw, degradation = score_brb(model, names, features)
                records = np.zeros(len(health), dtype=HARDWARE_DTYPE)
                records["timestamp_ms"] = timestamps
                records["shi"] = health
                records["raw_distance"] = raw
                records["degradation_score"] = degradation
                records["signal"][:, :job["axes"]] = signal
                stream.write(records.tobytes())
                count += len(records); minimum = min(minimum, float(np.min(health))); total += float(np.sum(health))
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, final)
        os.replace(pending_model, final_model)
    except Exception:
        temporary.unlink(missing_ok=True)
        pending_model.unlink(missing_ok=True)
        raise
    if archive_stat != [archive.stat().st_size, archive.stat().st_mtime_ns]:
        raise RuntimeError("archive changed during processing")
    output = {"path": str(final), "bytes": final.stat().st_size, "records": count,
              "record_bytes": HARDWARE_DTYPE.itemsize, "minimum_shi": minimum,
              "mean_shi": total / count if count else None}
    report = {
        "status": "success", "job": job, "processing_config": processing_config,
        "elapsed_seconds": time.monotonic() - started, "source_samples": sample_count,
        "calibration_windows_seen": reservoir.seen, "calibration_windows_used": len(reservoir.rows),
        "model_type": model.model_type,
        "recovered_interrupted_output": str(recovered) if recovered else None,
        "fault_timestamp_ms": first + int(float(job["protocol"]["fault_offset_s"]) * 1000.0),
        "outputs": [output, {"artifact": "model", "path": str(final_model),
                             "bytes": final_model.stat().st_size}],
    }
    write_json_atomic(marker, report)
    return {"status": "success", "job": job, "report": report}


def run_batch(batch: dict, plan: dict, input_dir: Path, output: Path, args: argparse.Namespace) -> bool:
    config = {"input": str(input_dir), "output": str(output), **plan["settings"]}
    successes, skipped, failures = [], [], []
    jobs = batch["jobs"]
    print(f"Hardware batch {batch['batch']}/{plan['total_batches']}: {len(jobs)} jobs, workers={args.workers}")
    if args.workers == 1:
        completed = ((job, None, None) for job in jobs)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        futures = {executor.submit(process_job, job, config): job for job in jobs}
        completed = ((futures[future], future, executor) for future in as_completed(futures))
    try:
        for index, (job, future, _executor) in enumerate(completed, start=1):
            try:
                result = process_job(job, config) if future is None else future.result()
                (skipped if result["status"] == "skipped" else successes).append(result)
                print(f"[{index}/{len(jobs)}] {result['status'].upper()} {job['member']} [{job['sensor']}]")
            except Exception as error:
                failure = {"job": job, "error_type": type(error).__name__, "message": str(error)}
                failures.append(failure)
                print(f"[{index}/{len(jobs)}] ERROR {job['member']} [{job['sensor']}]: {error}")
    finally:
        if args.workers != 1:
            executor.shutdown()
    write_json_atomic(output / f"batch_{batch['batch']:03d}_report.json",
                      {"batch": batch["batch"], "successes": successes, "skipped": skipped, "failures": failures})
    print(f"Batch {batch['batch']}: {len(successes)} success, {len(skipped)} skipped, {len(failures)} failed")
    return not failures


def main() -> int:
    args = parse_args()
    input_dir, output = args.input_dir.resolve(), args.output_dir.resolve()
    plan = ensure_plan(input_dir, output, args)
    print(f"Plan: {plan['total_jobs']} hardware sensors, {plan['total_batches']} batches, canonical BRB-r")
    if args.plan_only:
        for batch in plan["batches"]:
            print(f"Batch {batch['batch']}: {len(batch['jobs'])} jobs, {batch['weight'] / 2**20:.1f} MiB uncompressed")
        return 0
    selected = plan["batches"] if args.all_batches else [plan["batches"][args.batch - 1]]
    okay = True
    for batch in selected:
        okay = run_batch(batch, plan, input_dir, output, args) and okay
    return 0 if okay else 1


if __name__ == "__main__":
    raise SystemExit(main())
