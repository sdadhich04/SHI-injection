#!/usr/bin/env python3
"""Run canonical BRB-r SHI on clean and software-injected Project SHIELD data."""

from __future__ import annotations

import argparse
from collections import defaultdict
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

from analyze_simple_shi import (  # noqa: E402
    VARIANTS, dataset_fingerprint, discover_jobs, iter_window_batches, noisy_path,
)
from canonical_brb.core import (  # noqa: E402
    BinaryEventModel, fit_binary_event, fit_brb, score_binary_event, score_brb,
)
from canonical_brb.features import UPSTREAM_COMMIT, extract_batch  # noqa: E402
from canonical_brb.runtime import (  # noqa: E402
    PIPELINE_VERSION, Reservoir, completed_report, distribute_jobs, stable_seed,
    quarantine_incomplete, write_json_atomic,
)


SOFTWARE_DTYPE = np.dtype([
    ("timestamp_ms", "<u8"), ("shi", "<f4"), ("raw_distance", "<f4"),
    ("degradation_score", "<f4"), ("injected_fraction", "<f4"),
    ("ground_truth", "<f4", (3,)), ("observed", "<f4", (3,)),
], align=False)
SCENARIOS = ("ground_truth", *VARIANTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=WORKSPACE_ROOT / "noise_injection/outputs/post_noise_injection_1")
    parser.add_argument("--output-dir", type=Path, default=PACKAGE_ROOT / "outputs/canonical_brb/software")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--batch", type=int, help="one-based batch to process")
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
    parser.add_argument("--no-adev", action="store_true", help="disable canonical Allan-deviation features for faster exploratory runs")
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


def make_plan(dataset: Path, args: argparse.Namespace) -> dict:
    jobs = discover_jobs(dataset)
    settings = {
        "window_size": args.window_size, "stride": args.stride,
        "feature_batch_windows": args.feature_batch_windows,
        "max_calibration_windows": args.max_calibration_windows,
        "healthy_quantile": args.healthy_quantile, "wavelet_level": args.wavelet_level,
        "include_adev": not args.no_adev, "variants": list(VARIANTS),
    }
    return {
        "pipeline": "canonical_brb_software", "pipeline_version": PIPELINE_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "dataset_fingerprint": dataset_fingerprint(dataset, jobs),
        "total_batches": args.total_batches, "total_jobs": len(jobs), "settings": settings,
        "batches": distribute_jobs(jobs, args.total_batches, "records"),
    }


def ensure_plan(dataset: Path, output: Path, args: argparse.Namespace) -> dict:
    expected = make_plan(dataset, args)
    path = output / "batch_plan.json"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != expected:
            raise ValueError(f"existing plan differs from inputs/settings: {path}; use a new output directory")
        return current
    write_json_atomic(path, expected)
    return expected


def _input_signature(paths: list[Path]) -> list[list[int | str]]:
    return [[str(path), path.stat().st_size, path.stat().st_mtime_ns] for path in paths]


def process_job(job: dict, config: dict) -> dict:
    started = time.monotonic()
    dataset = Path(config["dataset"])
    output_root = Path(config["output"])
    relative = Path(job["clean_relative"])
    clean = dataset / "ground_truth" / relative
    observed = {variant: noisy_path(dataset, relative, variant) for variant in VARIANTS}
    inputs = [clean, *observed.values()]
    signature = _input_signature(inputs)
    destination = output_root / relative.parent / relative.stem
    marker = destination / "job_report.json"
    processing_config = {key: value for key, value in config.items() if key not in {"dataset", "output"}}
    previous = completed_report(marker, processing_config)
    if previous:
        return {"status": "skipped", "job": job, "report": previous}
    recovered = quarantine_incomplete(destination, output_root)
    destination.mkdir(parents=True, exist_ok=True)

    reservoirs: dict[str, Reservoir] = {}
    raw_reservoirs: dict[str, Reservoir] = {}
    for batch in iter_window_batches(clean, {}, config["window_size"], config["stride"], config["feature_batch_windows"]):
        names, features = extract_batch(
            batch.clean_windows, batch.fs, wavelet_level=config["wavelet_level"],
            include_adev=config["include_adev"],
        )
        reservoir = reservoirs.setdefault(
            batch.sensor,
            Reservoir(config["max_calibration_windows"], stable_seed(job["job_id"], batch.sensor)),
        )
        reservoir.update(names, features)
        if batch.sensor.endswith("vibration"):
            raw_reservoir = raw_reservoirs.setdefault(
                batch.sensor,
                Reservoir(config["max_calibration_windows"], stable_seed(job["job_id"], batch.sensor, "raw")),
            )
            raw_reservoir.update(
                [f"sample_{index}" for index in range(batch.clean_windows.shape[1])],
                batch.clean_windows[:, :, 0],
            )
    models = {}
    for sensor, reservoir in reservoirs.items():
        try:
            models[sensor] = fit_brb(
                reservoir.names or [], reservoir.matrix(), healthy_quantile=config["healthy_quantile"],
                upstream_commit=UPSTREAM_COMMIT,
            )
        except ValueError as error:
            raw_reservoir = raw_reservoirs.get(sensor)
            if "no finite varying features" not in str(error) or raw_reservoir is None:
                raise
            models[sensor] = fit_binary_event(
                raw_reservoir.matrix(), healthy_quantile=config["healthy_quantile"],
                upstream_commit=UPSTREAM_COMMIT,
            )
    model_paths = {}
    for sensor, model in models.items():
        final_model = destination / f"{sensor}__model.json"
        pending_model = destination / f"{sensor}__model.json.pending"
        model.to_json(pending_model)
        model_paths[sensor] = pending_model, final_model

    handles = {}
    paths = {}
    counts = defaultdict(int)
    minima = defaultdict(lambda: 1.0)
    try:
        for batch in iter_window_batches(
            clean, observed, config["window_size"], config["stride"], config["feature_batch_windows"],
        ):
            model = models[batch.sensor]
            if isinstance(model, BinaryEventModel):
                scenario_data = {
                    "ground_truth": (batch.clean_windows, batch.clean_end, np.zeros(len(batch.timestamps))),
                    **{
                        variant: (batch.observed_windows[variant], batch.observed_end[variant],
                                  batch.injected_fraction[variant])
                        for variant in VARIANTS
                    },
                }
                clean_names = []
            else:
                clean_names, clean_features = extract_batch(
                    batch.clean_windows, batch.fs, wavelet_level=config["wavelet_level"],
                    include_adev=config["include_adev"],
                )
                scenario_data = {
                    "ground_truth": (clean_features, batch.clean_end, np.zeros(len(batch.timestamps))),
                }
                for variant in VARIANTS:
                    names, features = extract_batch(
                        batch.observed_windows[variant], batch.fs,
                        wavelet_level=config["wavelet_level"], include_adev=config["include_adev"],
                    )
                    if names != clean_names:
                        raise ValueError("clean and injected feature schemas differ")
                    scenario_data[variant] = (features, batch.observed_end[variant], batch.injected_fraction[variant])
            for scenario, (features_or_windows, signal, fraction) in scenario_data.items():
                if isinstance(model, BinaryEventModel):
                    health, raw, degradation = score_binary_event(model, features_or_windows)
                else:
                    health, raw, degradation = score_brb(model, clean_names, features_or_windows)
                key = batch.sensor, scenario
                if key not in handles:
                    final = destination / f"{batch.sensor}__{scenario}.bin"
                    temporary = final.with_name(final.name + ".tmp")
                    if final.exists() or temporary.exists():
                        raise FileExistsError(f"refusing to overwrite {final}")
                    handles[key] = temporary.open("wb")
                    paths[key] = temporary, final
                records = np.zeros(len(health), dtype=SOFTWARE_DTYPE)
                records["timestamp_ms"] = batch.timestamps
                records["shi"] = health
                records["raw_distance"] = raw
                records["degradation_score"] = degradation
                records["injected_fraction"] = fraction
                records["ground_truth"][:, :batch.axes] = batch.clean_end
                records["observed"][:, :batch.axes] = signal
                handles[key].write(records.tobytes())
                counts[key] += len(records)
                minima[key] = min(minima[key], float(np.min(health)))
        for handle in handles.values():
            handle.flush(); os.fsync(handle.fileno()); handle.close()
        handles.clear()
        for temporary, final in paths.values():
            os.replace(temporary, final)
        for pending_model, final_model in model_paths.values():
            os.replace(pending_model, final_model)
    except Exception:
        for handle in handles.values():
            handle.close()
        for temporary, _final in paths.values():
            temporary.unlink(missing_ok=True)
        for pending_model, _final_model in model_paths.values():
            pending_model.unlink(missing_ok=True)
        raise
    if signature != _input_signature(inputs):
        raise RuntimeError("input changed during processing")
    outputs = [
        {"sensor": key[0], "scenario": key[1], "path": str(final),
         "bytes": final.stat().st_size, "records": counts[key], "minimum_shi": minima[key],
         "record_bytes": SOFTWARE_DTYPE.itemsize}
        for key, (_temporary, final) in sorted(paths.items())
    ]
    outputs.extend(
        {"sensor": sensor, "artifact": "model", "path": str(final_model),
         "bytes": final_model.stat().st_size}
        for sensor, (_pending, final_model) in sorted(model_paths.items())
    )
    report = {
        "status": "success", "job": job, "processing_config": processing_config,
        "elapsed_seconds": time.monotonic() - started,
        "calibration": {sensor: {"windows_seen": reservoir.seen, "windows_used": len(reservoir.rows)}
                        for sensor, reservoir in reservoirs.items()},
        "model_types": {sensor: model.model_type for sensor, model in models.items()},
        "recovered_interrupted_output": str(recovered) if recovered else None,
        "input_signature_unchanged": True, "outputs": outputs,
    }
    write_json_atomic(marker, report)
    return {"status": "success", "job": job, "report": report}


def run_batch(batch: dict, plan: dict, dataset: Path, output: Path, args: argparse.Namespace) -> bool:
    settings = dict(plan["settings"])
    config = {"dataset": str(dataset), "output": str(output), **settings}
    successes, skipped, failures = [], [], []
    jobs = batch["jobs"]
    print(f"Software batch {batch['batch']}/{plan['total_batches']}: {len(jobs)} jobs, workers={args.workers}")
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
                print(f"[{index}/{len(jobs)}] {result['status'].upper()} {job['clean_relative']}")
            except Exception as error:
                failure = {"job": job, "error_type": type(error).__name__, "message": str(error)}
                failures.append(failure)
                print(f"[{index}/{len(jobs)}] ERROR {job['clean_relative']}: {error}")
    finally:
        if args.workers != 1:
            executor.shutdown()
    report = {"batch": batch["batch"], "successes": successes, "skipped": skipped, "failures": failures}
    write_json_atomic(output / f"batch_{batch['batch']:03d}_report.json", report)
    print(f"Batch {batch['batch']}: {len(successes)} success, {len(skipped)} skipped, {len(failures)} failed")
    return not failures


def main() -> int:
    args = parse_args()
    dataset, output = args.dataset_dir.resolve(), args.output_dir.resolve()
    if dataset == output or dataset in output.parents:
        raise ValueError("output must be outside the input dataset")
    plan = ensure_plan(dataset, output, args)
    print(f"Plan: {plan['total_jobs']} software jobs, {plan['total_batches']} batches, canonical BRB-r")
    if args.plan_only:
        for batch in plan["batches"]:
            print(f"Batch {batch['batch']}: {len(batch['jobs'])} jobs, {batch['weight']:,} source records")
        return 0
    selected = plan["batches"] if args.all_batches else [plan["batches"][args.batch - 1]]
    okay = True
    for batch in selected:
        okay = run_batch(batch, plan, dataset, output, args) and okay
    return 0 if okay else 1


if __name__ == "__main__":
    raise SystemExit(main())
