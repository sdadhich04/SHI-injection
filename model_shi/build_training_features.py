#!/usr/bin/env python3
"""Build batched binary feature shards for the upstream RF/XGBoost models."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from common import (
    PIPELINE_VERSION, SCENARIOS, UPSTREAM_COMMIT, deterministic_sample,
    feature_dtype, labels_with_prediction_horizon, write_json_atomic,
    source_sensors,
)
from model_features import extract_feature_batch, get_pipeline_feature_names
from analyze_simple_shi import (
    VARIANTS, discover_jobs, iter_window_batches, make_plan, noisy_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=Path("noise_injection/outputs/post_noise_injection_1"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("model_shi/outputs/training_features"),
    )
    parser.add_argument("--batch", type=int)
    parser.add_argument("--total-batches", type=int, default=5)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--feature-batch-windows", type=int, default=128)
    parser.add_argument("--prediction-horizon", type=float, default=20.0)
    parser.add_argument(
        "--ar-burg", action=argparse.BooleanOptionalAction, default=True,
        help="include upstream AR-Burg features (use --no-ar-burg for the README 22/66-feature pipeline)",
    )
    parser.add_argument(
        "--max-windows-per-scenario", type=int, default=0,
        help="deterministic per-source/sensor/scenario cap; 0 reproduces upstream all-window behavior",
    )
    parser.add_argument("--run", action="append", help="only include this run name; repeatable")
    parser.add_argument("--sensor", action="append", help="only emit this sensor; repeatable")
    args = parser.parse_args()
    if args.total_batches < 1 or args.workers < 1 or args.stride < 1:
        parser.error("total-batches, workers, and stride must be positive")
    if args.window_size < 16 or args.feature_batch_windows < 1 or args.prediction_horizon < 0:
        parser.error("invalid window, feature-batch, or prediction-horizon value")
    if args.max_windows_per_scenario < 0:
        parser.error("max-windows-per-scenario cannot be negative")
    if not args.plan_only and args.batch is None:
        parser.error("--batch is required unless --plan-only is used")
    if args.batch is not None and not 1 <= args.batch <= args.total_batches:
        parser.error("--batch must be between 1 and --total-batches")
    return args


def ensure_training_plan(dataset: Path, output: Path, args: argparse.Namespace) -> dict:
    jobs = discover_jobs(dataset)
    if args.run:
        wanted = set(args.run)
        jobs = [job for job in jobs if Path(job["clean_relative"]).parts[0] in wanted]
    if args.sensor:
        wanted_sensors = set(args.sensor)
        jobs = [job for job in jobs if source_sensors(job["clean_relative"]) & wanted_sensors]
    if not jobs:
        raise ValueError("no synthetic-noise source jobs matched")
    plan = make_plan(dataset, jobs, args.total_batches)
    plan["model_pipeline_version"] = PIPELINE_VERSION
    plan["upstream_commit"] = UPSTREAM_COMMIT
    plan["training_settings"] = {
        "window_size": args.window_size, "stride": args.stride,
        "prediction_horizon": args.prediction_horizon,
        "max_windows_per_scenario": args.max_windows_per_scenario,
        "sensors": sorted(args.sensor) if args.sensor else None,
        "include_ar": args.ar_burg,
    }
    path = output / "training_plan.json"
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous != plan:
            raise ValueError(f"existing training plan differs: {path}; use a new output directory")
        return previous
    write_json_atomic(path, plan)
    return plan


def extract_multiaxis(windows: np.ndarray, fs: float, include_ar: bool = True) -> np.ndarray:
    return np.concatenate(
        [extract_feature_batch(windows[:, :, axis], fs, include_ar=include_ar)
         for axis in range(windows.shape[2])], axis=1,
    )


def collect_labels(clean_path: Path, observed: dict[str, Path], config: dict) -> dict:
    chunks: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: {"timestamps": [], **{variant: [] for variant in VARIANTS}}
    )
    for batch in iter_window_batches(
        clean_path, observed, config["window_size"], config["stride"],
        config["feature_batch_windows"],
    ):
        if config["sensors"] and batch.sensor not in config["sensors"]:
            continue
        state = chunks[batch.sensor]
        state["timestamps"].append(batch.timestamps)
        for variant in VARIANTS:
            active = np.any(
                batch.observed_end[variant][:, : batch.axes]
                != batch.clean_end[:, : batch.axes], axis=1,
            )
            state[variant].append(active)
    result = {}
    for sensor, state in chunks.items():
        timestamps = np.concatenate(state["timestamps"])
        labels = {"ground_truth": np.zeros(len(timestamps), dtype=np.uint8)}
        for variant in VARIANTS:
            active = np.concatenate(state[variant])
            labels[variant] = labels_with_prediction_horizon(
                timestamps, active, config["prediction_horizon"],
            )
        result[sensor] = {"timestamps": timestamps, "labels": labels}
    return result


def process_job(job: dict, config: dict) -> dict:
    started = time.monotonic()
    dataset, output = Path(config["dataset"]), Path(config["output"])
    relative = Path(job["clean_relative"])
    clean_path = dataset / "ground_truth" / relative
    observed = {variant: noisy_path(dataset, relative, variant) for variant in VARIANTS}
    destination = output / relative.parent / relative.stem
    marker = destination / "job_report.json"
    if marker.is_file():
        old = json.loads(marker.read_text(encoding="utf-8"))
        if old.get("status") == "success" and old.get("processing_config") == config["processing_config"]:
            if all(Path(item["path"]).is_file() and Path(item["path"]).stat().st_size == item["bytes"]
                   for item in old.get("outputs", [])):
                return {"status": "skipped", "job": job, "report": old}
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"incomplete existing training output: {destination}")

    label_data = collect_labels(clean_path, observed, config)
    if not label_data:
        raise ValueError(f"no selected sensor windows in {relative}")
    selected = {
        (sensor, scenario): deterministic_sample(labels, config["max_windows_per_scenario"])
        for sensor, data in label_data.items() for scenario, labels in data["labels"].items()
    }
    offsets = defaultdict(int)
    handles, paths, summaries = {}, {}, defaultdict(lambda: defaultdict(int))
    destination.mkdir(parents=True, exist_ok=True)
    try:
        for batch in iter_window_batches(
            clean_path, observed, config["window_size"], config["stride"],
            config["feature_batch_windows"],
        ):
            if batch.sensor not in label_data:
                continue
            start = offsets[batch.sensor]
            global_indices = np.arange(start, start + len(batch.timestamps))
            offsets[batch.sensor] += len(batch.timestamps)
            scenario_windows = {"ground_truth": batch.clean_windows, **batch.observed_windows}
            feature_count = len(get_pipeline_feature_names(
                batch.axes == 3, include_ar=config["include_ar"],
            ))
            dtype = feature_dtype(feature_count)
            key = batch.sensor
            if key not in handles:
                final = destination / f"{batch.sensor}.bin"
                temporary = final.with_suffix(".bin.tmp")
                if final.exists() or temporary.exists():
                    raise FileExistsError(f"refusing to overwrite {final} or {temporary}")
                handles[key] = temporary.open("xb")
                paths[key] = temporary, final, dtype, batch.axes
            for scenario, windows in scenario_windows.items():
                chosen = selected[(batch.sensor, scenario)]
                left = np.searchsorted(chosen, start, side="left")
                right = np.searchsorted(chosen, start + len(batch.timestamps), side="left")
                local_indices = chosen[left:right] - start
                mask = np.zeros(len(batch.timestamps), dtype=bool)
                mask[local_indices] = True
                if not np.any(mask):
                    continue
                features = extract_multiaxis(
                    windows[mask], batch.fs, include_ar=config["include_ar"],
                )
                labels = label_data[batch.sensor]["labels"][scenario][global_indices[mask]]
                records = np.zeros(len(features), dtype=dtype)
                records["timestamp_ms"] = batch.timestamps[mask]
                records["label"] = labels
                records["scenario"] = SCENARIOS[scenario]
                records["features"] = features
                handles[key].write(records.tobytes())
                summaries[key][f"scenario_{scenario}"] += len(records)
                for label in (0, 1, 2):
                    summaries[key][f"label_{label}"] += int(np.sum(labels == label))
        for handle in handles.values():
            handle.flush(); os.fsync(handle.fileno()); handle.close()
        handles.clear()
        for temporary, final, _dtype, _axes in paths.values():
            os.replace(temporary, final)
    except Exception:
        for handle in handles.values():
            handle.close()
        for temporary, _final, _dtype, _axes in paths.values():
            temporary.unlink(missing_ok=True)
        raise

    outputs = []
    for sensor, (_temporary, final, dtype, axes) in sorted(paths.items()):
        outputs.append({
            "sensor": sensor, "axes": axes,
            "features": (dtype.itemsize - 12) // 4,
            "record_bytes": dtype.itemsize, "path": str(final.resolve()),
            "bytes": final.stat().st_size, "rows": final.stat().st_size // dtype.itemsize,
            "counts": dict(summaries[sensor]),
        })
    report = {
        "status": "success", "job": job, "processing_config": config["processing_config"],
        "elapsed_seconds": time.monotonic() - started, "outputs": outputs,
    }
    write_json_atomic(marker, report)
    return {"status": "success", "job": job, "report": report}


def next_report(output: Path, batch: int, total: int) -> Path:
    base = output / f"feature_report_batch_{batch:02d}_of_{total:02d}.json"
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
    dataset, output = args.dataset_dir.resolve(), args.output_dir.resolve()
    if dataset == output or dataset in output.parents:
        raise ValueError("training output must be outside the input dataset")
    plan = ensure_training_plan(dataset, output, args)
    ar_state = "enabled" if plan["training_settings"]["include_ar"] else "disabled"
    print(f"Plan: {plan['total_jobs']} source files, {plan['total_batches']} batches; "
          f"AR-Burg {ar_state}; cap={args.max_windows_per_scenario or 'all'}")
    if args.plan_only:
        return 0
    processing_config = plan["training_settings"] | {
        "feature_batch_windows": args.feature_batch_windows,
    }
    config = {
        "dataset": str(dataset), "output": str(output),
        **processing_config, "processing_config": processing_config,
    }
    selected_jobs = plan["batches"][args.batch - 1]["jobs"]
    successful, skipped, failed = [], [], []
    started = time.monotonic()

    def accept(job, result=None, error=None):
        if error is not None:
            failed.append({"job": job, "error_type": type(error).__name__, "message": str(error)})
            print(f"[failed] {job['clean_relative']}: {type(error).__name__}: {error}", flush=True)
        else:
            (skipped if result["status"] == "skipped" else successful).append(result)
            print(f"[{result['status']}] {job['clean_relative']}", flush=True)

    if args.workers == 1:
        for job in selected_jobs:
            try: accept(job, result=process_job(job, config))
            except Exception as exc: accept(job, error=exc)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_job, job, config): job for job in selected_jobs}
            for future in as_completed(futures):
                job = futures[future]
                try: accept(job, result=future.result())
                except Exception as exc: accept(job, error=exc)
    report = {
        "pipeline_version": PIPELINE_VERSION, "upstream_commit": UPSTREAM_COMMIT,
        "batch": args.batch, "total_batches": args.total_batches,
        "elapsed_seconds": time.monotonic() - started,
        "successful_jobs": successful, "skipped_jobs": skipped, "failed_jobs": failed,
    }
    report_path = next_report(output, args.batch, args.total_batches)
    write_json_atomic(report_path, report)
    print(f"Batch {args.batch}: {len(successful)} succeeded, {len(skipped)} skipped, "
          f"{len(failed)} failed. Report: {report_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
