#!/usr/bin/env python3
"""Run upstream-model P(healthy) SHI on clean and injected software data."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np

from common import (
    PIPELINE_VERSION, PREDICTION_DTYPE, SCENARIOS, UPSTREAM_COMMIT,
    healthy_probability, model_ar_setting, write_json_atomic,
    source_sensors,
)
from build_training_features import collect_labels, extract_multiaxis
from analyze_simple_shi import VARIANTS, discover_jobs, iter_window_batches, make_plan, noisy_path
from quick_visualizer import first_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=Path("noise_injection/outputs/post_noise_injection_1"),
    )
    parser.add_argument("--models-dir", type=Path, default=Path("model_shi/models"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("model_shi/outputs/software_predictions"),
    )
    parser.add_argument("--batch", type=int)
    parser.add_argument("--total-batches", type=int, default=5)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--feature-batch-windows", type=int, default=128)
    parser.add_argument("--prediction-horizon", type=float, default=20.0)
    parser.add_argument("--run", action="append")
    parser.add_argument("--sensor", action="append")
    args = parser.parse_args()
    if args.total_batches < 1 or args.workers < 1 or args.stride < 1:
        parser.error("total-batches, workers, and stride must be positive")
    if args.window_size < 16 or args.feature_batch_windows < 1 or args.prediction_horizon < 0:
        parser.error("invalid window, feature-batch, or prediction-horizon")
    if not args.plan_only and args.batch is None:
        parser.error("--batch is required unless --plan-only is used")
    if args.batch is not None and not 1 <= args.batch <= args.total_batches:
        parser.error("--batch must be between 1 and --total-batches")
    return args


def available_sensors(models_dir: Path) -> set[str]:
    return {path.parent.name for path in models_dir.glob("*/model_metadata.json")}


def ensure_plan(dataset: Path, models: Path, output: Path, args) -> dict:
    jobs = discover_jobs(dataset)
    if args.run:
        jobs = [j for j in jobs if Path(j["clean_relative"]).parts[0] in set(args.run)]
    sensors = available_sensors(models)
    if args.sensor:
        sensors &= set(args.sensor)
    jobs = [job for job in jobs if source_sensors(job["clean_relative"]) & sensors]
    if not jobs or not sensors:
        raise ValueError("no source jobs or trained sensor models matched")
    include_ar = model_ar_setting(models, sensors)
    plan = make_plan(dataset, jobs, args.total_batches)
    plan.update({
        "model_pipeline_version": PIPELINE_VERSION, "upstream_commit": UPSTREAM_COMMIT,
        "prediction_settings": {
            "models_dir": str(models), "sensors": sorted(sensors),
            "window_size": args.window_size, "stride": args.stride,
            "prediction_horizon": args.prediction_horizon, "include_ar": include_ar,
        },
    })
    path = output / "software_prediction_plan.json"
    if path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))
        if old != plan:
            raise ValueError(f"existing prediction plan differs: {path}; use a new output directory")
        return old
    write_json_atomic(path, plan)
    return plan


@lru_cache(maxsize=64)
def load_models(models_dir: str, sensor: str):
    root = Path(models_dir) / sensor
    metadata = json.loads((root / "model_metadata.json").read_text(encoding="utf-8"))
    return (
        joblib.load(root / "scaler.joblib"), joblib.load(root / "rf.joblib"),
        joblib.load(root / "xgb.joblib"), metadata,
    )


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
        raise FileExistsError(f"incomplete existing software prediction output: {destination}")

    label_config = {
        "window_size": config["window_size"], "stride": config["stride"],
        "feature_batch_windows": config["feature_batch_windows"],
        "prediction_horizon": config["prediction_horizon"], "sensors": config["sensors"],
    }
    label_data = collect_labels(clean_path, observed, label_config)
    if not label_data:
        raise ValueError(f"no modeled sensor windows in {relative}")
    offsets = defaultdict(int)
    handles, paths = {}, {}
    summaries = defaultdict(lambda: {"rows": 0, "rf_sum": 0.0, "xgb_sum": 0.0, "correct_rf": 0, "correct_xgb": 0})
    destination.mkdir(parents=True, exist_ok=True)
    try:
        for batch in iter_window_batches(
            clean_path, observed, config["window_size"], config["stride"],
            config["feature_batch_windows"],
        ):
            if batch.sensor not in label_data:
                continue
            scaler, rf, xgb, metadata = load_models(config["models_dir"], batch.sensor)
            start = offsets[batch.sensor]
            indices = np.arange(start, start + len(batch.timestamps))
            offsets[batch.sensor] += len(batch.timestamps)
            scenarios = {"ground_truth": (batch.clean_windows, batch.clean_end), **{
                variant: (batch.observed_windows[variant], batch.observed_end[variant])
                for variant in VARIANTS
            }}
            for scenario, (windows, ends) in scenarios.items():
                features = extract_multiaxis(
                    windows, batch.fs, include_ar=config["include_ar"],
                )
                if features.shape[1] != metadata["training_config"]["feature_count"]:
                    raise ValueError(f"feature width mismatch for {batch.sensor}")
                scaled = scaler.transform(features)
                p_rf = healthy_probability(rf, scaled)
                p_xgb = healthy_probability(xgb, scaled)
                labels = label_data[batch.sensor]["labels"][scenario][indices]
                key = batch.sensor, scenario
                if key not in handles:
                    final = destination / f"{batch.sensor}__{scenario}.bin"
                    temporary = final.with_suffix(".bin.tmp")
                    if final.exists() or temporary.exists():
                        raise FileExistsError(f"refusing to overwrite {final} or {temporary}")
                    handles[key] = temporary.open("xb")
                    paths[key] = temporary, final, batch.axes
                records = np.zeros(len(features), dtype=PREDICTION_DTYPE)
                records["timestamp_ms"] = batch.timestamps
                records["shi_rf"] = p_rf
                records["shi_xgb"] = p_xgb
                records["ground_truth_label"] = labels
                records["scenario"] = SCENARIOS[scenario]
                records["signal"][:, : batch.axes] = ends[:, : batch.axes]
                handles[key].write(records.tobytes())
                summary = summaries[key]
                summary["rows"] += len(records)
                summary["rf_sum"] += float(np.sum(p_rf)); summary["xgb_sum"] += float(np.sum(p_xgb))
                truth = labels == 0
                summary["correct_rf"] += int(np.sum((p_rf >= 0.5) == truth))
                summary["correct_xgb"] += int(np.sum((p_xgb >= 0.5) == truth))
        for handle in handles.values():
            handle.flush(); os.fsync(handle.fileno()); handle.close()
        handles.clear()
        for temporary, final, _axes in paths.values(): os.replace(temporary, final)
    except Exception:
        for handle in handles.values(): handle.close()
        for temporary, _final, _axes in paths.values(): temporary.unlink(missing_ok=True)
        raise
    outputs = []
    for key, (_temporary, final, axes) in sorted(paths.items()):
        sensor, scenario = key; summary = summaries[key]; rows = summary["rows"]
        outputs.append({
            "sensor": sensor, "scenario": scenario, "axes": axes,
            "path": str(final.resolve()), "bytes": final.stat().st_size,
            "rows": rows, "record_bytes": PREDICTION_DTYPE.itemsize,
            "shi_rf_mean": summary["rf_sum"] / rows,
            "shi_xgb_mean": summary["xgb_sum"] / rows,
            "rf_accuracy_against_known_labels": summary["correct_rf"] / rows,
            "xgb_accuracy_against_known_labels": summary["correct_xgb"] / rows,
        })
    report = {
        "status": "success", "job": job, "processing_config": config["processing_config"],
        "shi_definition": "P(healthy), range 0..1", "outputs": outputs,
        "first_timestamp_ms": first_timestamp(clean_path),
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json_atomic(marker, report)
    return {"status": "success", "job": job, "report": report}


def report_path(output: Path, batch: int, total: int) -> Path:
    base = output / f"software_report_batch_{batch:02d}_of_{total:02d}.json"
    if not base.exists(): return base
    attempt = 2
    while True:
        path = base.with_name(f"{base.stem}_attempt_{attempt:02d}.json")
        if not path.exists(): return path
        attempt += 1


def main() -> int:
    args = parse_args()
    dataset, models, output = args.dataset_dir.resolve(), args.models_dir.resolve(), args.output_dir.resolve()
    plan = ensure_plan(dataset, models, output, args)
    print(f"Plan: {plan['total_jobs']} sources, {plan['total_batches']} batches, "
          f"models={','.join(plan['prediction_settings']['sensors'])}")
    if args.plan_only: return 0
    settings = plan["prediction_settings"]
    processing_config = {
        "models_dir": str(models), "sensors": settings["sensors"],
        "window_size": args.window_size, "stride": args.stride,
        "feature_batch_windows": args.feature_batch_windows,
        "prediction_horizon": args.prediction_horizon,
        "include_ar": settings["include_ar"],
    }
    config = {"dataset": str(dataset), "output": str(output), **processing_config,
              "processing_config": processing_config}
    jobs = plan["batches"][args.batch - 1]["jobs"]
    success, skipped, failed = [], [], []
    started = time.monotonic()
    def accept(job, result=None, error=None):
        if error:
            failed.append({"job": job, "error_type": type(error).__name__, "message": str(error)})
            print(f"[failed] {job['clean_relative']}: {type(error).__name__}: {error}", flush=True)
        else:
            (skipped if result["status"] == "skipped" else success).append(result)
            print(f"[{result['status']}] {job['clean_relative']}", flush=True)
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
    path = report_path(output, args.batch, args.total_batches); write_json_atomic(path, report)
    print(f"Batch {args.batch}: {len(success)} succeeded, {len(skipped)} skipped, {len(failed)} failed. Report: {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
