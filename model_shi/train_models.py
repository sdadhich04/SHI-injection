#!/usr/bin/env python3
"""Train upstream RandomForest/XGBoost models from binary feature shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from common import PIPELINE_VERSION, UPSTREAM_COMMIT, deterministic_sample, feature_dtype, write_json_atomic
from model_features import get_pipeline_feature_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--features-dir", type=Path,
        default=Path("model_shi/outputs/training_features"),
    )
    parser.add_argument("--models-dir", type=Path, default=Path("model_shi/models"))
    parser.add_argument("--sensor", action="append", help="train only this sensor; repeatable")
    parser.add_argument(
        "--exclude-sensor", action="append", default=[],
        help="exclude this non-deployable sensor identifier; repeatable",
    )
    parser.add_argument("--repeats", type=int, default=3, help="validation splits, starting at seed 42")
    parser.add_argument(
        "--max-rows-per-sensor", type=int, default=0,
        help="deterministic training cap; 0 matches upstream all-row behavior",
    )
    parser.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 2)))
    args = parser.parse_args()
    if args.repeats < 1 or args.max_rows_per_sensor < 0 or args.workers < 1:
        parser.error("repeats/workers must be positive and max-rows cannot be negative")
    return args


def discover_outputs(features_dir: Path, selected: set[str] | None) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for marker in sorted(features_dir.rglob("job_report.json")):
        report = json.loads(marker.read_text(encoding="utf-8"))
        if report.get("status") != "success":
            continue
        for output in report.get("outputs", []):
            if selected and output["sensor"] not in selected:
                continue
            path = Path(output["path"])
            if not path.is_file() or path.stat().st_size != output["bytes"]:
                raise ValueError(f"missing or changed feature shard: {path}")
            grouped[output["sensor"]].append(output)
    if not grouped:
        raise ValueError("no completed binary feature shards matched")
    return grouped


def describe_sensor(outputs: list[dict], limit: int) -> dict:
    feature_counts = {int(item["features"]) for item in outputs}
    axes_values = {int(item["axes"]) for item in outputs}
    if len(feature_counts) != 1 or len(axes_values) != 1:
        raise ValueError("inconsistent feature shard widths")
    feature_count, axes = feature_counts.pop(), axes_values.pop()
    expected_without_ar = 22 * axes
    expected_with_ar = 26 * axes
    if feature_count not in {expected_without_ar, expected_with_ar}:
        raise ValueError(
            f"unexpected feature width {feature_count} for {axes} axis/axes; "
            f"expected {expected_without_ar} without AR or {expected_with_ar} with AR"
        )
    total = sum(int(item["rows"]) for item in outputs)
    return {
        "feature_count": feature_count, "axes": axes, "available_rows": total,
        "rows_used": min(total, limit) if limit else total,
        "include_ar": feature_count == expected_with_ar,
    }


def load_sensor(outputs: list[dict], limit: int) -> tuple[np.ndarray, np.ndarray, dict]:
    shape = describe_sensor(outputs, limit)
    feature_count = shape["feature_count"]
    total = shape["available_rows"]
    allocations = [int(item["rows"]) for item in outputs]
    if limit and total > limit:
        raw = np.asarray(allocations, dtype=np.float64) * limit / total
        allocations = np.floor(raw).astype(int).tolist()
        while sum(allocations) < limit:
            candidates = [i for i, item in enumerate(outputs) if allocations[i] < item["rows"]]
            target = max(candidates, key=lambda i: raw[i] - allocations[i])
            allocations[target] += 1
    matrices, labels = [], []
    dtype = feature_dtype(feature_count)
    for output, amount in zip(outputs, allocations):
        if amount == 0:
            continue
        records = np.memmap(output["path"], dtype=dtype, mode="r")
        indices = deterministic_sample(records["label"], amount)
        matrices.append(np.asarray(records["features"][indices], dtype=np.float32))
        labels.append((np.asarray(records["label"][indices]) > 0).astype(np.uint8))
    X = np.concatenate(matrices)
    y = np.concatenate(labels)
    return X, y, shape


def dump_atomic(value, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    joblib.dump(value, temporary)
    os.replace(temporary, path)


def train_sensor(sensor: str, outputs: list[dict], args: argparse.Namespace) -> dict:
    started = time.monotonic()
    shape = describe_sensor(outputs, args.max_rows_per_sensor)
    destination = args.models_dir.resolve() / sensor
    config = {
        "pipeline_version": PIPELINE_VERSION, "upstream_commit": UPSTREAM_COMMIT,
        "sensor": sensor, "rows_used": shape["rows_used"],
        "available_rows": shape["available_rows"],
        "max_rows_per_sensor": args.max_rows_per_sensor, "repeats": args.repeats,
        "feature_count": shape["feature_count"], "axes": shape["axes"],
        "features_include_ar_burg": shape["include_ar"],
    }
    metadata_path = destination / "model_metadata.json"
    artifacts = [destination / name for name in ("scaler.joblib", "rf.joblib", "xgb.joblib")]
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text(encoding="utf-8"))
        if old.get("training_config") != config:
            raise ValueError(f"existing model configuration differs for {sensor}: {destination}")
        if not all(path.is_file() and path.stat().st_size > 0 for path in artifacts):
            raise FileNotFoundError(f"model metadata exists but artifacts are incomplete: {destination}")
        return {"status": "skipped", "sensor": sensor, "metadata": old}
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"incomplete existing model output: {destination}")

    X_raw, y, loaded_shape = load_sensor(outputs, args.max_rows_per_sensor)
    if loaded_shape != shape or len(y) != shape["rows_used"]:
        raise ValueError(f"loaded training shape changed unexpectedly for {sensor}")
    unique, counts = np.unique(y, return_counts=True)
    if set(unique) != {0, 1} or min(counts) < 2:
        raise ValueError(f"{sensor} needs at least two rows in healthy and fault classes")

    # This intentionally matches upstream: StandardScaler is fit before the
    # random split. The metadata flags the resulting evaluation leakage.
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    del X_raw
    validation = []
    saved_rf = saved_xgb = None
    for repeat in range(args.repeats):
        seed = 42 + repeat
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y,
        )
        rf = RandomForestClassifier(n_jobs=args.workers, random_state=seed)
        xgb = XGBClassifier(n_jobs=args.workers, random_state=seed, eval_metric="logloss")
        rf.fit(X_train, y_train)
        xgb.fit(X_train, y_train)
        repeat_result = {"seed": seed, "models": {}}
        for name, model in (("random_forest", rf), ("xgboost", xgb)):
            prediction = model.predict(X_test)
            repeat_result["models"][name] = {
                "confusion_matrix": confusion_matrix(y_test, prediction, labels=[0, 1]).tolist(),
                "classification_report": classification_report(
                    y_test, prediction, labels=[0, 1], target_names=["healthy", "fault"],
                    output_dict=True, zero_division=0,
                ),
            }
        validation.append(repeat_result)
        if repeat == 0:
            saved_rf, saved_xgb = rf, xgb
    assert saved_rf is not None and saved_xgb is not None

    destination.mkdir(parents=True, exist_ok=True)
    dump_atomic(scaler, destination / "scaler.joblib")
    dump_atomic(saved_rf, destination / "rf.joblib")
    dump_atomic(saved_xgb, destination / "xgb.joblib")
    feature_names = get_pipeline_feature_names(
        shape["axes"] == 3, include_ar=shape["include_ar"],
    )
    metadata = {
        "status": "success", "training_config": config,
        "class_counts": {"healthy": int(np.sum(y == 0)), "fault": int(np.sum(y == 1))},
        "feature_names": feature_names, "validation": validation,
        "evaluation_warning": (
            "Upstream-compatible stratified random window split and pre-split scaling can leak "
            "temporally adjacent information; use hardware results as external validation."
        ),
        "upstream_compatibility": {
            "features": (
                "Matches the checked-in upstream feature implementations. AR-Burg is "
                f"{'enabled to match features.py' if shape['include_ar'] else 'disabled to match the README 22/66-feature description'}. "
                "The upstream README and features.py disagree about its default."
            ),
            "scaling": (
                "Uses StandardScaler as implemented by upstream pipeline.py and documented by "
                "the README inference example. The upstream notebook later reloads unscaled "
                "chunks before fitting, which is inconsistent with both."
            ),
            "classifiers": (
                "RandomForestClassifier and XGBClassifier use the upstream defaults, worker "
                "count aside, with seeds starting at the upstream random_state=42."
            ),
        },
        "shi_definition": "P(healthy), independently reported for RandomForest and XGBoost",
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json_atomic(metadata_path, metadata)
    return {"status": "success", "sensor": sensor, "metadata": metadata}


def next_training_report(models_dir: Path) -> Path:
    base = models_dir / "training_report.json"
    if not base.exists():
        return base
    attempt = 2
    while True:
        candidate = models_dir / f"training_report_attempt_{attempt:02d}.json"
        if not candidate.exists():
            return candidate
        attempt += 1


def main() -> int:
    args = parse_args()
    args.features_dir = args.features_dir.resolve()
    args.models_dir = args.models_dir.resolve()
    grouped = discover_outputs(args.features_dir, set(args.sensor) if args.sensor else None)
    for sensor in args.exclude_sensor:
        grouped.pop(sensor, None)
    if not grouped:
        raise ValueError("no training sensors remain after exclusions")
    results, failures = [], []
    for sensor, outputs in sorted(grouped.items()):
        try:
            result = train_sensor(sensor, outputs, args)
            results.append(result)
            print(f"[{result['status']}] {sensor}", flush=True)
        except Exception as exc:
            failures.append({"sensor": sensor, "error_type": type(exc).__name__, "message": str(exc)})
            print(f"[failed] {sensor}: {type(exc).__name__}: {exc}", flush=True)
    report = {
        "pipeline_version": PIPELINE_VERSION, "upstream_commit": UPSTREAM_COMMIT,
        "models": results, "failures": failures,
    }
    report_path = next_training_report(args.models_dir)
    write_json_atomic(report_path, report)
    print(f"Training: {len(results)} model sets, {len(failures)} failures. Report: {report_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
