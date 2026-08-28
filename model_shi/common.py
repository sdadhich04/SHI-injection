"""Shared definitions for upstream-model SHI training and inference."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
NOISE_CODE = Path(
    os.environ.get("NOISE_INJECTION_CODE", ROOT.parent / "noise_injection_shield")
).resolve()
for directory in (ROOT, NOISE_CODE, Path(__file__).resolve().parent):
    value = str(directory)
    if value not in sys.path:
        sys.path.insert(0, value)

UPSTREAM_COMMIT = "c094797c96923449b6075d8c483df37856b26712"
PIPELINE_VERSION = 1
SCENARIOS = {"ground_truth": 0, "random_0_25": 1, "random_0_5": 2}
TIER_SENSORS = {
    "fast_data.bin": {"vibration", "microphone", "magnetometer", "gyroscope", "accelerometer"},
    "medium_data.bin": {"current", "photodiode"},
    "slow_data.bin": {"pressure", "temperature"},
}

PREDICTION_DTYPE = np.dtype(
    [
        ("timestamp_ms", "<u8"),
        ("shi_rf", "<f4"),
        ("shi_xgb", "<f4"),
        ("ground_truth_label", "u1"),
        ("scenario", "u1"),
        ("reserved", "<u2"),
        ("signal", "<f4", (3,)),
    ],
    align=False,
)
assert PREDICTION_DTYPE.itemsize == 32


def feature_dtype(feature_count: int) -> np.dtype:
    dtype = np.dtype(
        [
            ("timestamp_ms", "<u8"),
            ("label", "u1"),
            ("scenario", "u1"),
            ("reserved", "<u2"),
            ("features", "<f4", (feature_count,)),
        ],
        align=False,
    )
    assert dtype.itemsize == 12 + feature_count * 4
    return dtype


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def labels_with_prediction_horizon(
    timestamps: np.ndarray, active: np.ndarray, horizon_seconds: float,
) -> np.ndarray:
    """Create upstream 0/1/2 labels from exact injection activity.

    0 is healthy, 1 is the known pre-fault horizon, and 2 is active fault.
    Unlike upstream's single suffix fault, the random injection data can contain
    multiple active intervals, so every inactive-to-active transition is used.
    """
    timestamps = np.asarray(timestamps, dtype=np.int64)
    active = np.asarray(active, dtype=bool)
    if timestamps.shape != active.shape:
        raise ValueError("timestamps and activity must have identical shapes")
    labels = np.zeros(len(timestamps), dtype=np.uint8)
    labels[active] = 2
    if not len(labels) or horizon_seconds <= 0:
        return labels
    onsets = np.flatnonzero(active & ~np.r_[False, active[:-1]])
    horizon_ms = round(horizon_seconds * 1000)
    for onset in onsets:
        start = np.searchsorted(timestamps, timestamps[onset] - horizon_ms, side="left")
        candidates = np.arange(start, onset)
        candidates = candidates[~active[candidates]]
        labels[candidates] = 1
    return labels


def deterministic_sample(labels: np.ndarray, limit: int) -> np.ndarray:
    """Return stable, label-aware row indices; limit=0 keeps every row."""
    labels = np.asarray(labels)
    if limit <= 0 or len(labels) <= limit:
        return np.arange(len(labels), dtype=np.int64)
    unique, counts = np.unique(labels, return_counts=True)
    allocation = np.maximum(1, np.floor(limit * counts / counts.sum()).astype(int))
    while allocation.sum() > limit:
        target = int(np.argmax(allocation))
        if allocation[target] > 1:
            allocation[target] -= 1
        else:
            break
    while allocation.sum() < limit:
        room = counts - allocation
        target = int(np.argmax(room))
        if room[target] <= 0:
            break
        allocation[target] += 1
    selected: list[np.ndarray] = []
    for label, amount in zip(unique, allocation):
        positions = np.flatnonzero(labels == label)
        if amount >= len(positions):
            selected.append(positions)
        else:
            chosen = np.linspace(0, len(positions) - 1, amount, dtype=np.int64)
            selected.append(positions[chosen])
    return np.sort(np.concatenate(selected))


def healthy_probability(model, values: np.ndarray) -> np.ndarray:
    classes = list(model.classes_)
    if 0 not in classes:
        raise ValueError(f"model has no healthy class 0: {classes}")
    return model.predict_proba(values)[:, classes.index(0)]


def model_ar_setting(models_dir: Path, sensors: set[str]) -> bool:
    """Return the common AR-Burg setting and reject mixed model families."""
    settings = set()
    for sensor in sorted(sensors):
        metadata_path = models_dir / sensor / "model_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        settings.add(bool(metadata["training_config"]["features_include_ar_burg"]))
    if len(settings) != 1:
        raise ValueError("selected sensor models mix AR-Burg and non-AR feature sets")
    return settings.pop()


def safe_component(value: str) -> str:
    allowed = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_."
    result = "".join(character if character in allowed else "_" for character in value)
    return result.strip("._") or "item"


def source_sensors(relative: str) -> set[str]:
    path = Path(relative)
    if path.suffix.lower() == ".csv":
        return {path.stem.lower().removeprefix("proc_")}
    return TIER_SENSORS.get(path.name.lower(), set())
