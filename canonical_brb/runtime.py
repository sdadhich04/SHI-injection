"""Shared planning, atomic output, and bounded calibration utilities."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


PIPELINE_VERSION = 1


def stable_seed(*labels: object) -> int:
    digest = hashlib.sha256("\0".join(map(str, labels)).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


class Reservoir:
    """Deterministic Algorithm-R reservoir for bounded healthy calibration."""

    def __init__(self, capacity: int, seed: int) -> None:
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self.rows: list[np.ndarray] = []
        self.seen = 0
        self.names: list[str] | None = None

    def update(self, names: list[str], matrix: np.ndarray) -> None:
        if self.names is None:
            self.names = list(names)
        elif self.names != names:
            raise ValueError("feature schema changed within one sensor stream")
        for row in np.asarray(matrix):
            self.seen += 1
            if len(self.rows) < self.capacity:
                self.rows.append(row.copy())
                continue
            index = int(self.rng.integers(0, self.seen))
            if index < self.capacity:
                self.rows[index] = row.copy()

    def matrix(self) -> np.ndarray:
        if not self.rows:
            raise ValueError("no calibration windows were collected")
        return np.stack(self.rows)


def distribute_jobs(jobs: list[dict], total_batches: int, weight_key: str) -> list[dict]:
    batches = [{"batch": index + 1, "weight": 0, "jobs": []} for index in range(total_batches)]
    for job in sorted(jobs, key=lambda item: (-int(item[weight_key]), str(item))):
        target = min(batches, key=lambda item: (item["weight"], item["batch"]))
        target["jobs"].append(job)
        target["weight"] += int(job[weight_key])
    return batches


def completed_report(path: Path, processing_config: dict) -> dict | None:
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "success" or report.get("processing_config") != processing_config:
        return None
    for output in report.get("outputs", []):
        output_path = Path(output["path"])
        if not output_path.is_file() or output_path.stat().st_size != output["bytes"]:
            return None
    return report


def quarantine_incomplete(destination: Path, output_root: Path) -> Path | None:
    """Preserve an interrupted job directory and free its path for a retry."""
    if not destination.exists() or not any(destination.iterdir()):
        return None
    relative = destination.relative_to(output_root)
    quarantine_root = output_root / "_interrupted" / relative.parent
    quarantine_root.mkdir(parents=True, exist_ok=True)
    quarantine = quarantine_root / f"{destination.name}__{time.time_ns()}"
    os.replace(destination, quarantine)
    return quarantine
