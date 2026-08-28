#!/usr/bin/env python3
"""Plot signal and upstream-model P(healthy) SHI with known fault timing."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import textwrap

import numpy as np

from common import PREDICTION_DTYPE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="one prediction .bin or an output directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-points", type=int, default=20_000)
    parser.add_argument("--around-first-fault-seconds", type=float)
    parser.add_argument("--overwrite", action="store_true",
                        help="regenerate PNGs that already exist when plotting a directory")
    args = parser.parse_args()
    if args.max_points < 2:
        parser.error("max-points must be at least 2")
    if args.around_first_fault_seconds is not None and args.around_first_fault_seconds <= 0:
        parser.error("around-first-fault-seconds must be positive")
    return args


def metadata_for(source: Path) -> tuple[dict, dict]:
    sidecar = source.with_suffix(".json")
    if sidecar.is_file():
        report = json.loads(sidecar.read_text(encoding="utf-8"))
        return report, {"sensor": report["job"]["sensor"], "axes": report["job"]["axes"],
                        "scenario": "hardware", "source": report["job"]["member"]}
    marker = source.parent / "job_report.json"
    if not marker.is_file():
        raise ValueError(f"metadata not found for {source}")
    report = json.loads(marker.read_text(encoding="utf-8"))
    source_resolved = source.resolve()
    output = next((item for item in report["outputs"] if Path(item["path"]).resolve() == source_resolved), None)
    if output is None:
        raise ValueError(f"{source} is not listed in {marker}")
    return report, {"sensor": output["sensor"], "axes": output["axes"],
                    "scenario": output["scenario"], "source": report["job"]["clean_relative"]}


def active_onsets(records: np.ndarray) -> np.ndarray:
    active = records["ground_truth_label"] == 2
    return np.flatnonzero(active & ~np.r_[False, active[:-1]])


def plot_one(source: Path, args) -> Path:
    if source.stat().st_size == 0:
        raise ValueError(f"no full SHI windows were produced: {source}")
    report, info = metadata_for(source)
    records = np.memmap(source, dtype=PREDICTION_DTYPE, mode="r")
    if not len(records): raise ValueError(f"empty prediction file: {source}")
    first = float(report.get("first_timestamp_ms", records[0]["timestamp_ms"]))
    elapsed_all = (records["timestamp_ms"].astype(np.float64) - first) / 1000.0
    onsets = active_onsets(records)
    if info["scenario"] == "hardware":
        fault_times = np.asarray([report["fault_start_seconds_from_signal"]], dtype=float)
    else:
        fault_times = elapsed_all[onsets]
    indices = np.arange(len(records))
    view_start = 0.0
    view_end = float(elapsed_all[-1])
    if args.around_first_fault_seconds is not None and len(fault_times):
        radius, center = args.around_first_fault_seconds, fault_times[0]
        view_start = max(0.0, center - radius)
        view_end = center + radius
        indices = indices[(elapsed_all >= center - radius) & (elapsed_all <= center + radius)]
    if not len(indices): raise ValueError("selected interval has no prediction windows")
    stride = max(1, int(np.ceil(len(indices) / args.max_points)))
    selected = indices[::stride]
    sample, elapsed = records[selected], elapsed_all[selected]

    cache = Path(tempfile.gettempdir()) / "model-shi-mpl"; cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt
    figure, (signal_ax, shi_ax) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for axis in range(info["axes"]):
        signal_ax.plot(elapsed, sample["signal"][:, axis], linewidth=0.8,
                       label=("value" if info["axes"] == 1 else "xyz"[axis]))
    signal_ax.set_ylabel(info["sensor"]); signal_ax.grid(alpha=0.25)
    shi_ax.plot(elapsed, sample["shi_rf"], linewidth=0.9, label="RandomForest P(healthy)")
    shi_ax.plot(elapsed, sample["shi_xgb"], linewidth=0.9, label="XGBoost P(healthy)")
    shi_ax.set_ylabel("Model SHI (0–1)"); shi_ax.set_xlabel("Seconds from signal start")
    shi_ax.set_ylim(-0.02, 1.02); shi_ax.grid(alpha=0.25); shi_ax.legend(loc="best")
    visible_faults = fault_times[(fault_times >= view_start) & (fault_times <= view_end)]
    for index, fault_time in enumerate(visible_faults):
        for axis in (signal_ax, shi_ax):
            axis.axvline(fault_time, color="#d62728", linestyle="--", linewidth=1.1,
                         label="known fault start" if index == 0 else None)
    if len(visible_faults): signal_ax.legend(loc="best")
    elif info["axes"] > 1: signal_ax.legend(loc="best")
    figure.suptitle(
        f"{info['sensor']} [{info['scenario']}] — "
        f"{textwrap.shorten(info['source'], width=120, placeholder='…')}"
    )
    figure.tight_layout()
    destination = (args.output or source.with_suffix(".png")).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160); plt.close(figure)
    return destination


def main() -> int:
    args = parse_args(); requested = args.input.resolve()
    if requested.is_dir():
        if args.output: raise ValueError("--output is only valid for one input file")
        sources = sorted(requested.rglob("*.bin"))
    else: sources = [requested]
    if not sources: raise ValueError(f"no prediction .bin files found under {requested}")
    skipped_existing = 0
    skipped_empty = 0
    failures: list[tuple[Path, Exception]] = []
    for source in sources:
        # A zero-length result is valid: the source signal was shorter than
        # the SHI analysis window, so inference had no record to emit.
        if source.stat().st_size == 0:
            skipped_empty += 1
            print(f"[skipped: no full SHI window] {source}")
            continue
        if requested.is_dir() and not args.overwrite and source.with_suffix(".png").is_file():
            skipped_existing += 1
            continue
        try:
            print(plot_one(source, args))
        except Exception as error:
            failures.append((source, error))
            print(f"[failed] {source}: {error}")
    if skipped_existing:
        print(f"[skipped: existing PNG] {skipped_existing}")
    if skipped_empty:
        print(f"[skipped: no full SHI window] {skipped_empty}")
    if failures:
        print(f"[failed] {len(failures)} plot(s)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
