#!/usr/bin/env python3
"""Plot signal and reconstructed SHIBench reference SHI against time."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import textwrap

import numpy as np

from shibench_reference_shi.core import REFERENCE_DTYPE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="one prediction .bin or an output directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-points", type=int, default=20_000)
    parser.add_argument("--around-fault-seconds", type=float)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_points < 2:
        parser.error("max-points must be at least 2")
    if args.around_fault_seconds is not None and args.around_fault_seconds <= 0:
        parser.error("around-fault-seconds must be positive")
    return args


def plot_one(source: Path, args: argparse.Namespace) -> Path:
    marker = source.with_suffix(".json")
    if source.stat().st_size == 0 or not marker.is_file():
        raise ValueError(f"prediction data or metadata missing: {source}")
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    records = np.memmap(source, dtype=REFERENCE_DTYPE, mode="r")
    first = float(metadata["first_timestamp_ms"])
    elapsed_all = (records["timestamp_ms"].astype(np.float64) - first) / 1000.0
    fault_seconds = (float(metadata["fault_start_ms"]) - first) / 1000.0
    indices = np.arange(len(records))
    if args.around_fault_seconds is not None:
        radius = args.around_fault_seconds
        indices = indices[
            (elapsed_all >= fault_seconds - radius) & (elapsed_all <= fault_seconds + radius)
        ]
    if not len(indices):
        raise ValueError("selected interval contains no prediction windows")
    stride = max(1, int(np.ceil(len(indices) / args.max_points)))
    selected = indices[::stride]
    sample, elapsed = records[selected], elapsed_all[selected]

    cache = Path(tempfile.gettempdir()) / "shibench-reference-shi-mpl"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    figure, (signal_ax, health_ax) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes = int(metadata["job"]["axes"])
    sensor = metadata["job"]["sensor"]
    for axis in range(axes):
        signal_ax.plot(
            elapsed, sample["signal"][:, axis], linewidth=0.8,
            label=("value" if axes == 1 else "xyz"[axis]),
        )
    signal_ax.set_ylabel(sensor)
    signal_ax.grid(alpha=0.25)
    if axes > 1:
        signal_ax.legend(loc="best")

    health_ax.plot(elapsed, sample["shi"], color="#111111", linewidth=1.1,
                   label="fused continuous SHI")
    health_ax.plot(elapsed, sample["mahalanobis_health"], linewidth=0.65, alpha=0.55,
                   label="Mahalanobis")
    health_ax.plot(elapsed, sample["isolation_health"], linewidth=0.65, alpha=0.55,
                   label="Isolation Forest")
    health_ax.plot(elapsed, sample["ewma_health"], linewidth=0.65, alpha=0.55,
                   label="EWMA residual")
    if metadata["calibration"]["event_enabled"]:
        health_ax.plot(elapsed, sample["event_health"], linewidth=0.8, alpha=0.75,
                       label="event-rate health")
    health_ax.axhline(0.5, color="#777777", linestyle=":", linewidth=1.0,
                      label="calibrated alarm threshold")
    health_ax.set_ylabel("Health (0–1)")
    health_ax.set_xlabel("Seconds from signal start")
    health_ax.set_ylim(-0.02, 1.02)
    health_ax.grid(alpha=0.25)
    health_ax.legend(loc="best", ncol=2)
    for axis in (signal_ax, health_ax):
        axis.axvline(fault_seconds, color="#d62728", linestyle="--", linewidth=1.2,
                     label="documented intervention start")
    signal_ax.legend(loc="best")
    member_parts = Path(metadata["job"]["member"]).parts
    short_member = " / ".join(member_parts[-5:])
    member_title = textwrap.fill(short_member, width=95)
    figure.suptitle(
        f"{sensor} [reconstructed SHIBench reference] — {member_title}\n"
        f"baseline={metadata['baseline_strategy']}; calibration=within-recording; "
        f"exact paper reproduction=no",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    destination = (args.output or source.with_suffix(".png")).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


def main() -> int:
    args = parse_args()
    requested = args.input.resolve()
    if requested.is_dir():
        if args.output:
            raise ValueError("--output is valid only for one .bin")
        sources = sorted(requested.rglob("*.bin"))
    else:
        sources = [requested]
    if not sources:
        raise ValueError(f"no prediction .bin files found under {requested}")
    failures = 0
    skipped = 0
    for source in sources:
        if source.stat().st_size == 0:
            print(f"[skipped: empty] {source}")
            skipped += 1
            continue
        if requested.is_dir() and not args.overwrite and source.with_suffix(".png").is_file():
            skipped += 1
            continue
        try:
            print(plot_one(source, args))
        except Exception as error:
            failures += 1
            print(f"[failed] {source}: {error}")
    if skipped:
        print(f"Skipped {skipped} empty or already-plotted file(s).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
