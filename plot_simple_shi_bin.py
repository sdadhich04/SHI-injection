#!/usr/bin/env python3
"""Plot sensor output and SHI directly from one binary analysis file."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_simple_shi import BINARY_DTYPE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-points", type=int, default=10_000)
    parser.add_argument("--axes", type=int, choices=(1, 2, 3), default=3)
    args = parser.parse_args()
    if args.max_points < 2:
        parser.error("--max-points must be at least 2")
    return args


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise ValueError(f"input file not found: {args.input}")
    size = args.input.stat().st_size
    if size == 0 or size % BINARY_DTYPE.itemsize:
        raise ValueError(
            f"invalid SHI binary size {size}; expected a positive multiple of {BINARY_DTYPE.itemsize}"
        )
    records = np.memmap(args.input, dtype=BINARY_DTYPE, mode="r")
    step = max(1, int(np.ceil(len(records) / args.max_points)))
    sample = records[::step]
    elapsed = (sample["timestamp_ms"].astype(np.float64) - float(sample["timestamp_ms"][0])) / 1000.0
    output = args.output or args.input.with_suffix(".png")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing plot: {output}")

    figure, (signal_axis, health_axis) = plt.subplots(
        2, 1, figsize=(13, 7), sharex=True, constrained_layout=True
    )
    labels = ("x", "y", "z")[: args.axes]
    for index, label in enumerate(labels):
        signal_axis.plot(
            elapsed, sample["ground_truth"][:, index], color="#777777", alpha=0.65,
            linewidth=0.9, label=f"ground truth {label}",
        )
        signal_axis.plot(
            elapsed, sample["observed"][:, index], alpha=0.8,
            linewidth=0.8, label=f"observed {label}",
        )
    signal_axis.set_ylabel("Sensor output")
    signal_axis.grid(alpha=0.2)
    signal_axis.legend(loc="upper right", fontsize=8, ncol=min(3, args.axes))

    active = sample["injected_fraction"] > 0
    health_axis.fill_between(
        elapsed, 0, 100, where=active, color="#d62728", alpha=0.10,
        step="mid", label="injected-noise window",
    )
    health_axis.plot(elapsed, sample["shi"], color="#111111", linewidth=1.0, label="SHI")
    health_axis.set_ylim(-2, 102)
    health_axis.set_ylabel("SHI (0–100)")
    health_axis.set_xlabel("Elapsed time (s)")
    health_axis.grid(alpha=0.2)
    health_axis.legend(loc="lower left", fontsize=8)
    figure.suptitle(args.input.stem.replace("__", " · "))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"Wrote {output} from {len(records):,} records (plotted every {step} record(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
