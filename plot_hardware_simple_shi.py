#!/usr/bin/env python3
"""Plot a hardware sensor signal and its SHI, including the fault onset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import textwrap

import numpy as np

from hardware_simple_shi import HARDWARE_DTYPE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="one hardware SHI .bin, or an output directory")
    parser.add_argument("--output", type=Path, help="PNG destination (default: beside input)")
    parser.add_argument("--max-points", type=int, default=20_000)
    parser.add_argument(
        "--around-fault-seconds", type=float,
        help="show this many seconds before and after fault onset instead of the entire run",
    )
    args = parser.parse_args()
    if args.max_points < 2 or (args.around_fault_seconds is not None and args.around_fault_seconds <= 0):
        parser.error("max-points must be >=2 and around-fault-seconds must be positive")
    return args


def plot_one(source: Path, args: argparse.Namespace) -> Path:
    metadata_path = source.with_suffix(".json")
    if not source.is_file() or not metadata_path.is_file():
        raise ValueError("the .bin and matching .json metadata must both exist")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    records = np.memmap(source, dtype=HARDWARE_DTYPE, mode="r")
    if not len(records):
        raise ValueError(f"no SHI records in {source}")
    first = float(metadata["first_timestamp_ms"])
    fault_seconds = (float(metadata["fault_start_ms"]) - first) / 1000.0
    elapsed_all = (records["timestamp_ms"].astype(np.float64) - first) / 1000.0
    indices = np.arange(len(records))
    if args.around_fault_seconds is not None:
        radius = args.around_fault_seconds
        indices = indices[(elapsed_all >= fault_seconds - radius) & (elapsed_all <= fault_seconds + radius)]
        if not len(indices):
            raise ValueError("selected fault-centered interval has no SHI windows")
    stride = max(1, int(np.ceil(len(indices) / args.max_points)))
    selected = indices[::stride]
    sample = records[selected]
    elapsed = elapsed_all[selected]

    cache = Path(tempfile.gettempdir()) / "hardware-shi-mpl"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib.pyplot as plt

    axes_count = int(metadata["job"]["axes"])
    sensor = metadata["job"]["sensor"]
    figure, (signal_ax, shi_ax) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for axis in range(axes_count):
        signal_ax.plot(elapsed, sample["signal"][:, axis], linewidth=0.8,
                       label=("value" if axes_count == 1 else "xyz"[axis]))
    signal_ax.set_ylabel(sensor)
    signal_ax.grid(alpha=0.25)
    if axes_count > 1:
        signal_ax.legend(loc="best")
    shi_ax.plot(elapsed, sample["shi"], color="#7b2cbf", linewidth=0.9, label="SHI")
    shi_ax.set_ylabel("SHI (0–100)")
    shi_ax.set_xlabel("Seconds from signal start")
    shi_ax.set_ylim(-2, 102)
    shi_ax.grid(alpha=0.25)
    for axis in (signal_ax, shi_ax):
        axis.axvline(fault_seconds, color="#d62728", linestyle="--", linewidth=1.4,
                     label=f"fault starts: {fault_seconds:.3f}s")
    signal_ax.legend(loc="best")
    figure.suptitle(
        f"{sensor} — {textwrap.shorten(metadata['job']['member'], width=130, placeholder='…')}\n"
        f"baseline={metadata['baseline_strategy']}; onset confidence={metadata['fault_timestamp_confidence']}"
    )
    figure.tight_layout()
    destination = (args.output or source.with_suffix(".png")).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160)
    plt.close(figure)
    return destination


def main() -> int:
    args = parse_args()
    requested = args.input.resolve()
    if requested.is_dir():
        if args.output is not None:
            raise ValueError("--output can only be used when plotting one .bin file")
        sources = sorted(requested.rglob("*.bin"))
    else:
        sources = [requested]
    if not sources:
        raise ValueError(f"no hardware SHI .bin files found under {requested}")
    for source in sources:
        print(plot_one(source, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
