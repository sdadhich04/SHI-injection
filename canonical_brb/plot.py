#!/usr/bin/env python3
"""Plot signal and canonical BRB-r SHI timelines from binary outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from canonical_brb.run_hardware import HARDWARE_DTYPE
from canonical_brb.run_software import SOFTWARE_DTYPE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("software", "hardware"))
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-points", type=int, default=5000)
    return parser.parse_args()


def _downsample(count: int, maximum: int) -> np.ndarray:
    return np.arange(0, count, max(1, int(np.ceil(count / maximum))))


def plot_software(source: Path, root: Path, output_root: Path, maximum: int) -> Path | None:
    if source.stat().st_size == 0:
        return None
    records = np.memmap(source, dtype=SOFTWARE_DTYPE, mode="r")
    if not len(records):
        return None
    selected = _downsample(len(records), maximum)
    rows = records[selected]
    time = (rows["timestamp_ms"].astype(np.float64) - float(rows["timestamp_ms"][0])) / 1000.0
    observed = rows["observed"]
    axes = 3 if np.any(observed[:, 1:] != 0) else 1
    signal = np.linalg.norm(observed[:, :axes], axis=1) if axes > 1 else observed[:, 0]
    figure, (signal_axis, health_axis) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    signal_axis.plot(time, signal, linewidth=0.7, color="#2455a4")
    signal_axis.set_ylabel("signal norm" if axes > 1 else "signal")
    signal_axis.grid(alpha=0.25)
    health_axis.plot(time, rows["shi"], linewidth=0.9, color="#159447", label="BRB-r SHI")
    active = rows["injected_fraction"] > 0
    if np.any(active):
        health_axis.fill_between(time, 0, 1, where=active, color="#d95f02", alpha=0.15,
                                 label="injection active in window")
    health_axis.set_ylim(-0.02, 1.02); health_axis.set_ylabel("SHI"); health_axis.set_xlabel("time (s)")
    health_axis.grid(alpha=0.25); health_axis.legend(loc="best")
    figure.suptitle(str(source.relative_to(root)))
    destination = (output_root / source.relative_to(root)).with_suffix(".png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(destination, dpi=140); plt.close(figure)
    return destination


def plot_hardware(source: Path, root: Path, output_root: Path, maximum: int) -> Path | None:
    if source.stat().st_size == 0:
        return None
    records = np.memmap(source, dtype=HARDWARE_DTYPE, mode="r")
    if not len(records):
        return None
    report = json.loads((source.parent / "job_report.json").read_text(encoding="utf-8"))
    selected = _downsample(len(records), maximum)
    rows = records[selected]
    first_timestamp = float(records["timestamp_ms"][0])
    time = (rows["timestamp_ms"].astype(np.float64) - first_timestamp) / 1000.0
    axes = int(report["job"]["axes"])
    signal = np.linalg.norm(rows["signal"][:, :axes], axis=1) if axes > 1 else rows["signal"][:, 0]
    fault_time = (float(report["fault_timestamp_ms"]) - first_timestamp) / 1000.0
    figure, (signal_axis, health_axis) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    signal_axis.plot(time, signal, linewidth=0.7, color="#2455a4")
    signal_axis.set_ylabel("signal norm" if axes > 1 else "signal"); signal_axis.grid(alpha=0.25)
    health_axis.plot(time, rows["shi"], linewidth=0.9, color="#159447", label="BRB-r SHI")
    for axis in (signal_axis, health_axis):
        axis.axvline(fault_time, color="#d95f02", linestyle="--", linewidth=1.2, label="fault start")
    health_axis.set_ylim(-0.02, 1.02); health_axis.set_ylabel("SHI"); health_axis.set_xlabel("time (s)")
    health_axis.grid(alpha=0.25); health_axis.legend(loc="best")
    figure.suptitle(f"{report['job']['sensor']} — {report['job']['member']}")
    destination = (output_root / source.parent.relative_to(root) / "current.png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(destination, dpi=140); plt.close(figure)
    return destination


def main() -> int:
    args = parse_args()
    root, output = args.input_dir.resolve(), args.output_dir.resolve()
    sources = (
        [path for path in root.rglob("*.bin") if "__model" not in path.name]
        if args.kind == "software" else list(root.rglob("predictions.bin"))
    )
    failures = []
    for source in sources:
        try:
            destination = (plot_software if args.kind == "software" else plot_hardware)(
                source, root, output, args.max_points,
            )
            print(destination if destination else f"[skipped empty] {source}")
        except Exception as error:
            failures.append((source, error))
            print(f"ERROR {source}: {error}")
    print(f"Plots: {len(sources) - len(failures)} successful, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
