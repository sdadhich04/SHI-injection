#!/usr/bin/env python3
"""Plot Tier-2 signal, recovered fused SHI, components, and fault onset."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import sys
import tempfile
import textwrap

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from tier2_fused_shi.core import TIER2_DTYPE  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path,
        default=PACKAGE_ROOT / "outputs/tier2_fused_shi/hardware",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=PACKAGE_ROOT / "outputs/tier2_fused_shi/hardware_plots",
    )
    parser.add_argument("--max-points", type=int, default=5000)
    parser.add_argument("--around-fault-seconds", type=float)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.max_points < 2 or args.workers < 1:
        parser.error("max-points must be >=2 and workers must be positive")
    if args.around_fault_seconds is not None and args.around_fault_seconds <= 0:
        parser.error("around-fault-seconds must be positive")
    return args


def destination_for(source: Path, input_root: Path, output_root: Path) -> Path:
    return output_root / source.parent.relative_to(input_root) / "current.png"


def plot_one(source_text: str, input_text: str, output_text: str, maximum: int,
             around_fault: float | None, overwrite: bool) -> dict:
    source, input_root, output_root = map(Path, (source_text, input_text, output_text))
    destination = destination_for(source, input_root, output_root)
    if destination.is_file() and not overwrite:
        return {"status": "skipped", "source": str(source), "output": str(destination)}
    marker = source.parent / "job_report.json"
    if source.stat().st_size == 0 or not marker.is_file():
        raise ValueError(f"prediction data or job report missing: {source}")
    if source.stat().st_size % TIER2_DTYPE.itemsize:
        raise ValueError(f"invalid prediction byte count: {source}")
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    records = np.memmap(source, dtype=TIER2_DTYPE, mode="r")
    if not len(records):
        raise ValueError(f"empty prediction file: {source}")
    first = float(metadata["first_timestamp_ms"])
    elapsed_all = (records["timestamp_ms"].astype(np.float64) - first) / 1000.0
    fault_seconds = (float(metadata["fault_start_ms"]) - first) / 1000.0
    indices = np.arange(len(records))
    if around_fault is not None:
        indices = indices[
            (elapsed_all >= fault_seconds - around_fault)
            & (elapsed_all <= fault_seconds + around_fault)
        ]
    if not len(indices):
        raise ValueError("selected interval contains no prediction windows")
    step = max(1, int(np.ceil(len(indices) / maximum)))
    selected = indices[::step]
    sample, elapsed = records[selected], elapsed_all[selected]

    cache = Path(tempfile.gettempdir()) / "shield-tier2-fused-mpl"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, (signal_axis, health_axis) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes = int(metadata["job"]["axes"])
    sensor = metadata["job"]["sensor"]
    for axis in range(axes):
        signal_axis.plot(
            elapsed, sample["signal"][:, axis], linewidth=0.75,
            label="value" if axes == 1 else "xyz"[axis],
        )
    signal_axis.set_ylabel(sensor)
    signal_axis.grid(alpha=0.25)
    if axes > 1:
        signal_axis.legend(loc="best")

    if np.any(np.isfinite(sample["shi"])):
        health_axis.plot(elapsed, sample["shi"], color="#111111", linewidth=1.15,
                         label="fused SHI")
        health_axis.plot(elapsed, sample["isolation_health"], linewidth=0.65, alpha=0.58,
                         label="Isolation Forest health")
        health_axis.plot(elapsed, sample["mahalanobis_health"], linewidth=0.65, alpha=0.58,
                         label="Mahalanobis health")
        health_axis.plot(elapsed, sample["ewma_health"], linewidth=0.65, alpha=0.58,
                         label="EWMA health")
        threshold = float(sample["alarm_threshold"][np.flatnonzero(
            np.isfinite(sample["alarm_threshold"])
        )[0]])
        health_axis.axhline(threshold, color="#777777", linestyle=":", linewidth=1.0,
                            label=f"continuous threshold ({threshold:.3f})")
    model_type = metadata.get("model_type", "")
    if "event" in model_type:
        health_axis.plot(elapsed, sample["event_health"], color="#9467bd", linewidth=0.9,
                         label="event-rate health")

    calibration_start = (float(metadata["calibration_start_ms"]) - first) / 1000.0
    calibration_end = (float(metadata["calibration_end_ms"]) - first) / 1000.0
    for axis in (signal_axis, health_axis):
        axis.axvspan(calibration_start, calibration_end, color="#2ca02c", alpha=0.09,
                     label="calibration interval")
        axis.axvline(fault_seconds, color="#d62728", linestyle="--", linewidth=1.2,
                    label="documented intervention start")
    health_axis.set_ylabel("Health (0–1)")
    health_axis.set_xlabel("Seconds from signal start")
    health_axis.set_ylim(-0.02, 1.02)
    health_axis.grid(alpha=0.25)
    handles, labels = health_axis.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    health_axis.legend(unique.values(), unique.keys(), loc="best", ncol=2, fontsize=8)

    short_member = " / ".join(Path(metadata["job"]["member"]).parts[-5:])
    title = (
        f"{sensor} — recovered fused Tier-2 SHI\n"
        f"{textwrap.fill(short_member, width=100)}\n"
        f"baseline={metadata['baseline_strategy']}; onset confidence="
        f"{metadata['fault_timestamp_confidence']}"
    )
    if metadata.get("calibration_warning"):
        title += f"; WARNING: {metadata['calibration_warning']}"
    figure.suptitle(title, fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.91))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".png.tmp")
    figure.savefig(temporary, format="png", dpi=150)
    plt.close(figure)
    os.replace(temporary, destination)
    return {"status": "success", "source": str(source), "output": str(destination)}


def main() -> int:
    args = parse_args()
    input_root, output_root = args.input_dir.resolve(), args.output_dir.resolve()
    sources = sorted(input_root.rglob("predictions.bin"))
    if not sources:
        raise ValueError(f"no predictions.bin files found under {input_root}")
    successes = skipped = 0
    failures: list[dict] = []

    def accept(source: Path, result: dict | None = None, error: Exception | None = None) -> None:
        nonlocal successes, skipped
        if error is not None:
            failures.append({"source": str(source), "error": str(error)})
            print(f"[ERROR] {source}: {error}", flush=True)
        elif result is not None:
            if result["status"] == "skipped":
                skipped += 1
            else:
                successes += 1
                print(result["output"], flush=True)

    parameters = (str(input_root), str(output_root), args.max_points,
                  args.around_fault_seconds, args.overwrite)
    if args.workers == 1:
        for source in sources:
            try:
                accept(source, result=plot_one(str(source), *parameters))
            except Exception as error:
                accept(source, error=error)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            future_map = {
                executor.submit(plot_one, str(source), *parameters): source for source in sources
            }
            for future in as_completed(future_map):
                source = future_map[future]
                try:
                    accept(source, result=future.result())
                except Exception as error:
                    accept(source, error=error)
    report = {
        "input": str(input_root), "output": str(output_root), "discovered": len(sources),
        "successful": successes, "skipped": skipped, "failed": len(failures),
        "failures": failures,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    temporary = output_root / "plot_report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output_root / "plot_report.json")
    print(f"Plots: {successes} successful, {skipped} skipped, {len(failures)} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
