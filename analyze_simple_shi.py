#!/usr/bin/env python3
"""Process every SHIELD run into batched, parallel, binary SHI outputs.

Only the random_0_25 and random_0_5 injection variants are processed. Inputs
are opened read-only; every output is written through a temporary file and
atomically renamed after its source job succeeds.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import glob
import hashlib
from itertools import zip_longest
import json
import os
from pathlib import Path
import time
from typing import BinaryIO, Iterator

import numpy as np

from quick_visualizer import SENSOR_NAMES, Record, detect_binary_layout, iter_binary, iter_csv
from simple_shi_features import extract_feature_batch, get_pipeline_feature_names


PIPELINE_VERSION = 2
UPSTREAM_COMMIT = "c094797c96923449b6075d8c483df37856b26712"
VARIANTS = ("random_0_25", "random_0_5")
WINDOW_SIZE = 256
STRIDE = 64
BINARY_DTYPE = np.dtype(
    [
        ("timestamp_ms", "<u8"),
        ("shi", "<f4"),
        ("feature_distance", "<f4"),
        ("injected_fraction", "<f4"),
        ("ground_truth", "<f4", (3,)),
        ("observed", "<f4", (3,)),
    ],
    align=False,
)
assert BINARY_DTYPE.itemsize == 44


@dataclass
class StreamState:
    name: str
    axes: int
    clean: deque
    observed: dict[str, deque]
    timestamps: deque
    changed: dict[str, deque]
    count: int = 0
    fs: float | None = None
    pending_clean: list[np.ndarray] = field(default_factory=list)
    pending_observed: dict[str, list[np.ndarray]] = field(default_factory=dict)
    pending_timestamps: list[int] = field(default_factory=list)
    pending_clean_end: list[np.ndarray] = field(default_factory=list)
    pending_observed_end: dict[str, list[np.ndarray]] = field(default_factory=dict)
    pending_changed: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class WindowBatch:
    sensor: str
    axes: int
    fs: float
    timestamps: np.ndarray
    clean_windows: np.ndarray
    observed_windows: dict[str, np.ndarray]
    clean_end: np.ndarray
    observed_end: dict[str, np.ndarray]
    injected_fraction: dict[str, np.ndarray]


@dataclass
class RunningStats:
    count: int = 0
    mean: np.ndarray | None = None
    m2: np.ndarray | None = None
    abs_sum: np.ndarray | None = None

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if not len(values):
            return
        batch_count = len(values)
        batch_mean = np.mean(values, axis=0)
        batch_m2 = np.sum((values - batch_mean) ** 2, axis=0)
        batch_abs = np.sum(np.abs(values), axis=0)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            self.abs_sum = batch_abs
            return
        assert self.mean is not None and self.m2 is not None and self.abs_sum is not None
        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.m2 += batch_m2 + delta**2 * self.count * batch_count / total
        self.mean += delta * batch_count / total
        self.abs_sum += batch_abs
        self.count = total

    def scale(self, relative_tolerance: float) -> np.ndarray:
        if self.count == 0 or self.m2 is None or self.abs_sum is None:
            raise ValueError("cannot calculate scale without calibration windows")
        standard_deviation = np.sqrt(self.m2 / max(1, self.count - 1))
        mean_absolute = self.abs_sum / self.count
        floor = np.finfo(np.float64).eps * 100
        return np.maximum(np.maximum(standard_deviation, relative_tolerance * mean_absolute), floor)


@dataclass
class ResultSummary:
    windows: int = 0
    active_windows: int = 0
    shi_sum: float = 0.0
    shi_active_sum: float = 0.0
    shi_clean_sum: float = 0.0
    clean_windows: int = 0
    first_timestamp_ms: int | None = None
    last_timestamp_ms: int | None = None

    def update(self, timestamps: np.ndarray, shi: np.ndarray, fractions: np.ndarray) -> None:
        active = fractions > 0
        count = len(shi)
        self.windows += count
        self.active_windows += int(np.sum(active))
        self.clean_windows += int(np.sum(~active))
        self.shi_sum += float(np.sum(shi))
        self.shi_active_sum += float(np.sum(shi[active]))
        self.shi_clean_sum += float(np.sum(shi[~active]))
        if count:
            first, last = int(timestamps[0]), int(timestamps[-1])
            self.first_timestamp_ms = first if self.first_timestamp_ms is None else min(self.first_timestamp_ms, first)
            self.last_timestamp_ms = last if self.last_timestamp_ms is None else max(self.last_timestamp_ms, last)

    def as_dict(self) -> dict:
        return {
            "windows": self.windows,
            "active_windows": self.active_windows,
            "shi_mean": self.shi_sum / self.windows if self.windows else None,
            "shi_active_mean": self.shi_active_sum / self.active_windows if self.active_windows else None,
            "shi_clean_mean": self.shi_clean_sum / self.clean_windows if self.clean_windows else None,
            "first_timestamp_ms": self.first_timestamp_ms,
            "last_timestamp_ms": self.last_timestamp_ms,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir", type=Path,
        default=Path("../noise_injection_shield/outputs/post_noise_injection_1"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/simple_shi_batches"),
    )
    parser.add_argument("--batch", type=int, help="one-based batch index to process")
    parser.add_argument("--total-batches", type=int, default=5)
    parser.add_argument("--plan-only", action="store_true", help="write/validate the plan without processing")
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--feature-batch-windows", type=int, default=256)
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--relative-tolerance", type=float, default=0.10)
    parser.add_argument(
        "--include-ar", action="store_true",
        help="include AR-Burg features (disabled by default upstream and substantially slower)",
    )
    args = parser.parse_args()
    if args.total_batches < 1:
        parser.error("--total-batches must be positive")
    if not args.plan_only and args.batch is None:
        parser.error("--batch is required unless --plan-only is used")
    if args.batch is not None and not 1 <= args.batch <= args.total_batches:
        parser.error("--batch must be between 1 and --total-batches")
    if args.workers < 1 or args.feature_batch_windows < 1 or args.stride < 1:
        parser.error("workers, feature-batch-windows, and stride must be positive")
    if args.window_size < 16 or args.relative_tolerance <= 0:
        parser.error("window-size must be at least 16 and relative-tolerance must be positive")
    return args


def write_json_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")
    os.replace(temporary, path)


def noisy_path(dataset: Path, clean_relative: Path, variant: str) -> Path:
    path = dataset / "noise_injection" / clean_relative.parts[0] / variant
    path = path.joinpath(*clean_relative.parts[1:])
    if clean_relative.suffix.lower() == ".csv":
        path = path.with_suffix(".bin")
    return path


def load_record_counts(dataset: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    reports = sorted(glob.glob(str(dataset / "noise_injection" / "processing_report_batch_*.json")))
    for report_path in reports:
        with open(report_path, "r", encoding="utf-8") as stream:
            report = json.load(stream)
        for success in report.get("successes", []):
            _archive, member = success["source"].split(":", 1)
            counts[Path(member).as_posix()] = int(success["records"])
    return counts


def fallback_record_count(path: Path) -> int:
    if path.suffix.lower() == ".bin":
        return path.stat().st_size // detect_binary_layout(path).size
    raise ValueError(f"missing record count for CSV input {path}")


def discover_jobs(dataset: Path) -> list[dict]:
    ground_truth = dataset / "ground_truth"
    if not ground_truth.is_dir():
        raise ValueError(f"ground-truth directory not found: {ground_truth}")
    record_counts = load_record_counts(dataset)
    jobs = []
    for clean in sorted(ground_truth.rglob("*")):
        if not clean.is_file() or clean.suffix.lower() not in {".bin", ".csv"}:
            continue
        relative = clean.relative_to(ground_truth)
        observed = {variant: noisy_path(dataset, relative, variant) for variant in VARIANTS}
        if not all(path.is_file() for path in observed.values()):
            continue
        key = relative.as_posix()
        records = record_counts.get(key, fallback_record_count(clean) if clean.suffix.lower() == ".bin" else None)
        if records is None:
            raise ValueError(f"record count unavailable for {relative}")
        jobs.append(
            {
                "job_id": hashlib.sha256(key.encode()).hexdigest()[:16],
                "clean_relative": key,
                "records": int(records),
            }
        )
    if not jobs:
        raise ValueError("no ground-truth files with both random variants were found")
    return jobs


def dataset_fingerprint(dataset: Path, jobs: list[dict]) -> str:
    digest = hashlib.sha256()
    for job in sorted(jobs, key=lambda item: item["clean_relative"]):
        relative = Path(job["clean_relative"])
        paths = [dataset / "ground_truth" / relative]
        paths.extend(noisy_path(dataset, relative, variant) for variant in VARIANTS)
        for path in paths:
            stat = path.stat()
            digest.update(f"{path.relative_to(dataset)}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def make_plan(dataset: Path, jobs: list[dict], total_batches: int) -> dict:
    batches = [{"batch": index + 1, "records": 0, "jobs": []} for index in range(total_batches)]
    for job in sorted(jobs, key=lambda item: (-item["records"], item["clean_relative"])):
        target = min(batches, key=lambda item: (item["records"], item["batch"]))
        target["jobs"].append(job)
        target["records"] += job["records"]
    for batch in batches:
        batch["jobs"].sort(key=lambda item: item["clean_relative"])
    return {
        "pipeline_version": PIPELINE_VERSION,
        "dataset_fingerprint": dataset_fingerprint(dataset, jobs),
        "total_batches": total_batches,
        "variants": list(VARIANTS),
        "total_jobs": len(jobs),
        "total_records_per_variant": sum(job["records"] for job in jobs),
        "batches": batches,
    }


def ensure_plan(dataset: Path, output: Path, total_batches: int) -> dict:
    jobs = discover_jobs(dataset)
    expected = make_plan(dataset, jobs, total_batches)
    plan_path = output / "batch_plan.json"
    if plan_path.exists():
        with plan_path.open("r", encoding="utf-8") as stream:
            existing = json.load(stream)
        if existing != expected:
            raise ValueError(
                f"existing plan does not match current data/settings: {plan_path}; "
                "use a new output directory to preserve prior results"
            )
        return existing
    write_json_atomic(plan_path, expected)
    return expected


def record_iterator(path: Path) -> Iterator[Record]:
    if path.suffix.lower() == ".csv":
        return iter_csv(path)
    return iter_binary(path, detect_binary_layout(path))


def estimate_fs(timestamps: np.ndarray) -> float:
    differences = np.diff(timestamps.astype(np.float64))
    differences = differences[differences > 0]
    return 1000.0 / float(np.median(differences)) if len(differences) else 1.0


def make_state(record: Record, variants: tuple[str, ...], window_size: int) -> StreamState:
    name = SENSOR_NAMES.get(record.sensor_id, f"sensor_{record.sensor_id}")
    if record.kind:
        name = f"proc_{name}"
    return StreamState(
        name=name,
        axes=max(1, min(3, record.axis_count)),
        clean=deque(maxlen=window_size),
        observed={variant: deque(maxlen=window_size) for variant in variants},
        timestamps=deque(maxlen=window_size),
        changed={variant: deque(maxlen=window_size) for variant in variants},
        pending_observed={variant: [] for variant in variants},
        pending_observed_end={variant: [] for variant in variants},
        pending_changed={variant: [] for variant in variants},
    )


def flush_state(state: StreamState) -> WindowBatch:
    batch = WindowBatch(
        sensor=state.name,
        axes=state.axes,
        fs=float(state.fs),
        timestamps=np.asarray(state.pending_timestamps, dtype=np.uint64),
        clean_windows=np.stack(state.pending_clean),
        observed_windows={key: np.stack(value) for key, value in state.pending_observed.items()},
        clean_end=np.stack(state.pending_clean_end),
        observed_end={key: np.stack(value) for key, value in state.pending_observed_end.items()},
        injected_fraction={key: np.asarray(value) for key, value in state.pending_changed.items()},
    )
    state.pending_clean.clear()
    state.pending_timestamps.clear()
    state.pending_clean_end.clear()
    for values in state.pending_observed.values():
        values.clear()
    for values in state.pending_observed_end.values():
        values.clear()
    for values in state.pending_changed.values():
        values.clear()
    return batch


def iter_window_batches(
    clean_path: Path,
    observed_paths: dict[str, Path],
    window_size: int,
    stride: int,
    batch_windows: int,
) -> Iterator[WindowBatch]:
    variants = tuple(observed_paths)
    clean_stream = record_iterator(clean_path)
    observed_streams = {key: record_iterator(path) for key, path in observed_paths.items()}
    streams = [clean_stream, *(observed_streams[key] for key in variants)]
    sentinel = object()
    states: dict[tuple[int, int], StreamState] = {}

    for records in zip_longest(*streams, fillvalue=sentinel):
        if records[0] is sentinel:
            if any(record is not sentinel for record in records[1:]):
                raise ValueError(f"observed file has extra records for {clean_path}")
            break
        if any(record is sentinel for record in records[1:]):
            raise ValueError(f"observed file ended before {clean_path}")
        clean = records[0]
        observed = dict(zip(variants, records[1:]))
        assert isinstance(clean, Record)
        if any(clean.key != record.key or clean.axis_count != record.axis_count for record in observed.values()):
            raise ValueError(f"unaligned clean/noisy records for {clean_path}")
        key = clean.sensor_id, clean.kind
        state = states.setdefault(key, make_state(clean, variants, window_size))
        if state.axes != max(1, min(3, clean.axis_count)):
            raise ValueError(f"axis count changed within {clean_path}")
        clean_values = np.asarray(clean.values[: state.axes], dtype=np.float64)
        state.clean.append(clean_values)
        state.timestamps.append(clean.timestamp)
        for variant, record in observed.items():
            values = np.asarray(record.values[: state.axes], dtype=np.float64)
            state.observed[variant].append(values)
            state.changed[variant].append(float(np.any(values != clean_values)))
        state.count += 1
        if len(state.clean) < window_size or (state.count - window_size) % stride:
            continue
        if state.fs is None:
            state.fs = estimate_fs(np.asarray(state.timestamps))
        state.pending_clean.append(np.asarray(state.clean).copy())
        state.pending_timestamps.append(clean.timestamp)
        state.pending_clean_end.append(clean_values)
        for variant in variants:
            state.pending_observed[variant].append(np.asarray(state.observed[variant]).copy())
            state.pending_observed_end[variant].append(np.asarray(state.observed[variant][-1]).copy())
            state.pending_changed[variant].append(float(np.mean(state.changed[variant])))
        if len(state.pending_clean) >= batch_windows:
            yield flush_state(state)

    for state in states.values():
        if state.pending_clean:
            yield flush_state(state)


def extract_multiaxis(windows: np.ndarray, fs: float, include_ar: bool) -> np.ndarray:
    parts = [extract_feature_batch(windows[:, :, axis], fs, include_ar) for axis in range(windows.shape[2])]
    return np.concatenate(parts, axis=1)


def calculate_shi(
    clean_features: np.ndarray,
    observed_features: np.ndarray,
    scale: np.ndarray,
    injected_fraction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    normalized = np.minimum(np.abs(observed_features - clean_features) / scale, 10.0)
    distance = np.sqrt(np.mean(normalized**2, axis=1))
    shi = 100.0 * np.exp(-distance)
    shi[injected_fraction == 0.0] = 100.0
    return shi, distance


def input_signature(paths: list[Path]) -> list[tuple[str, int, int]]:
    return [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in paths]


def completed_job_report(path: Path, processing_config: dict) -> dict | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as stream:
        report = json.load(stream)
    if report.get("status") != "success":
        return None
    if report.get("processing_config") != processing_config:
        return None
    for output in report.get("outputs", []):
        output_path = Path(output["path"])
        if not output_path.is_file() or output_path.stat().st_size != output["bytes"]:
            return None
    return report


def process_job(job: dict, config: dict) -> dict:
    started = time.monotonic()
    dataset = Path(config["dataset"])
    output_root = Path(config["output"])
    relative = Path(job["clean_relative"])
    clean_path = dataset / "ground_truth" / relative
    observed_paths = {variant: noisy_path(dataset, relative, variant) for variant in VARIANTS}
    input_paths = [clean_path, *observed_paths.values()]
    signature_before = input_signature(input_paths)
    source_output = output_root / relative.parent / relative.stem
    marker = source_output / "job_report.json"
    processing_config = {
        "window_size": config["window_size"],
        "stride": config["stride"],
        "feature_batch_windows": config["feature_batch_windows"],
        "relative_tolerance": config["relative_tolerance"],
        "include_ar": config["include_ar"],
        "variants": list(VARIANTS),
        "binary_record_bytes": BINARY_DTYPE.itemsize,
    }
    previous = completed_job_report(marker, processing_config)
    if previous:
        return {"status": "skipped", "job": job, "report": previous}
    if marker.exists() or (source_output.exists() and any(source_output.iterdir())):
        raise FileExistsError(
            f"incomplete existing output for {relative}: {source_output}; "
            "use a new output directory to avoid overwriting it"
        )
    source_output.mkdir(parents=True, exist_ok=True)

    calibration: dict[str, RunningStats] = defaultdict(RunningStats)
    for batch in iter_window_batches(
        clean_path, {}, config["window_size"], config["stride"], config["feature_batch_windows"]
    ):
        features = extract_multiaxis(batch.clean_windows, batch.fs, config["include_ar"])
        calibration[batch.sensor].update(features)
    scales = {sensor: stats.scale(config["relative_tolerance"]) for sensor, stats in calibration.items()}

    handles: dict[tuple[str, str], BinaryIO] = {}
    paths: dict[tuple[str, str], tuple[Path, Path]] = {}
    summaries: dict[tuple[str, str], ResultSummary] = defaultdict(ResultSummary)
    try:
        for batch in iter_window_batches(
            clean_path, observed_paths, config["window_size"], config["stride"], config["feature_batch_windows"]
        ):
            clean_features = extract_multiaxis(batch.clean_windows, batch.fs, config["include_ar"])
            scale = scales[batch.sensor]
            for variant in VARIANTS:
                observed_features = extract_multiaxis(batch.observed_windows[variant], batch.fs, config["include_ar"])
                fractions = batch.injected_fraction[variant]
                shi, distance = calculate_shi(clean_features, observed_features, scale, fractions)
                key = batch.sensor, variant
                if key not in handles:
                    final = source_output / f"{batch.sensor}__{variant}.bin"
                    temporary = final.with_name(f"{final.name}.tmp")
                    if final.exists() or temporary.exists():
                        raise FileExistsError(f"refusing to overwrite {final} or {temporary}")
                    paths[key] = temporary, final
                    handles[key] = temporary.open("wb")
                output = np.zeros(len(shi), dtype=BINARY_DTYPE)
                output["timestamp_ms"] = batch.timestamps
                output["shi"] = shi
                output["feature_distance"] = distance
                output["injected_fraction"] = fractions
                output["ground_truth"][:, : batch.axes] = batch.clean_end
                output["observed"][:, : batch.axes] = batch.observed_end[variant]
                handles[key].write(output.tobytes())
                summaries[key].update(batch.timestamps, shi, fractions)
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
            handle.close()
        handles.clear()
        for temporary, final in paths.values():
            os.replace(temporary, final)
    except Exception:
        for handle in handles.values():
            handle.close()
        for temporary, _final in paths.values():
            temporary.unlink(missing_ok=True)
        raise

    if signature_before != input_signature(input_paths):
        raise RuntimeError(f"input changed while processing {relative}")
    outputs = []
    for key, (_temporary, final) in sorted(paths.items()):
        sensor, variant = key
        outputs.append(
            {
                "sensor": sensor,
                "variant": variant,
                "path": str(final),
                "bytes": final.stat().st_size,
                "record_bytes": BINARY_DTYPE.itemsize,
                **summaries[key].as_dict(),
            }
        )
    report = {
        "status": "success",
        "job": job,
        "processing_config": processing_config,
        "elapsed_seconds": time.monotonic() - started,
        "input_signature_unchanged": True,
        "outputs": outputs,
    }
    write_json_atomic(marker, report)
    return {"status": "success", "job": job, "report": report}


def validate_output_location(dataset: Path, output: Path) -> None:
    forbidden = (dataset, dataset / "ground_truth", dataset / "noise_injection")
    if any(output == path or path in output.parents for path in forbidden):
        raise ValueError("output directory must be outside the input dataset tree")


def main() -> int:
    args = parse_args()
    dataset = args.dataset_dir.resolve()
    output = args.output_dir.resolve()
    validate_output_location(dataset, output)
    plan = ensure_plan(dataset, output, args.total_batches)
    print(
        f"Plan: {plan['total_jobs']} jobs, {plan['total_batches']} batches, "
        f"{plan['total_records_per_variant']:,} records/variant"
    )
    if args.plan_only:
        for batch in plan["batches"]:
            print(f"Batch {batch['batch']}: {len(batch['jobs'])} jobs, {batch['records']:,} records/variant")
        return 0

    selected = plan["batches"][args.batch - 1]
    config = {
        "dataset": str(dataset),
        "output": str(output),
        "window_size": args.window_size,
        "stride": args.stride,
        "feature_batch_windows": args.feature_batch_windows,
        "relative_tolerance": args.relative_tolerance,
        "include_ar": args.include_ar,
    }
    print(
        f"Processing batch {args.batch}/{args.total_batches}: {len(selected['jobs'])} jobs, "
        f"workers={args.workers}, variants={','.join(VARIANTS)}"
    )
    started = time.monotonic()
    successes, skipped, failures = [], [], []
    def accept(index: int, total: int, job: dict, result=None, error=None) -> None:
        if error is not None:
            failures.append(
                {"job": job, "error_type": type(error).__name__, "message": str(error)}
            )
            print(f"[{index}/{total}] ERROR {job['clean_relative']}: {error}")
        else:
            target = skipped if result["status"] == "skipped" else successes
            target.append(result)
            print(f"[{index}/{total}] {result['status'].upper()} {job['clean_relative']}")

    if args.workers == 1:
        for index, job in enumerate(selected["jobs"], 1):
            try:
                accept(index, len(selected["jobs"]), job, result=process_job(job, config))
            except Exception as error:
                accept(index, len(selected["jobs"]), job, error=error)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_job, job, config): job for job in selected["jobs"]}
            for index, future in enumerate(as_completed(futures), 1):
                job = futures[future]
                try:
                    accept(index, len(futures), job, result=future.result())
                except Exception as error:
                    accept(index, len(futures), job, error=error)

    report = {
        "pipeline_version": PIPELINE_VERSION,
        "upstream_commit": UPSTREAM_COMMIT,
        "batch": args.batch,
        "total_batches": args.total_batches,
        "variants": list(VARIANTS),
        "workers": args.workers,
        "window_size": args.window_size,
        "stride": args.stride,
        "feature_batch_windows": args.feature_batch_windows,
        "include_ar": args.include_ar,
        "feature_count_scalar": len(get_pipeline_feature_names(False, args.include_ar)),
        "binary_record_format": "<Q9f (44 bytes): timestamp_ms, shi, feature_distance, injected_fraction, ground_truth_xyz, observed_xyz",
        "scale_formula": "max(sample_std(clean_feature), relative_tolerance*mean_abs(clean_feature), numerical_floor)",
        "elapsed_seconds": time.monotonic() - started,
        "successful_jobs": successes,
        "skipped_jobs": skipped,
        "failed_jobs": failures,
    }
    report_stem = f"processing_report_batch_{args.batch:02d}_of_{args.total_batches:02d}"
    report_path = output / f"{report_stem}.json"
    attempt = 2
    while report_path.exists():
        report_path = output / f"{report_stem}_attempt_{attempt:02d}.json"
        attempt += 1
    write_json_atomic(report_path, report)
    print(
        f"Batch {args.batch} finished: {len(successes)} successful, {len(skipped)} skipped, "
        f"{len(failures)} failed in {report['elapsed_seconds'] / 60:.1f} minutes"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
