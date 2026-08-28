#!/usr/bin/env python3
"""Calculate baseline-referenced SHI directly from hardware-injection ZIP files.

The archives are read in place and are never extracted or modified. Results are
fixed-width binary files accompanied by JSON metadata that records the inferred
fault onset and the calibration strategy used for each experiment family.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import struct
import time
from typing import Iterator
import zipfile

import numpy as np

from analyze_simple_shi import RunningStats, extract_multiaxis, write_json_atomic
from quick_visualizer import (
    SENSOR_NAMES, V1_SCALAR, V1_VECTOR, V2_RECORD, looks_like_v2, score_v1,
)


PIPELINE_VERSION = 1
WINDOW_SIZE = 256
STRIDE = 64
SENSORS = {
    "vibration", "current", "pressure", "temperature", "microphone",
    "photodiode", "magnetometer", "gyroscope", "accelerometer",
}
HARDWARE_DTYPE = np.dtype(
    [
        ("timestamp_ms", "<u8"),
        ("shi", "<f4"),
        ("feature_distance", "<f4"),
        ("signal", "<f4", (3,)),
    ],
    align=False,
)
assert HARDWARE_DTYPE.itemsize == 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("../Data/hardware_injection"))
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("outputs/hardware_simple_shi"),
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--batch", type=int, help="one-based batch to process")
    parser.add_argument("--total-batches", type=int, default=10)
    parser.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--feature-batch-windows", type=int, default=256)
    parser.add_argument("--relative-tolerance", type=float, default=0.10)
    parser.add_argument("--include-ar", action="store_true")
    parser.add_argument(
        "--archive-contains", help="only include archives containing this text (case-insensitive)",
    )
    parser.add_argument(
        "--member-contains", help="only include ZIP members containing this text (case-insensitive)",
    )
    parser.add_argument(
        "--fault-start-seconds", type=float,
        help="override inferred fault onset for all selected jobs, relative to signal start",
    )
    args = parser.parse_args()
    if args.total_batches < 1 or args.workers < 1 or args.stride < 1:
        parser.error("total-batches, workers, and stride must be positive")
    if args.window_size < 16 or args.feature_batch_windows < 1:
        parser.error("window-size must be >=16 and feature-batch-windows must be positive")
    if args.relative_tolerance <= 0:
        parser.error("relative-tolerance must be positive")
    if args.fault_start_seconds is not None and args.fault_start_seconds < 0:
        parser.error("fault-start-seconds must be non-negative")
    if not args.plan_only and args.batch is None:
        parser.error("--batch is required unless --plan-only is used")
    if args.batch is not None and not 1 <= args.batch <= args.total_batches:
        parser.error("--batch must be between 1 and --total-batches")
    return args


def protocol_for(member: str, override: float | None = None) -> dict:
    """Return onset and baseline rules, with limitations made explicit."""
    lower = member.lower()
    if override is not None:
        return {
            "fault_offset_s": override,
            "baseline": "pre_fault" if override > 0 else "initial_proxy",
            "baseline_seconds": override if override > 0 else 60.0,
            "confidence": "manual",
            "reason": "command-line --fault-start-seconds override",
        }
    if "experiment_a1" in lower or lower.startswith("8-3-26 run/"):
        return {
            "fault_offset_s": 300.0, "baseline": "pre_fault", "baseline_seconds": 300.0,
            "confidence": "protocol", "reason": "A1 protocol: 5 min settle before shaker starts",
        }
    if "experiment_a3" in lower:
        return {
            "fault_offset_s": 300.0, "baseline": "pre_fault", "baseline_seconds": 300.0,
            "confidence": "protocol", "reason": "A3 protocol: 5 min settle before EMI phase",
        }
    if "experiment_a2" in lower:
        return {
            "fault_offset_s": 0.0, "baseline": "tail_recovery", "baseline_seconds": 300.0,
            "confidence": "protocol", "reason": "A2 recording starts under cold exposure; tail is recovery reference",
        }
    if any(name in lower for name in ("loose connector", "partial-occlusion")):
        return {
            "fault_offset_s": 300.0, "baseline": "pre_fault", "baseline_seconds": 300.0,
            "confidence": "protocol_family",
            "reason": "B intermittent-fault protocol uses the first 5 min as baseline",
        }
    if "supply-voltage-droop" in lower:
        return {
            "fault_offset_s": 0.0, "baseline": "initial_proxy", "baseline_seconds": 60.0,
            "confidence": "limited",
            "reason": "B3 voltage condition is already active when each ~5 min run starts",
        }
    if "imu bias injection" in lower:
        return {
            "fault_offset_s": 0.0, "baseline": "initial_proxy", "baseline_seconds": 60.0,
            "confidence": "limited",
            "reason": "B4 bias is active for the run; no healthy segment exists in the same recording",
        }
    if "accelerated aging" in lower:
        return {
            "fault_offset_s": 0.0, "baseline": "initial_proxy", "baseline_seconds": 60.0,
            "confidence": "limited",
            "reason": "aged state exists at recording start; initial segment is only a stability proxy",
        }
    return {
        "fault_offset_s": 300.0, "baseline": "pre_fault", "baseline_seconds": 300.0,
        "confidence": "assumed", "reason": "unknown family; default 5 min baseline assumption",
    }


def member_sensor(member: str) -> str | None:
    path = PurePosixPath(member)
    if path.suffix.lower() != ".csv":
        return None
    sensor = path.stem.lower().removeprefix("proc_")
    return sensor if sensor in SENSORS else None


def valid_member(member: str) -> bool:
    parts = PurePosixPath(member).parts
    lowered = [part.lower() for part in parts]
    return (
        member_sensor(member) is not None
        and "converted" in lowered
        and not any(part.startswith(".") or part == "__macosx" for part in lowered)
        and ".trashes" not in lowered
    )


def discover_jobs(input_dir: Path, archive_filter: str | None, member_filter: str | None,
                  onset_override: float | None) -> list[dict]:
    jobs: list[dict] = []
    archive_filter = archive_filter.lower() if archive_filter else None
    member_filter = member_filter.lower() if member_filter else None
    for archive in sorted(input_dir.glob("*.zip")):
        if archive_filter and archive_filter not in archive.name.lower():
            continue
        with zipfile.ZipFile(archive) as bundle:
            infos = bundle.infolist()
            converted = []
            for info in infos:
                member = info.filename
                if info.is_dir() or not valid_member(member):
                    continue
                if member_filter and member_filter not in member.lower():
                    continue
                with bundle.open(info) as stream:
                    header = stream.readline().decode("utf-8-sig", "replace").strip()
                if header not in {"timestamp_ms,value", "timestamp_ms,x,y,z"}:
                    continue
                converted.append((info, header))
            for info, header in converted:
                member = info.filename
                identity = f"{archive.name}\0{member}"
                jobs.append({
                    "job_id": hashlib.sha256(identity.encode()).hexdigest()[:16],
                    "archive": archive.name,
                    "member": member,
                    "sensor": member_sensor(member),
                    "axes": 3 if header.endswith("x,y,z") else 1,
                    "source_format": "csv",
                    "uncompressed_bytes": info.file_size,
                    "crc32": f"{info.CRC:08x}",
                    "protocol": protocol_for(member, onset_override),
                })
            # The small "26 run" archive contains only raw SHIELD binaries.
            # Use it only when an archive has no converted sensor CSVs, avoiding
            # duplicate raw/converted processing in the larger experiment ZIPs.
            if not converted:
                tier_sensors = {
                    "fast_data.bin": ("vibration", "microphone", "magnetometer", "gyroscope", "accelerometer"),
                    "medium_data.bin": ("current", "photodiode"),
                    "slow_data.bin": ("pressure", "temperature"),
                }
                for info in infos:
                    member = info.filename
                    path = PurePosixPath(member)
                    sensors = tier_sensors.get(path.name.lower())
                    lowered = [part.lower() for part in path.parts]
                    if (info.is_dir() or info.file_size == 0 or sensors is None
                            or any(part.startswith(".") for part in lowered)
                            or (member_filter and member_filter not in member.lower())):
                        continue
                    for sensor in sensors:
                        identity = f"{archive.name}\0{member}\0{sensor}"
                        jobs.append({
                            "job_id": hashlib.sha256(identity.encode()).hexdigest()[:16],
                            "archive": archive.name, "member": member, "sensor": sensor,
                            "axes": 3 if sensor in {"magnetometer", "gyroscope", "accelerometer"} else 1,
                            "source_format": "bin", "uncompressed_bytes": info.file_size,
                            "crc32": f"{info.CRC:08x}", "protocol": protocol_for(member, onset_override),
                        })
    return jobs


def fingerprint(input_dir: Path, jobs: list[dict]) -> str:
    digest = hashlib.sha256()
    archives = sorted({job["archive"] for job in jobs})
    for name in archives:
        stat = (input_dir / name).stat()
        digest.update(f"{name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
    for job in jobs:
        digest.update(f"{job['archive']}\0{job['member']}\0{job['crc32']}\n".encode())
    return digest.hexdigest()


def make_plan(input_dir: Path, jobs: list[dict], total_batches: int) -> dict:
    batches = [{"batch": i + 1, "bytes": 0, "jobs": []} for i in range(total_batches)]
    for job in sorted(jobs, key=lambda item: (-item["uncompressed_bytes"], item["member"])):
        target = min(batches, key=lambda item: (item["bytes"], item["batch"]))
        target["jobs"].append(job)
        target["bytes"] += job["uncompressed_bytes"]
    for batch in batches:
        batch["jobs"].sort(key=lambda item: (item["archive"], item["member"]))
    return {
        "pipeline_version": PIPELINE_VERSION,
        "dataset_fingerprint": fingerprint(input_dir, jobs),
        "total_batches": total_batches,
        "total_jobs": len(jobs),
        "total_uncompressed_bytes": sum(job["uncompressed_bytes"] for job in jobs),
        "batches": batches,
    }


def ensure_plan(input_dir: Path, output_dir: Path, args: argparse.Namespace) -> dict:
    jobs = discover_jobs(
        input_dir, args.archive_contains, args.member_contains, args.fault_start_seconds,
    )
    if not jobs:
        raise ValueError("no supported converted sensor CSV members matched the selection")
    expected = make_plan(input_dir, jobs, args.total_batches)
    plan_path = output_dir / "hardware_simple_shi_plan.json"
    if plan_path.exists():
        existing = json.loads(plan_path.read_text(encoding="utf-8"))
        if existing != expected:
            raise ValueError(f"existing plan differs: {plan_path}; use a new output directory")
        return existing
    write_json_atomic(plan_path, expected)
    return expected


def detect_member_layout(sample: bytes, size: int, name: str) -> struct.Struct:
    if looks_like_v2(sample):
        return V2_RECORD
    if name.lower() == "fast_data.bin":
        candidates = [layout for layout in (V1_SCALAR, V1_VECTOR) if size % layout.size == 0]
        if candidates:
            return max(candidates, key=lambda layout: score_v1(sample, layout))
    if size % V1_SCALAR.size == 0:
        return V1_SCALAR
    raise ValueError(f"unsupported binary layout for {name}")


def iter_records(archive: Path, job: dict) -> Iterator[tuple[int, np.ndarray]]:
    member = job["member"]
    with zipfile.ZipFile(archive) as bundle, bundle.open(member) as raw:
        if job["source_format"] == "bin":
            info = bundle.getinfo(member)
            sample = raw.read(V2_RECORD.size * 30)
            layout = detect_member_layout(sample, info.file_size, PurePosixPath(member).name)
            sensor_id = next(key for key, value in SENSOR_NAMES.items() if value == job["sensor"])
            data = sample
            while data:
                usable = len(data) - (len(data) % layout.size)
                for offset in range(0, usable, layout.size):
                    values = layout.unpack_from(data, offset)
                    if values[1] != sensor_id:
                        continue
                    if layout is V2_RECORD:
                        timestamp, _sid, _kind, axes, _flags, x, y, z = values
                        yield timestamp, np.asarray((x, y, z)[:axes], dtype=np.float64)
                    elif layout is V1_VECTOR:
                        timestamp, _sid, x, y, z = values
                        axes = 3 if sensor_id in {7, 8, 9} else 1
                        yield timestamp, np.asarray((x, y, z)[:axes], dtype=np.float64)
                    else:
                        timestamp, _sid, value = values
                        yield timestamp, np.asarray((value,), dtype=np.float64)
                remainder = data[usable:]
                chunk = raw.read(1024 * 1024)
                data = remainder + chunk
            return
        with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
            reader = csv.DictReader(text)
            fields = reader.fieldnames or []
            axes = ("value",) if fields == ["timestamp_ms", "value"] else ("x", "y", "z")
            if fields not in (["timestamp_ms", "value"], ["timestamp_ms", "x", "y", "z"]):
                raise ValueError(f"unsupported columns {fields} in {member}")
            for line, row in enumerate(reader, start=2):
                try:
                    yield int(row["timestamp_ms"]), np.asarray([float(row[a]) for a in axes])
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError(f"invalid row {line} in {member}: {exc}") from exc


def timestamp_bounds(archive: Path, job: dict) -> tuple[int, int, int]:
    first = last = None
    count = 0
    for timestamp, _values in iter_records(archive, job):
        first = timestamp if first is None else first
        if last is not None and timestamp < last:
            raise ValueError(f"timestamps decrease in {job['member']}: {last} -> {timestamp}")
        last = timestamp
        count += 1
    if first is None or last is None:
        raise ValueError(f"empty sensor stream: {job['member']} ({job['sensor']})")
    return first, last, count


def iter_window_batches(archive: Path, job: dict, window_size: int, stride: int,
                        batch_windows: int) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, float]]:
    values: deque[np.ndarray] = deque(maxlen=window_size)
    timestamps: deque[int] = deque(maxlen=window_size)
    pending_windows: list[np.ndarray] = []
    pending_values: list[np.ndarray] = []
    pending_timestamps: list[int] = []
    count = 0
    fs: float | None = None
    for timestamp, value in iter_records(archive, job):
        values.append(value)
        timestamps.append(timestamp)
        count += 1
        if len(values) < window_size or (count - window_size) % stride:
            continue
        if fs is None:
            differences = np.diff(np.asarray(timestamps, dtype=np.float64))
            differences = differences[differences > 0]
            fs = 1000.0 / float(np.median(differences)) if len(differences) else 1.0
        pending_windows.append(np.asarray(values).copy())
        pending_values.append(value.copy())
        pending_timestamps.append(timestamp)
        if len(pending_windows) >= batch_windows:
            yield (np.stack(pending_windows), np.stack(pending_values),
                   np.asarray(pending_timestamps, dtype=np.uint64), fs)
            pending_windows.clear(); pending_values.clear(); pending_timestamps.clear()
    if pending_windows:
        assert fs is not None
        yield (np.stack(pending_windows), np.stack(pending_values),
               np.asarray(pending_timestamps, dtype=np.uint64), fs)


def output_paths(output_dir: Path, job: dict) -> tuple[Path, Path]:
    archive_stem = Path(job["archive"]).stem
    base = output_dir / archive_stem / f"{job['job_id']}__{job['sensor']}"
    return base.with_suffix(".bin"), base.with_suffix(".json")


def process_job(job: dict, config: dict) -> dict:
    started = time.monotonic()
    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])
    archive = input_dir / job["archive"]
    final, marker = output_paths(output_dir, job)
    if marker.is_file():
        old = json.loads(marker.read_text(encoding="utf-8"))
        if old.get("status") == "success" and final.is_file() and final.stat().st_size == old.get("output_bytes"):
            return {"status": "skipped", "job": job, "report": old}
    if final.exists() or marker.exists() or final.with_suffix(".bin.tmp").exists():
        raise FileExistsError(f"incomplete existing hardware SHI output for {job['job_id']}")

    first, last, records = timestamp_bounds(archive, job)
    protocol = job["protocol"]
    fault_start = first + round(protocol["fault_offset_s"] * 1000)
    baseline_ms = round(protocol["baseline_seconds"] * 1000)
    if protocol["baseline"] == "tail_recovery":
        calibration_start, calibration_end = max(first, last - baseline_ms), last + 1
    elif protocol["baseline"] == "pre_fault":
        calibration_start, calibration_end = first, min(last + 1, fault_start)
    else:
        calibration_start, calibration_end = first, min(last + 1, first + baseline_ms)

    stats = RunningStats()
    for windows, _ends, timestamps, fs in iter_window_batches(
        archive, job, config["window_size"], config["stride"], config["feature_batch_windows"],
    ):
        selected = (timestamps >= calibration_start) & (timestamps < calibration_end)
        if np.any(selected):
            stats.update(extract_multiaxis(windows[selected], fs, config["include_ar"]))
    if stats.count < 2 or stats.mean is None:
        raise ValueError(f"insufficient calibration windows in {job['member']}")
    scale = stats.scale(config["relative_tolerance"])

    final.parent.mkdir(parents=True, exist_ok=True)
    temporary = final.with_suffix(".bin.tmp")
    windows_written = 0
    shi_sum = 0.0
    shi_min = 100.0
    try:
        with temporary.open("xb") as stream:
            for windows, ends, timestamps, fs in iter_window_batches(
                archive, job, config["window_size"], config["stride"],
                config["feature_batch_windows"],
            ):
                features = extract_multiaxis(windows, fs, config["include_ar"])
                normalized = np.minimum(np.abs(features - stats.mean) / scale, 10.0)
                distance = np.sqrt(np.mean(normalized ** 2, axis=1))
                shi = 100.0 * np.exp(-distance)
                output = np.zeros(len(shi), dtype=HARDWARE_DTYPE)
                output["timestamp_ms"] = timestamps
                output["shi"] = shi
                output["feature_distance"] = distance
                output["signal"][:, : job["axes"]] = ends
                stream.write(output.tobytes())
                windows_written += len(output)
                shi_sum += float(np.sum(shi))
                shi_min = min(shi_min, float(np.min(shi)))
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, final)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    archive_stat = archive.stat()
    report = {
        "status": "success",
        "pipeline_version": PIPELINE_VERSION,
        "job": job,
        "input_signature": {
            "archive_bytes": archive_stat.st_size, "archive_mtime_ns": archive_stat.st_mtime_ns,
            "member_crc32": job["crc32"], "member_bytes": job["uncompressed_bytes"],
        },
        "processing_config": config["processing_config"],
        "first_timestamp_ms": first,
        "last_timestamp_ms": last,
        "fault_start_ms": fault_start,
        "fault_start_seconds_from_signal": protocol["fault_offset_s"],
        "fault_timestamp_source": protocol["reason"],
        "fault_timestamp_confidence": protocol["confidence"],
        "baseline_strategy": protocol["baseline"],
        "calibration_start_ms": calibration_start,
        "calibration_end_ms": calibration_end,
        "input_records": records,
        "output_windows": windows_written,
        "output_path": str(final.resolve()),
        "output_bytes": final.stat().st_size,
        "record_bytes": HARDWARE_DTYPE.itemsize,
        "shi_mean": shi_sum / windows_written if windows_written else None,
        "shi_min": shi_min if windows_written else None,
        "elapsed_seconds": time.monotonic() - started,
    }
    write_json_atomic(marker, report)
    return {"status": "success", "job": job, "report": report}


def next_report_path(output: Path, batch: int, total: int) -> Path:
    base = output / f"hardware_processing_report_batch_{batch:02d}_of_{total:02d}.json"
    if not base.exists():
        return base
    attempt = 2
    while True:
        candidate = base.with_name(f"{base.stem}_attempt_{attempt:02d}.json")
        if not candidate.exists():
            return candidate
        attempt += 1


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")
    if output_dir == input_dir or input_dir in output_dir.parents:
        raise ValueError("output must be outside the input archive directory")
    plan = ensure_plan(input_dir, output_dir, args)
    print(f"Plan: {plan['total_jobs']} sensor files, {plan['total_uncompressed_bytes']:,} bytes, "
          f"{plan['total_batches']} batches")
    if args.plan_only:
        return 0

    selected = plan["batches"][args.batch - 1]
    processing_config = {
        "window_size": args.window_size, "stride": args.stride,
        "feature_batch_windows": args.feature_batch_windows,
        "relative_tolerance": args.relative_tolerance, "include_ar": args.include_ar,
        "binary_record_bytes": HARDWARE_DTYPE.itemsize,
    }
    config = {
        "input_dir": str(input_dir), "output_dir": str(output_dir),
        **processing_config, "processing_config": processing_config,
    }
    successes, skipped, failures = [], [], []
    started = time.monotonic()
    def accept(job: dict, result: dict | None = None, error: Exception | None = None) -> None:
        if error is not None:
            failure = {"job": job, "error_type": type(error).__name__, "message": str(error)}
            failures.append(failure)
            print(f"[failed] {job['member']}: {type(error).__name__}: {error}", flush=True)
            return
        assert result is not None
        (skipped if result["status"] == "skipped" else successes).append(result)
        print(f"[{result['status']}] {job['member']} [{job['sensor']}]", flush=True)

    if args.workers == 1:
        for job in selected["jobs"]:
            try:
                accept(job, result=process_job(job, config))
            except Exception as exc:
                accept(job, error=exc)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            pending = {executor.submit(process_job, job, config): job for job in selected["jobs"]}
            for future in as_completed(pending):
                job = pending[future]
                try:
                    accept(job, result=future.result())
                except Exception as exc:
                    accept(job, error=exc)
    report = {
        "pipeline_version": PIPELINE_VERSION, "batch": args.batch,
        "total_batches": args.total_batches, "workers": args.workers,
        "elapsed_seconds": time.monotonic() - started,
        "successful_jobs": sorted(successes, key=lambda x: x["job"]["member"]),
        "skipped_jobs": sorted(skipped, key=lambda x: x["job"]["member"]),
        "failed_jobs": sorted(failures, key=lambda x: x["job"]["member"]),
    }
    report_path = next_report_path(output_dir, args.batch, args.total_batches)
    write_json_atomic(report_path, report)
    print(f"Batch {args.batch}: {len(successes)} succeeded, {len(skipped)} skipped, "
          f"{len(failures)} failed. Report: {report_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
