import numpy as np
import os
import gc
import glob
import pandas as pd
from pathlib import Path
from multiprocessing import Pool
from sklearn.preprocessing import StandardScaler
from features import build_feature_pipeline, get_pipeline_feature_names
from fault_injection import load_signal, inject_fault_multiaxis, FAULT_CONFIGS

WINDOW_SIZE = 256  # samples
STRIDE = 64  # samples
OUTPUT_DIR = "processed_windows"


def make_windows(sig_1d: np.ndarray) -> np.ndarray:
    """Zero-copy strided window view. Returns (n_windows, WINDOW_SIZE)."""
    from numpy.lib.stride_tricks import sliding_window_view

    return sliding_window_view(sig_1d, WINDOW_SIZE)[::STRIDE]


def process_entry(args):
    i, path, sensor_id, fault_type, severity, onset, pre_fault_samples, output_dir = args

    df = pd.read_csv(path)
    signal, fs = load_signal(df)
    del df

    n = len(signal)
    is_imu = signal.ndim == 2
    axes = [0, 1, 2] if is_imu else None

    if fault_type == "healthy":
        labels = np.zeros(n, dtype=int)
        corrupted = signal
    else:
        corrupted, labels = inject_fault_multiaxis(
            signal,
            fault_type,
            onset_idx=onset,
            severity=severity,
            pre_fault_samples=pre_fault_samples,
            axes=axes,
        )
    del signal

    feature_pipeline = build_feature_pipeline(fs)

    if is_imu:
        axes_features = []
        for ax in range(3):
            windows = make_windows(corrupted[:, ax])
            axes_features.append(feature_pipeline.transform(windows))
        X_windows = np.concatenate(axes_features, axis=1)
        del axes_features
    else:
        windows = make_windows(corrupted)
        X_windows = feature_pipeline.transform(windows)
    del corrupted

    y_windows = np.array(
        [
            labels[start + WINDOW_SIZE - 1]
            for start in range(0, len(labels) - WINDOW_SIZE + 1, STRIDE)
        ]
    )

    n_windows = len(X_windows)
    tag = "imu" if is_imu else "scalar"
    np.save(os.path.join(output_dir, f"X_{tag}_{i:04d}.npy"), X_windows.astype(np.float32))
    np.save(os.path.join(output_dir, f"y_{tag}_{i:04d}.npy"), y_windows.astype(np.int8))
    np.save(os.path.join(output_dir, f"sensor_{tag}_{i:04d}.npy"), np.array([sensor_id] * n_windows))
    print(f"Processed entry {i + 1} (sensor={sensor_id}, fault={fault_type})")


def build_job_list(csv_paths, prediction_horizon_sec=20.0, output_dir=OUTPUT_DIR):
    """
    Reads only the timestamp column of each CSV to compute fs and n,
    then returns a flat list of job tuples — no signal data loaded.
    """
    jobs = []
    i = 0

    for path in csv_paths:
        sensor_id = Path(path).stem

        df_head = pd.read_csv(path, usecols=[0])
        timestamps = df_head.iloc[:, 0].values
        n = len(timestamps)
        fs = 1000.0 / np.median(np.diff(timestamps))
        del df_head

        pre_fault_samples = int(prediction_horizon_sec * fs)
        min_onset = pre_fault_samples + int(fs * 5)

        if n <= min_onset + int(fs * 10):
            print(f"Skipping {path} — too short")
            continue

        jobs.append((i, path, sensor_id, "healthy", 0.0, None, pre_fault_samples, output_dir))
        i += 1

        for fault_type, (lo, hi) in FAULT_CONFIGS:
            sev = np.random.uniform(lo, hi)
            onset = np.random.randint(min_onset, n - int(fs * 10))
            jobs.append(
                (i, path, sensor_id, fault_type, sev, onset, pre_fault_samples, output_dir)
            )
            i += 1

    return jobs


def run_pipeline(
    csv_files,
    prediction_horizon_sec=20.0,
    n_workers=1,
    sensor_filter: str | None = None,
):
    """
    Process CSV files into windowed feature matrices.

    sensor_filter: if set, only process the CSV whose stem matches this name,
                   and write outputs to a sensor-namespaced subdirectory so
                   runs for different sensors don't collide.

    Returns
    -------
    X_imu, X_scalar         : scaled feature matrices (or None)
    y_imu, y_scalar         : label arrays
    scaler_imu, scaler_scalar
    sensor_imu, sensor_scalar: string arrays — sensor name for each row
    feature_names_imu       : ordered column names for X_imu
    feature_names_scalar    : ordered column names for X_scalar
    """
    output_dir = os.path.join(OUTPUT_DIR, sensor_filter) if sensor_filter else OUTPUT_DIR

    if sensor_filter is not None:
        csv_files = [f for f in csv_files if Path(f).stem == sensor_filter]
        if not csv_files:
            raise ValueError(f"No CSV files match sensor_filter='{sensor_filter}'")
        print(f"Filtered to {len(csv_files)} file(s) for sensor '{sensor_filter}'")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Build job list (cheap — only reads timestamps)
    jobs = build_job_list(csv_files, prediction_horizon_sec, output_dir)
    print(f"Total jobs: {len(jobs)}")

    # 2. Process in parallel
    with Pool(processes=n_workers) as pool:
        pool.map(process_entry, jobs, chunksize=1)

    # 3. Load all chunks and stack — IMU and scalar separately (different feature widths)
    X_imu_files     = sorted(glob.glob(os.path.join(output_dir, "X_imu_*.npy")))
    y_imu_files     = sorted(glob.glob(os.path.join(output_dir, "y_imu_*.npy")))
    sensor_imu_files = sorted(glob.glob(os.path.join(output_dir, "sensor_imu_*.npy")))
    X_scalar_files  = sorted(glob.glob(os.path.join(output_dir, "X_scalar_*.npy")))
    y_scalar_files  = sorted(glob.glob(os.path.join(output_dir, "y_scalar_*.npy")))
    sensor_scalar_files = sorted(glob.glob(os.path.join(output_dir, "sensor_scalar_*.npy")))

    X_imu      = np.vstack([np.load(f) for f in X_imu_files])          if X_imu_files      else None
    y_imu      = np.concatenate([np.load(f) for f in y_imu_files])     if y_imu_files      else None
    sensor_imu = np.concatenate([np.load(f) for f in sensor_imu_files]) if sensor_imu_files else None
    X_scalar      = np.vstack([np.load(f) for f in X_scalar_files])         if X_scalar_files      else None
    y_scalar      = np.concatenate([np.load(f) for f in y_scalar_files])    if y_scalar_files      else None
    sensor_scalar = np.concatenate([np.load(f) for f in sensor_scalar_files]) if sensor_scalar_files else None

    # 4. Scale each stream independently
    scaler_imu, scaler_scalar = StandardScaler(), StandardScaler()
    if X_imu is not None:
        X_imu = scaler_imu.fit_transform(X_imu)
        print(f"IMU:    X={X_imu.shape}  y={y_imu.shape}  sensors={np.unique(sensor_imu)}  labels={np.unique(y_imu, return_counts=True)}")
    if X_scalar is not None:
        X_scalar = scaler_scalar.fit_transform(X_scalar)
        print(f"Scalar: X={X_scalar.shape}  y={y_scalar.shape}  sensors={np.unique(sensor_scalar)}  labels={np.unique(y_scalar, return_counts=True)}")

    feature_names_imu    = get_pipeline_feature_names(is_imu=True)
    feature_names_scalar = get_pipeline_feature_names(is_imu=False)

    return (
        X_imu, X_scalar,
        y_imu, y_scalar,
        scaler_imu, scaler_scalar,
        sensor_imu, sensor_scalar,
        feature_names_imu, feature_names_scalar,
    )


USAGE = "usage: python pipeline.py <data_dir> [--prediction-horizon FLOAT] [--n-workers INT] [--sensor SENSOR_ID]"

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=str, help="Path to directory containing CSV files")
    parser.add_argument("--prediction-horizon", type=float, default=20.0)
    parser.add_argument("--n-workers", type=int, default=os.cpu_count() // 2)
    parser.add_argument("--sensor", type=str, default=None, help="CSV stem to process (single-sensor mode)")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise ValueError(f"data_dir '{args.data_dir}' is not a valid directory\n{USAGE}")

    csv_files = glob.glob(os.path.join(args.data_dir, "**/*.csv"), recursive=True)
    if not csv_files:
        raise ValueError(f"No CSV files found in '{args.data_dir}'\n{USAGE}")

    run_pipeline(csv_files, args.prediction_horizon, args.n_workers, sensor_filter=args.sensor)
