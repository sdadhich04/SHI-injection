import numpy as np
import gc
import pandas as pd
from dataclasses import dataclass
from typing import Literal

FaultType = Literal[
    "bias_offset",
    "drift",
    "noise_increase",
    "dropout",
    "stuck_at",
    "latency",
    "saturation",
    "calibration_shift",
]


@dataclass
class FaultEvent:
    fault_type: FaultType
    onset_idx: int
    severity: float
    label: int


FAULT_CONFIGS: list[tuple[str, tuple[float, float]]] = [
    ("bias_offset",        (0.5, 3.0)),   # DC shift; 0.5 = subtle, 3.0 = obvious
    ("drift",              (0.05, 0.5)),  # max ramp value over the fault window
    # ("noise_increase",   (0.01, 0.3)),  # added noise std — keep below signal std
    # ("dropout",          (5, 100)),     # zero-out period in samples
    # ("latency",          (5, 50)),      # delay in samples
    # ("saturation",       (0.5, 0.95)), # clip percentile multiplier; 0.5 = aggressive
    # ("calibration_shift",(0.8, 1.5)),  # multiplicative scale; <1 shrinks, >1 amplifies
]


def load_signal(data: pd.DataFrame) -> tuple[np.ndarray, float]:
    """
    Accepts either:
        - Two-column DataFrame: [timestamp_ms, value]
        - Four-column DataFrame: [timestamp_ms, x, y, z]
    Returns (signal, fs) where signal is shape (n_samples,) or (n_samples, 3).
    fs is estimated from the median sample interval.
    """
    timestamps_ms = data.iloc[:, 0].values
    fs = 1000.0 / np.median(np.diff(timestamps_ms))

    if data.shape[1] == 2:
        signal = data.iloc[:, 1].values
    elif data.shape[1] == 4:
        signal = data.iloc[:, 1:4].values
    else:
        raise ValueError(f"Expected 2 or 4 columns, got {data.shape[1]}")

    return signal, fs


def inject_fault(
    signal: np.ndarray,
    fault_type: FaultType,
    onset_idx: int,
    severity: float,
    pre_fault_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (corrupted_signal, sample_labels).
    Labels: 0=healthy, 1=pre-fault warning zone, 2=active fault.
    """
    sig = signal.copy().astype(float)
    labels = np.zeros(len(sig), dtype=int)

    warn_start = max(0, onset_idx - pre_fault_samples)
    labels[warn_start:onset_idx] = 1
    labels[onset_idx:] = 2

    n = len(sig)
    fault_len = n - onset_idx

    if fault_type == "bias_offset":
        sig[onset_idx:] += severity

    elif fault_type == "drift":
        ramp = np.linspace(0, severity, fault_len)
        sig[onset_idx:] += ramp

    elif fault_type == "noise_increase":
        added_noise = np.random.normal(0, severity, fault_len)
        sig[onset_idx:] += added_noise

    elif fault_type == "dropout":
        period = max(1, int(severity))
        for i in range(onset_idx, n, period):
            sig[i : i + max(1, period // 4)] = 0.0

    elif fault_type == "stuck_at":
        stuck_value = sig[onset_idx - 1]
        sig[onset_idx:] = stuck_value

    elif fault_type == "latency":
        delay = int(severity)
        if onset_idx + delay < n:
            sig[onset_idx + delay :] = sig[onset_idx : n - delay]
            sig[onset_idx : onset_idx + delay] = sig[onset_idx - 1]

    elif fault_type == "saturation":
        clip_val = np.percentile(np.abs(sig[:onset_idx]), 95) * severity
        sig[onset_idx:] = np.clip(sig[onset_idx:], -clip_val, clip_val)

    elif fault_type == "calibration_shift":
        sig[onset_idx:] *= severity

    return sig, labels


def inject_fault_multiaxis(
    signal: np.ndarray,
    fault_type: FaultType,
    onset_idx: int,
    severity: float,
    pre_fault_samples: int = 0,
    axes: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Wraps inject_fault to handle both 1D (n_samples,) and 2D (n_samples, 3) signals.
    """
    if signal.ndim == 1:
        return inject_fault(signal, fault_type, onset_idx, severity, pre_fault_samples)

    corrupted = signal.copy()
    target_axes = axes if axes is not None else list(range(signal.shape[1]))

    labels = np.zeros(len(signal), dtype=int)
    warn_start = max(0, onset_idx - pre_fault_samples)
    labels[warn_start:onset_idx] = 1
    labels[onset_idx:] = 2

    for ax in target_axes:
        corrupted[:, ax], _ = inject_fault(
            signal[:, ax], fault_type, onset_idx, severity, pre_fault_samples
        )

    return corrupted, labels


def generate_faulty_dataset(csv_paths: list[str], prediction_horizon_sec: float = 20.0):
    """
    csv_paths: list of paths to CSV files, each either
               [timestamp_ms, value] or [timestamp_ms, x, y, z]
    Yields one entry at a time to avoid holding the full dataset in memory.
    """
    fault_configs = FAULT_CONFIGS

    for path in csv_paths:
        # Load one CSV at a time, free immediately after parsing
        df = pd.read_csv(path)
        signal, fs = load_signal(df)
        del df
        gc.collect()

        n = len(signal)
        pre_fault_samples = int(prediction_horizon_sec * fs)
        min_onset = pre_fault_samples + int(fs * 5)

        if n <= min_onset + int(fs * 10):
            print(
                f"Warning: recording too short ({n} samples at {fs:.1f} Hz), skipping."
            )
            del signal
            continue

        is_imu = signal.ndim == 2

        # Yield healthy version first
        yield {
            "signal": signal,
            "labels": np.zeros(n, dtype=int),
            "fault_type": "healthy",
            "severity": 0.0,
            "fs": fs,
            "is_imu": is_imu,
        }

        # Yield each fault variant one at a time
        for fault_type, (lo, hi) in fault_configs:
            sev = np.random.uniform(lo, hi)
            onset = np.random.randint(min_onset, n - int(fs * 10))
            axes = [0, 1, 2] if is_imu else None

            corrupted, labels = inject_fault_multiaxis(
                signal,
                fault_type,
                onset_idx=onset,
                severity=sev,
                pre_fault_samples=pre_fault_samples,
                axes=axes,
            )

            yield {
                "signal": corrupted,
                "labels": labels,
                "fault_type": fault_type,
                "severity": sev,
                "fs": fs,
                "is_imu": is_imu,
                "axes_injected": axes,
            }

            del corrupted, labels
            gc.collect()

        del signal
        gc.collect()
