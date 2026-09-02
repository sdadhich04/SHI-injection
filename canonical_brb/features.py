"""NumPy implementation of the canonical SHIELD feature extractor.

The formulas and names follow GilliamWong/SHIELD-Sensor-Modality at commit
222dbb1b0d742c7e8ea9b719c841ac5b1d2f2a72.  This module deliberately avoids a
DataFrame dependency so long recordings can be processed as bounded batches.
"""

from __future__ import annotations

from collections.abc import Iterable
import math

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew


UPSTREAM_COMMIT = "222dbb1b0d742c7e8ea9b719c841ac5b1d2f2a72"
EPS = 1e-12
SYM4_SCALING = np.asarray([
    -0.07576571478927333, -0.02963552764599851,
    0.49761866763201545, 0.80373875180591614,
    0.29785779560527736, -0.09921954357684722,
    -0.01260396726203783, 0.03222310060404270,
], dtype=np.float64)
SYM4_WAVELET = np.asarray(
    [((-1) ** n) * SYM4_SCALING[len(SYM4_SCALING) - 1 - n]
     for n in range(len(SYM4_SCALING))], dtype=np.float64,
)


def _finite(values: np.ndarray) -> np.ndarray:
    """Replace non-finite samples by linear interpolation within a window."""
    values = np.asarray(values, dtype=np.float64).reshape(-1).copy()
    valid = np.isfinite(values)
    if valid.all():
        return values
    if not valid.any():
        return np.zeros_like(values)
    indices = np.arange(values.size)
    values[~valid] = np.interp(indices[~valid], indices[valid], values[valid])
    return values


def _modwt_fast(values: np.ndarray, level: int = 5) -> dict[str, np.ndarray | int]:
    values = _finite(values)
    n = values.size
    if n < 2:
        return {"levels": 0}
    max_level = max(1, int(np.floor(np.log2(n / (len(SYM4_SCALING) - 1)))))
    levels = min(level, max_level)
    low = SYM4_SCALING / np.sqrt(2.0)
    high = SYM4_WAVELET / np.sqrt(2.0)
    result: dict[str, np.ndarray | int] = {}
    approximation = values.copy()
    for current in range(1, levels + 1):
        stride = 2 ** (current - 1)
        high_filter = np.zeros(n)
        low_filter = np.zeros(n)
        for index in range(len(low)):
            position = (index * stride) % n
            high_filter[position] = high[index]
            low_filter[position] = low[index]
        spectrum = np.fft.fft(approximation)
        result[f"D{current}"] = np.real(np.fft.ifft(spectrum * np.fft.fft(high_filter)))
        approximation = np.real(np.fft.ifft(spectrum * np.fft.fft(low_filter)))
    result[f"A{levels}"] = approximation
    result["levels"] = levels
    return result


def _entropy_from_energy(coefficients: np.ndarray) -> float:
    energy = coefficients ** 2
    probabilities = np.clip(energy / (float(np.sum(energy)) + EPS), EPS, None)
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _wavelet_features(values: np.ndarray, level: int) -> dict[str, float]:
    decomposition = _modwt_fast(values, level)
    levels = int(decomposition["levels"])
    if not levels:
        return {}
    output: dict[str, float] = {"modwt_levels": float(levels)}
    energies: list[float] = []
    for current in range(1, levels + 1):
        coefficients = np.asarray(decomposition[f"D{current}"])
        energy = float(np.sum(coefficients ** 2))
        std = float(np.std(coefficients))
        centered = coefficients - float(np.mean(coefficients))
        wavelet_kurtosis = 0.0 if std < EPS else float(np.mean(centered ** 4) / std ** 4 - 3.0)
        output.update({
            f"modwt_d{current}_energy": energy,
            f"modwt_d{current}_variance": float(np.var(coefficients)),
            f"modwt_d{current}_entropy": _entropy_from_energy(coefficients),
            f"modwt_d{current}_rms": float(np.sqrt(np.mean(coefficients ** 2))),
            f"modwt_d{current}_kurtosis": wavelet_kurtosis,
        })
        energies.append(energy)
    approximation = np.asarray(decomposition[f"A{levels}"])
    approximation_std = float(np.std(approximation))
    approximation_centered = approximation - float(np.mean(approximation))
    approximation_kurtosis = (
        0.0 if approximation_std < EPS
        else float(np.mean(approximation_centered ** 4) / approximation_std ** 4 - 3.0)
    )
    approximation_energy = float(np.sum(approximation ** 2))
    output.update({
        "modwt_a_energy": approximation_energy,
        "modwt_a_variance": float(np.var(approximation)),
        "modwt_a_entropy": _entropy_from_energy(approximation),
        "modwt_a_rms": float(np.sqrt(np.mean(approximation ** 2))),
        "modwt_a_kurtosis": approximation_kurtosis,
    })
    total = float(sum(energies) + approximation_energy)
    output["modwt_total_energy"] = total
    for current, energy in enumerate(energies, start=1):
        output[f"modwt_d{current}_rel_energy"] = energy / (total + EPS)
    output["modwt_a_rel_energy"] = approximation_energy / (total + EPS)
    output["modwt_energy_ratio_hf_lf"] = sum(energies[:2]) / (approximation_energy + EPS)
    output["modwt_max_energy_level"] = float(np.argmax(energies) + 1)
    return output


def _longest_repeat_fraction(values: np.ndarray) -> float:
    longest = current = 1
    for index in range(1, values.size):
        if values[index] == values[index - 1]:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return float(longest / values.size) if longest >= 2 else 0.0


def _signal_quality(values: np.ndarray) -> dict[str, float]:
    differences = np.diff(values)
    difference_median = float(np.median(differences))
    noise_floor = float(np.median(np.abs(differences - difference_median)) * 1.4826 / np.sqrt(2.0))
    segments = np.array_split(values[: (values.size // 4) * 4], 4)
    baseline_stability = float(np.std([np.mean(segment) for segment in segments]))
    dropout_samples = run = 0
    for index in range(1, values.size):
        if values[index] == values[index - 1]:
            run += 1
        else:
            if run + 1 >= 3:
                dropout_samples += run + 1
            run = 0
    if run + 1 >= 3:
        dropout_samples += run + 1
    segment_size = max(values.size // 8, 4)
    step = max(segment_size // 2, 1)
    peak_to_peak = np.asarray([
        np.ptp(values[start:start + segment_size])
        for start in range(0, values.size - segment_size + 1, step)
    ])
    peak_consistency = (
        0.0 if peak_to_peak.size < 2 or np.mean(peak_to_peak) < EPS
        else float(np.std(peak_to_peak) / np.mean(peak_to_peak))
    )
    signal_variance = float(np.var(values))
    noise_variance = float(np.var(differences) / 2.0)
    snr = 60.0 if noise_variance < 1e-20 else (
        -60.0 if signal_variance < 1e-20
        else float(10.0 * np.log10(signal_variance / noise_variance))
    )
    time = np.arange(values.size, dtype=np.float64)
    centered_time = time - np.mean(time)
    centered_signal = values - np.mean(values)
    denominator = float(np.sum(centered_time ** 2))
    prediction = np.mean(values) + centered_time * float(np.sum(centered_time * centered_signal) / denominator)
    total_error = float(np.sum(centered_signal ** 2))
    linearity = 1.0 if total_error < EPS else float(1.0 - np.sum((values - prediction) ** 2) / total_error)
    return {
        "sq_noise_floor": noise_floor,
        "sq_baseline_stability": baseline_stability,
        "sq_dropout_rate": float(dropout_samples / values.size),
        "sq_identical_sample_rate": float(np.mean(differences == 0.0)),
        "sq_longest_repeated_run_frac": _longest_repeat_fraction(values),
        "sq_peak_consistency": peak_consistency,
        "sq_snr": snr,
        "sq_response_linearity": linearity,
    }


def _allan_features(values: np.ndarray, fs: float) -> dict[str, float]:
    """Equivalent overlapping frequency Allan deviation without allantools."""
    max_tau = values.size / fs / 2.0
    if max_tau <= 1.0 / fs:
        return {}
    requested = np.logspace(np.log10(1.0 / fs), np.log10(max_tau), num=10)
    cluster_sizes = sorted({max(1, int(round(tau * fs))) for tau in requested if tau < max_tau})
    output: dict[str, float] = {}
    taus: list[float] = []
    deviations: list[float] = []
    cumulative = np.concatenate(([0.0], np.cumsum(values)))
    for cluster in cluster_sizes:
        if values.size < 2 * cluster + 1:
            continue
        averages = (cumulative[cluster:] - cumulative[:-cluster]) / cluster
        delta = averages[cluster:] - averages[:-cluster]
        deviation = float(np.sqrt(0.5 * np.mean(delta ** 2)))
        tau = cluster / fs
        output[f"adev_{tau:.2f}s"] = deviation
        if deviation > 0 and np.isfinite(deviation):
            taus.append(tau)
            deviations.append(deviation)
    # Keep the feature schema stable across windows.  A constant temperature
    # window has zero ADEV at every tau and therefore no valid log slope, but
    # it must still expose the same feature column as a varying window.
    output["adev_slope"] = (
        float(np.polyfit(np.log10(taus), np.log10(deviations), 1)[0])
        if len(taus) > 2 else 0.0
    )
    return output


def extract_window(values: np.ndarray, fs: float, *, wavelet_level: int = 5,
                   include_adev: bool = True) -> dict[str, float]:
    values = _finite(values)
    variance = float(np.var(values))
    output = {
        "mean": float(np.mean(values)), "variance": variance,
        "rms": float(np.sqrt(np.mean(values ** 2))),
        "skewness": 0.0 if variance <= EPS else float(skew(values, bias=False)),
        "kurtosis": 0.0 if variance <= EPS else float(kurtosis(values, bias=False)),
        "zcr": float(np.count_nonzero(np.diff(np.sign(values))) / max(values.size - 1, 1)),
        "quantile_25": float(np.quantile(values, 0.25)),
        "quantile_50": float(np.quantile(values, 0.50)),
        "quantile_75": float(np.quantile(values, 0.75)),
    }
    frequencies, power = welch(values, fs=fs, nperseg=min(256, values.size), scaling="density")
    positive = (frequencies > 0) & (power > 0)
    output["psd_slope"] = (
        float(np.polyfit(np.log10(frequencies[positive]), np.log10(power[positive]), 1)[0])
        if np.count_nonzero(positive) >= 2 else 0.0
    )
    probability = np.clip(power, EPS, None)
    probability /= np.sum(probability)
    output["spectral_entropy"] = float(-np.sum(probability * np.log(probability)))
    output["spectral_flatness"] = float(np.exp(np.mean(np.log(np.clip(power, EPS, None)))) / np.mean(np.clip(power, EPS, None)))
    output["spectral_centroid"] = float(np.sum(frequencies * power) / (np.sum(power) + EPS))
    if include_adev and values.size >= 50:
        output.update(_allan_features(values, fs))
    if values.size >= 8:
        output.update(_wavelet_features(values, wavelet_level))
        output.update(_signal_quality(values))
    return {key: (value if np.isfinite(value) else 0.0) for key, value in output.items()}


def extract_batch(windows: np.ndarray, fs: float, *, wavelet_level: int = 5,
                  include_adev: bool = True) -> tuple[list[str], np.ndarray]:
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim == 2:
        windows = windows[:, :, None]
    if windows.ndim != 3 or not len(windows):
        raise ValueError("windows must have shape (windows, samples[, axes])")
    rows: list[dict[str, float]] = []
    for window in windows:
        row: dict[str, float] = {}
        for axis in range(window.shape[1]):
            for name, value in extract_window(
                window[:, axis], fs, wavelet_level=wavelet_level, include_adev=include_adev,
            ).items():
                row[f"axis{axis}_{name}"] = value
        rows.append(row)
    names = sorted(set.intersection(*(set(row) for row in rows)))
    matrix = np.asarray([[row[name] for name in names] for row in rows], dtype=np.float64)
    return names, matrix


def iter_feature_batches(window_batches: Iterable[tuple[np.ndarray, float]], **kwargs):
    for windows, fs in window_batches:
        yield extract_batch(windows, fs, **kwargs)
