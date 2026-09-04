"""Recovered Tier-1 fused SHI core, adapted for streaming Tier-2 recordings.

The continuous score in this module follows ``shield-tier1-shi``:

* 3 s windows with 50% overlap are selected by the runner;
* 20 features per axis (9 time, 8 sym4 SWT/MODWT, 3 signal-quality);
* IsolationForest(200, contamination=0.05, random_state=0);
* EmpiricalCovariance Mahalanobis distance;
* EWMA of the standardized feature-vector norm with lambda=0.1;
* calibration median/P99 branch normalization and equal-weight fusion;
* SHI = 1 - fused anomaly and a calibration-P5 alarm threshold.

The paper describes a complementary event-rate branch for sparse/binary Tier-2
channels but its exact equation is not present in the recovered source.  The
small ``EventRateModel`` below is therefore explicitly metadata-labelled as a
Tier-2 adaptation and never changes the continuous fused SHI.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pywt
from scipy.stats import kurtosis, skew
from sklearn.covariance import EmpiricalCovariance
from sklearn.ensemble import IsolationForest


PIPELINE_VERSION = 2
METHOD_NAME = "recovered Tier-1 fused SHI applied to Tier-2 hardware"

FLAG_CONTINUOUS_BREACH = 1 << 0
FLAG_CONTINUOUS_PERSISTENT = 1 << 1
FLAG_EVENT_ALARM = 1 << 2

TIER2_DTYPE = np.dtype(
    [
        ("timestamp_ms", "<u8"),
        ("shi", "<f4"),
        ("isolation_health", "<f4"),
        ("mahalanobis_health", "<f4"),
        ("ewma_health", "<f4"),
        ("event_health", "<f4"),
        ("alarm_threshold", "<f4"),
        ("flags", "u1"),
        ("reserved", "u1", (3,)),
        ("signal", "<f4", (3,)),
    ],
    align=False,
)
assert TIER2_DTYPE.itemsize == 48


@dataclass(frozen=True)
class FusedConfig:
    isolation_estimators: int = 200
    isolation_contamination: float = 0.05
    isolation_random_state: int = 0
    ewma_lambda: float = 0.1
    normalization_percentile: float = 99.0
    alarm_percentile: float = 5.0
    minimum_calibration_windows: int = 20

    def validate(self) -> None:
        if self.isolation_estimators < 1:
            raise ValueError("isolation_estimators must be positive")
        if not 0 < self.isolation_contamination < 0.5:
            raise ValueError("isolation_contamination must be in (0, 0.5)")
        if not 0 < self.ewma_lambda <= 1:
            raise ValueError("ewma_lambda must be in (0, 1]")
        if not 50 < self.normalization_percentile <= 100:
            raise ValueError("normalization_percentile must be in (50, 100]")
        if not 0 <= self.alarm_percentile < 50:
            raise ValueError("alarm_percentile must be in [0, 50)")
        if self.minimum_calibration_windows < 2:
            raise ValueError("minimum_calibration_windows must be at least 2")


def per_axis_feature_names() -> list[str]:
    return [
        "mean", "std", "rms", "skew", "kurtosis", "p25", "p50", "p75", "zcr",
        "swt_approx_L5_energy", "swt_detail_L5_energy", "swt_detail_L4_energy",
        "swt_detail_L3_energy", "swt_detail_L2_energy", "swt_detail_L1_energy",
        "swt_energy_entropy", "swt_hf_lf_ratio",
        "noise_floor", "baseline_stability", "log10_snr",
    ]


def feature_names(axes: int) -> list[str]:
    base = per_axis_feature_names()
    if axes == 1:
        return base
    return [f"{'xyz'[axis]}__{name}" for axis in range(axes) for name in base]


def _wavelet_features(x: np.ndarray, wavelet: str = "sym4", level: int = 5) -> np.ndarray:
    """Exact feature ordering and calculations from recovered pipeline.py."""
    n_out = level + 1 + 2
    if len(x) < 8:
        return np.zeros(n_out, dtype=np.float64)
    divisor = 2**level
    padding = (-len(x)) % divisor
    if padding:
        x = np.concatenate([x, np.zeros(padding, dtype=x.dtype)])
    try:
        coefficients = pywt.swt(x, wavelet, level=level, trim_approx=True, norm=True)
    except Exception:
        try:
            raw = pywt.swt(x, wavelet, level=level)
            coefficients = [raw[0][0]] + [detail for _approximation, detail in raw]
        except Exception:
            return np.zeros(n_out, dtype=np.float64)
    energies = [float(np.sum(coefficient**2)) for coefficient in coefficients]
    while len(energies) < level + 1:
        energies.append(0.0)
    energies = energies[: level + 1]
    total = sum(energies) + 1e-12
    relative = np.asarray([energy / total for energy in energies])
    positive = relative > 0
    entropy = float(-(relative[positive] * np.log(relative[positive])).sum())
    hf_lf = float(sum(energies[1:3]) / (energies[-1] + 1e-12))
    return np.asarray(energies + [entropy, hf_lf], dtype=np.float64)


def _quality_features(x: np.ndarray) -> np.ndarray:
    if len(x) < 4:
        return np.zeros(3, dtype=np.float64)
    differences = np.diff(x)
    mad = float(np.median(np.abs(differences - np.median(differences))))
    noise = mad * 1.4826 / math.sqrt(2.0)
    segment_size = max(1, len(x) // 5)
    segments = [
        x[index * segment_size : (index + 1) * segment_size]
        for index in range(5)
        if (index + 1) * segment_size <= len(x)
    ]
    baseline = float(np.std([np.mean(segment) for segment in segments])) if len(segments) >= 2 else 0.0
    signal_power = float(np.var(x))
    snr = float(signal_power / max(noise**2, 1e-12))
    return np.asarray([noise, baseline, np.log10(snr + 1e-12)], dtype=np.float64)


def window_features(window: np.ndarray) -> np.ndarray:
    window = np.asarray(window, dtype=np.float64)
    if window.ndim == 1:
        window = window[:, None]
    features: list[float] = []
    for axis in range(window.shape[1]):
        x = window[:, axis]
        standard_deviation = float(np.std(x))
        features.extend([
            float(np.mean(x)), standard_deviation, float(np.sqrt(np.mean(x**2))),
            float(skew(x) if standard_deviation > 0 else 0.0),
            float(kurtosis(x) if standard_deviation > 0 else 0.0),
            float(np.percentile(x, 25)), float(np.percentile(x, 50)),
            float(np.percentile(x, 75)), float(((x[:-1] * x[1:]) < 0).mean()),
        ])
        features.extend(_wavelet_features(x))
        features.extend(_quality_features(x))
    return np.asarray(features, dtype=np.float64)


def extract_features(windows: np.ndarray) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim == 2:
        windows = windows[:, :, None]
    if windows.ndim != 3 or windows.shape[2] not in {1, 2, 3}:
        raise ValueError("windows must have shape (window_count, samples, axes)")
    result = np.stack([window_features(window) for window in windows])
    if not np.all(np.isfinite(result)):
        raise ValueError("feature extraction produced non-finite values")
    return result


def event_rates(windows: np.ndarray) -> np.ndarray:
    """Fraction of sample-to-sample state changes in each window."""
    windows = np.asarray(windows)
    if windows.ndim == 2:
        windows = windows[:, :, None]
    changed = np.any(np.diff(windows, axis=1) != 0, axis=2)
    return np.mean(changed, axis=1)


class EventRateModel:
    """Two-sided event-rate adaptation for the draft's sparse Tier-2 branch."""

    def fit(self, rates: np.ndarray) -> "EventRateModel":
        rates = np.asarray(rates, dtype=np.float64)
        if len(rates) < 2 or not np.all(np.isfinite(rates)):
            raise ValueError("event-rate calibration needs at least two finite windows")
        self.rate_median = float(np.median(rates))
        deviations = np.abs(rates - self.rate_median)
        self.deviation_median = float(np.median(deviations))
        self.deviation_p95 = float(np.percentile(deviations, 95))
        if self.deviation_p95 <= self.deviation_median:
            self.deviation_p95 = self.deviation_median + max(abs(self.rate_median) * 1e-9, 1e-12)
        # score_rates maps the calibration P95 deviation to exactly zero.
        # Testing <= 0 therefore implements the P95 anomaly boundary.  Taking
        # P5 of the clipped health values is not equivalent for a degenerate
        # binary baseline: all calibration windows may have health 1, which
        # would incorrectly alarm every window.
        self.alarm_threshold = 0.0
        return self

    def score_rates(self, rates: np.ndarray) -> np.ndarray:
        deviations = np.abs(np.asarray(rates, dtype=np.float64) - self.rate_median)
        anomaly = np.clip(
            (deviations - self.deviation_median) / (self.deviation_p95 - self.deviation_median),
            0.0, 1.0,
        )
        return 1.0 - anomaly

    def metadata(self) -> dict:
        return {
            "method": "two-sided event-rate deviation",
            "exact_recovered_formula": False,
            "reason": "draft describes the branch but recovered source omits its equation",
            "rate_median": self.rate_median,
            "deviation_median": self.deviation_median,
            "deviation_p95": self.deviation_p95,
            "alarm_threshold": self.alarm_threshold,
        }


class FusedSHI:
    """Stateful recovered fused detector; call reset before each recording."""

    def __init__(self, config: FusedConfig | None = None) -> None:
        self.config = config or FusedConfig()
        self.config.validate()
        self.fitted = False

    @staticmethod
    def _normalizer(raw: np.ndarray, percentile: float) -> dict[str, float]:
        lower = float(np.median(raw))
        upper = float(np.percentile(raw, percentile))
        if upper <= lower:
            upper = lower + 1e-9
        return {"lower": lower, "upper": upper}

    @staticmethod
    def _normalize(raw: np.ndarray, normalizer: dict[str, float]) -> np.ndarray:
        return np.clip(
            (np.asarray(raw) - normalizer["lower"])
            / (normalizer["upper"] - normalizer["lower"]),
            0.0, 1.0,
        )

    def fit(self, calibration_features: np.ndarray, names: list[str]) -> "FusedSHI":
        features = np.asarray(calibration_features, dtype=np.float64)
        if features.ndim != 2 or len(features) < self.config.minimum_calibration_windows:
            raise ValueError(
                f"need at least {self.config.minimum_calibration_windows} calibration windows"
            )
        if not np.all(np.isfinite(features)):
            raise ValueError("calibration features contain non-finite values")
        if features.shape[1] != len(names):
            raise ValueError("feature names do not match calibration matrix")

        self.feature_names = list(names)
        self.mean = features.mean(axis=0)
        self.standard_deviation = features.std(axis=0) + 1e-9
        standardized = (features - self.mean) / self.standard_deviation
        self.keep = self.standard_deviation > 1e-6
        selected = standardized[:, self.keep]
        if not selected.shape[1]:
            raise ValueError("calibration contains no finite varying features")

        self.isolation = IsolationForest(
            n_estimators=self.config.isolation_estimators,
            contamination=self.config.isolation_contamination,
            random_state=self.config.isolation_random_state,
            n_jobs=1,
        ).fit(selected)
        self.covariance = EmpiricalCovariance().fit(selected)

        ewma_state = 0.0
        ewma_calibration = np.zeros(len(selected), dtype=np.float64)
        for index, row in enumerate(selected):
            ewma_state = (
                self.config.ewma_lambda * np.linalg.norm(row)
                + (1.0 - self.config.ewma_lambda) * ewma_state
            )
            ewma_calibration[index] = ewma_state

        isolation_raw = -self.isolation.score_samples(selected)
        mahalanobis_raw = self.covariance.mahalanobis(selected)
        self.isolation_normalizer = self._normalizer(
            isolation_raw, self.config.normalization_percentile,
        )
        self.mahalanobis_normalizer = self._normalizer(
            mahalanobis_raw, self.config.normalization_percentile,
        )
        self.ewma_normalizer = self._normalizer(
            ewma_calibration, self.config.normalization_percentile,
        )
        isolation = self._normalize(isolation_raw, self.isolation_normalizer)
        mahalanobis = self._normalize(mahalanobis_raw, self.mahalanobis_normalizer)
        ewma = self._normalize(ewma_calibration, self.ewma_normalizer)
        calibration_shi = 1.0 - (isolation + mahalanobis + ewma) / 3.0
        self.alarm_threshold = float(np.percentile(calibration_shi, self.config.alarm_percentile))
        self.ewma_calibration_mean = float(ewma_calibration.mean())
        self.calibration_shi = calibration_shi
        self.fitted = True
        self.reset()
        return self

    def reset(self) -> None:
        if self.fitted:
            self._ewma_state = self.ewma_calibration_mean
            self._breach_streak = 0

    def score(self, features: np.ndarray) -> dict[str, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("FusedSHI must be fitted before scoring")
        features = np.asarray(features, dtype=np.float64)
        standardized = (features - self.mean) / self.standard_deviation
        selected = standardized[:, self.keep]
        selected = np.nan_to_num(selected, nan=0.0, posinf=10.0, neginf=-10.0)
        isolation = self._normalize(
            -self.isolation.score_samples(selected), self.isolation_normalizer,
        )
        mahalanobis = self._normalize(
            self.covariance.mahalanobis(selected), self.mahalanobis_normalizer,
        )
        ewma_raw = np.zeros(len(selected), dtype=np.float64)
        for index, row in enumerate(selected):
            self._ewma_state = (
                self.config.ewma_lambda * np.linalg.norm(row)
                + (1.0 - self.config.ewma_lambda) * self._ewma_state
            )
            ewma_raw[index] = self._ewma_state
        ewma = self._normalize(ewma_raw, self.ewma_normalizer)
        shi = 1.0 - (isolation + mahalanobis + ewma) / 3.0

        persistent = np.zeros(len(shi), dtype=bool)
        breaches = shi < self.alarm_threshold
        for index, breach in enumerate(breaches):
            self._breach_streak = self._breach_streak + 1 if breach else 0
            persistent[index] = self._breach_streak >= 2
        return {
            "shi": shi,
            "isolation_health": 1.0 - isolation,
            "mahalanobis_health": 1.0 - mahalanobis,
            "ewma_health": 1.0 - ewma,
            "continuous_breach": breaches,
            "continuous_persistent": persistent,
        }

    def metadata(self) -> dict:
        if not self.fitted:
            raise RuntimeError("FusedSHI must be fitted before metadata is available")
        return {
            "method": METHOD_NAME,
            "continuous_formula_exact_to_recovered_source": True,
            "feature_count_total": len(self.feature_names),
            "retained_feature_count": int(np.count_nonzero(self.keep)),
            "feature_names": self.feature_names,
            "retained_features": [
                name for name, retained in zip(self.feature_names, self.keep) if retained
            ],
            "parameters": {
                "isolation_forest": {
                    "n_estimators": self.config.isolation_estimators,
                    "contamination": self.config.isolation_contamination,
                    "random_state": self.config.isolation_random_state,
                },
                "covariance": "sklearn.covariance.EmpiricalCovariance",
                "ewma_lambda": self.config.ewma_lambda,
                "normalization": {
                    "lower": "calibration median",
                    "upper_percentile": self.config.normalization_percentile,
                    "clip": [0.0, 1.0],
                },
                "fusion_weights": [1.0 / 3.0] * 3,
                "alarm_percentile": self.config.alarm_percentile,
                "persistence_windows": 2,
            },
            "normalizers": {
                "isolation_forest": self.isolation_normalizer,
                "mahalanobis": self.mahalanobis_normalizer,
                "ewma": self.ewma_normalizer,
            },
            "alarm_threshold": self.alarm_threshold,
            "calibration_window_count": len(self.calibration_shi),
            "calibration_shi_mean": float(np.mean(self.calibration_shi)),
            "calibration_shi_min": float(np.min(self.calibration_shi)),
            "ewma_calibration_mean": self.ewma_calibration_mean,
        }
