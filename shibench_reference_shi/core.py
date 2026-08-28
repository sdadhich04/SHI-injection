"""Configurable reconstruction of the SHIBench reference SHI.

The available paper draft names the detector branches but omits its equations.
This module therefore makes every consequential reconstruction choice explicit
and serializable rather than presenting those choices as the paper's exact code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from model_shi.model_features import extract_feature_batch, get_pipeline_feature_names


REFERENCE_DTYPE = np.dtype(
    [
        ("timestamp_ms", "<u8"),
        ("shi", "<f4"),
        ("mahalanobis_health", "<f4"),
        ("isolation_health", "<f4"),
        ("ewma_health", "<f4"),
        ("event_health", "<f4"),
        ("flags", "u1"),
        # Explicit padding keeps each record at a round, versioned 48 bytes
        # without relying on platform-specific native alignment.
        ("reserved", "u1", (7,)),
        ("signal", "<f4", (3,)),
    ],
    align=False,
)
assert REFERENCE_DTYPE.itemsize == 48

FLAG_CONTINUOUS_ALARM = 1 << 0
FLAG_EVENT_ALARM = 1 << 1
FLAG_LIVENESS_ALARM = 1 << 2
FLAG_PLAUSIBILITY_ALARM = 1 << 3


@dataclass(frozen=True)
class ReconstructionConfig:
    """All choices not recoverable from the incomplete paper draft."""

    calibration_false_positive_rate: float = 0.05
    ewma_alpha: float = 0.15
    isolation_estimators: int = 64
    isolation_max_samples: int = 256
    random_state: int = 42
    logistic_odds_at_median: float = 9.0
    minimum_calibration_windows: int = 20
    reliability_floor: float = 0.05
    reliability_ratio_cap: float = 3.0
    event_branch: str = "auto"
    sparse_unique_limit: int = 16
    liveness_identical_fraction: float = 0.98
    plausibility_mad_multiplier: float = 20.0

    def validate(self) -> None:
        if not 0 < self.calibration_false_positive_rate < 0.5:
            raise ValueError("calibration_false_positive_rate must be between 0 and 0.5")
        if not 0 < self.ewma_alpha <= 1:
            raise ValueError("ewma_alpha must be in (0, 1]")
        if self.isolation_estimators < 1 or self.isolation_max_samples < 2:
            raise ValueError("invalid Isolation Forest size")
        if self.minimum_calibration_windows < 5:
            raise ValueError("minimum_calibration_windows must be at least 5")
        if self.event_branch not in {"auto", "all", "none"}:
            raise ValueError("event_branch must be auto, all, or none")
        if self.logistic_odds_at_median <= 1:
            raise ValueError("logistic_odds_at_median must exceed 1")


def feature_names(axes: int) -> list[str]:
    names = get_pipeline_feature_names(include_ar=False)
    if axes == 1:
        return names
    labels = ["x", "y", "z"][:axes]
    return [f"{axis}__{name}" for axis in labels for name in names]


def extract_reference_features(windows: np.ndarray, fs: float) -> np.ndarray:
    """Use the existing 22-feature-per-axis non-AR feature family."""
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim == 2:
        windows = windows[:, :, None]
    if windows.ndim != 3 or windows.shape[2] not in {1, 2, 3}:
        raise ValueError("windows must have shape (windows, samples, axes)")
    result = np.concatenate(
        [extract_feature_batch(windows[:, :, axis], fs, include_ar=False)
         for axis in range(windows.shape[2])],
        axis=1,
    )
    # Constant and quantized channels can make skew/kurtosis undefined. Treat
    # undefined derived values as neutral finite inputs and record liveness
    # separately instead of allowing NaNs to poison every detector branch.
    return np.nan_to_num(result, nan=0.0, posinf=1e12, neginf=-1e12)


def window_diagnostics(windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return maximum jump and exact-flat fraction for every window."""
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim == 2:
        windows = windows[:, :, None]
    differences = np.diff(windows, axis=1)
    maximum_jump = np.max(np.abs(differences), axis=(1, 2))
    identical = np.all(differences == 0.0, axis=2)
    flat_fraction = np.mean(identical, axis=1)
    return maximum_jump, flat_fraction


def _sample_jumps(windows: np.ndarray) -> np.ndarray:
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim == 2:
        windows = windows[:, :, None]
    return np.max(np.abs(np.diff(windows, axis=1)), axis=2)


def _event_rates(windows: np.ndarray, jump_threshold: float) -> np.ndarray:
    """Fraction of sample transitions classified as abrupt events per window."""
    jumps = _sample_jumps(windows)
    return np.mean(jumps > jump_threshold, axis=1)


def _quantile(values: np.ndarray, probability: float) -> float:
    value = float(np.quantile(np.asarray(values, dtype=np.float64), probability))
    return value if math.isfinite(value) else 0.0


def _logistic_health(scores: np.ndarray, median: float, threshold: float,
                     odds_at_median: float) -> np.ndarray:
    scale = abs(threshold - median)
    numerical_floor = max(abs(median) * 1e-9, 1e-12)
    if scale < numerical_floor:
        # A perfectly constant healthy baseline has no empirical 95th-percentile
        # gap. Give its observed value health 0.9 and make the first meaningful
        # positive deviation cross the transition, rather than leaving healthy
        # samples ambiguously at 0.5.
        scale = numerical_floor
        threshold = median + scale
    argument = np.log(odds_at_median) * (np.asarray(scores) - threshold) / scale
    return np.clip(1.0 / (1.0 + np.exp(np.clip(argument, -60.0, 60.0))), 0.0, 1.0)


def _final_health(raw_health: np.ndarray, threshold: float, median: float,
                  odds_at_median: float) -> np.ndarray:
    scale = abs(median - threshold)
    numerical_floor = max(abs(median) * 1e-9, 1e-12)
    if scale < numerical_floor:
        scale = numerical_floor
        threshold = median - scale
    argument = np.log(odds_at_median) * (np.asarray(raw_health) - threshold) / scale
    return np.clip(1.0 / (1.0 + np.exp(np.clip(-argument, -60.0, 60.0))), 0.0, 1.0)


def _ewma_residuals(z: np.ndarray, alpha: float, initial: np.ndarray | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    if not len(z):
        state = np.zeros(z.shape[1], dtype=np.float64) if initial is None else initial.copy()
        return np.empty(0, dtype=np.float64), state
    state = np.zeros(z.shape[1], dtype=np.float64) if initial is None else initial.copy()
    scores = np.empty(len(z), dtype=np.float64)
    divisor = math.sqrt(max(1, z.shape[1]))
    for index, row in enumerate(z):
        scores[index] = float(np.linalg.norm(row - state) / divisor)
        state = alpha * row + (1.0 - alpha) * state
    return scores, state


class ReferenceSHI:
    """Healthy-calibrated Mahalanobis/IsolationForest/EWMA monitor."""

    def __init__(self, config: ReconstructionConfig | None = None) -> None:
        self.config = config or ReconstructionConfig()
        self.config.validate()
        self.fitted = False
        self._ewma_state: np.ndarray | None = None

    def fit(self, features: np.ndarray, windows: np.ndarray, timestamps_ms: np.ndarray,
            sensor: str) -> "ReferenceSHI":
        features = np.asarray(features, dtype=np.float64)
        timestamps_ms = np.asarray(timestamps_ms, dtype=np.float64)
        if features.ndim != 2 or len(features) < self.config.minimum_calibration_windows:
            raise ValueError(
                f"need at least {self.config.minimum_calibration_windows} calibration windows"
            )
        if len(features) != len(windows) or len(features) != len(timestamps_ms):
            raise ValueError("calibration features/windows/timestamps must align")

        self.sensor = sensor
        self.axes = int(windows.shape[2] if windows.ndim == 3 else 1)
        self.scaler = StandardScaler().fit(features)
        z = self.scaler.transform(features)
        self.covariance = LedoitWolf().fit(z)
        self.isolation = IsolationForest(
            n_estimators=self.config.isolation_estimators,
            max_samples=min(self.config.isolation_max_samples, len(features)),
            contamination="auto",
            random_state=self.config.random_state,
            n_jobs=1,
        ).fit(z)

        mahalanobis = self.covariance.mahalanobis(z)
        isolation = -self.isolation.decision_function(z)
        ewma, _state = _ewma_residuals(z, self.config.ewma_alpha)
        anomaly_sets = [mahalanobis, isolation, ewma]
        probability = 1.0 - self.config.calibration_false_positive_rate
        self.component_calibration = []
        component_health = []
        for scores in anomaly_sets:
            median = _quantile(scores, 0.5)
            threshold = _quantile(scores, probability)
            self.component_calibration.append({"median": median, "threshold": threshold})
            component_health.append(
                _logistic_health(scores, median, threshold, self.config.logistic_odds_at_median)
            )
        health_matrix = np.column_stack(component_health)

        # Reliability is inverse healthy-baseline dispersion. Capping the ratio
        # prevents one unusually flat branch from dominating the fusion.
        dispersion = np.std(health_matrix, axis=0)
        raw_reliability = 1.0 / np.maximum(dispersion, self.config.reliability_floor)
        center = float(np.median(raw_reliability))
        raw_reliability = np.clip(
            raw_reliability,
            center / self.config.reliability_ratio_cap,
            center * self.config.reliability_ratio_cap,
        )
        self.weights = raw_reliability / raw_reliability.sum()
        raw_fused = np.exp(np.sum(self.weights * np.log(np.clip(health_matrix, 1e-6, 1.0)), axis=1))
        self.fused_threshold = _quantile(
            raw_fused, self.config.calibration_false_positive_rate,
        )
        self.fused_median = _quantile(raw_fused, 0.5)

        _maximum_jump, flat = window_diagnostics(windows)
        sample_jumps = _sample_jumps(windows)
        self.event_jump_threshold = _quantile(sample_jumps.reshape(-1), 0.995)
        event_rates = _event_rates(windows, self.event_jump_threshold)
        event_median = _quantile(event_rates, 0.5)
        event_threshold = _quantile(event_rates, probability)
        self.event_calibration = {
            "sample_jump_threshold": self.event_jump_threshold,
            "median_rate": event_median,
            "threshold_rate": event_threshold,
        }
        unique_values = np.unique(np.asarray(windows).reshape(-1, self.axes), axis=0)
        if self.config.event_branch == "all":
            self.event_enabled = True
        elif self.config.event_branch == "none":
            self.event_enabled = False
        else:
            self.event_enabled = sensor == "vibration" or len(unique_values) <= self.config.sparse_unique_limit

        flattened = np.asarray(windows, dtype=np.float64).reshape(-1, self.axes)
        self.signal_median = np.median(flattened, axis=0)
        signal_mad = np.median(np.abs(flattened - self.signal_median), axis=0)
        signal_std = np.std(flattened, axis=0)
        self.signal_scale = np.maximum(1.4826 * signal_mad, np.maximum(signal_std * 0.1, 1e-9))
        positive_gaps = np.diff(timestamps_ms)
        positive_gaps = positive_gaps[positive_gaps > 0]
        self.expected_window_gap_ms = float(np.median(positive_gaps)) if len(positive_gaps) else 0.0

        calibration_shi = _final_health(
            raw_fused, self.fused_threshold, self.fused_median,
            self.config.logistic_odds_at_median,
        )
        event_health = _logistic_health(
            event_rates, event_median, event_threshold,
            self.config.logistic_odds_at_median,
        )
        self.calibration_summary = {
            "windows": len(features),
            "continuous_false_positive_rate": float(np.mean(calibration_shi < 0.5)),
            "event_false_positive_rate": float(np.mean(event_health < 0.5)),
            "flat_fraction_median": _quantile(flat, 0.5),
        }
        self.fitted = True
        self.reset()
        return self

    def reset(self) -> None:
        if self.fitted:
            self._ewma_state = np.zeros(self.scaler.n_features_in_, dtype=np.float64)
            self._previous_timestamp_ms: float | None = None

    def score(self, features: np.ndarray, windows: np.ndarray, timestamps_ms: np.ndarray,
              ends: np.ndarray) -> dict[str, np.ndarray]:
        if not self.fitted:
            raise RuntimeError("ReferenceSHI must be fitted before scoring")
        features = np.asarray(features, dtype=np.float64)
        timestamps_ms = np.asarray(timestamps_ms, dtype=np.float64)
        ends = np.asarray(ends, dtype=np.float64)
        z = self.scaler.transform(features)
        mahalanobis = self.covariance.mahalanobis(z)
        isolation = -self.isolation.decision_function(z)
        ewma, self._ewma_state = _ewma_residuals(
            z, self.config.ewma_alpha, self._ewma_state,
        )
        raw_components = [mahalanobis, isolation, ewma]
        component_health = []
        for scores, calibration in zip(raw_components, self.component_calibration):
            component_health.append(_logistic_health(
                scores, calibration["median"], calibration["threshold"],
                self.config.logistic_odds_at_median,
            ))
        health_matrix = np.column_stack(component_health)
        raw_fused = np.exp(np.sum(
            self.weights * np.log(np.clip(health_matrix, 1e-6, 1.0)), axis=1,
        ))
        shi = _final_health(
            raw_fused, self.fused_threshold, self.fused_median,
            self.config.logistic_odds_at_median,
        )

        _maximum_jump, flat_fraction = window_diagnostics(windows)
        event_rates = _event_rates(windows, self.event_jump_threshold)
        event_health = _logistic_health(
            event_rates, self.event_calibration["median_rate"],
            self.event_calibration["threshold_rate"],
            self.config.logistic_odds_at_median,
        )
        if not self.event_enabled:
            event_health[:] = 1.0

        plausibility_distance = np.max(
            np.abs(ends[:, :self.axes] - self.signal_median) / self.signal_scale,
            axis=1,
        )
        liveness_alarm = flat_fraction >= self.config.liveness_identical_fraction
        plausibility_alarm = plausibility_distance > self.config.plausibility_mad_multiplier
        continuous_alarm = shi < 0.5
        event_alarm = event_health < 0.5
        flags = (
            continuous_alarm.astype(np.uint8) * FLAG_CONTINUOUS_ALARM
            | event_alarm.astype(np.uint8) * FLAG_EVENT_ALARM
            | liveness_alarm.astype(np.uint8) * FLAG_LIVENESS_ALARM
            | plausibility_alarm.astype(np.uint8) * FLAG_PLAUSIBILITY_ALARM
        )
        self._previous_timestamp_ms = float(timestamps_ms[-1]) if len(timestamps_ms) else self._previous_timestamp_ms
        return {
            "shi": shi,
            "mahalanobis_health": health_matrix[:, 0],
            "isolation_health": health_matrix[:, 1],
            "ewma_health": health_matrix[:, 2],
            "event_health": event_health,
            "flags": flags,
        }

    def metadata(self) -> dict:
        if not self.fitted:
            raise RuntimeError("ReferenceSHI is not fitted")
        return {
            "method": "reconstructed SHIBench reference SHI",
            "exact_reproduction": False,
            "config": asdict(self.config),
            "sensor": self.sensor,
            "axes": self.axes,
            "feature_count": int(self.scaler.n_features_in_),
            "feature_names": feature_names(self.axes),
            "component_order": ["mahalanobis", "isolation_forest", "ewma_residual"],
            "component_calibration": self.component_calibration,
            "reliability_weights": self.weights.tolist(),
            "fused_threshold": self.fused_threshold,
            "fused_median": self.fused_median,
            "event_enabled": self.event_enabled,
            "event_calibration": self.event_calibration,
            "signal_median": self.signal_median.tolist(),
            "signal_scale": self.signal_scale.tolist(),
            "expected_window_gap_ms": self.expected_window_gap_ms,
            "calibration_summary": self.calibration_summary,
            "alarm_logic": {
                "continuous": "shi < 0.5",
                "event": "event_health < 0.5 when enabled",
                "liveness": "identical-sample fraction exceeds configured limit",
                "plausibility": "endpoint exceeds calibrated robust envelope",
            },
        }
