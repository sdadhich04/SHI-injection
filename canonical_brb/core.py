"""Canonical BRB-r reliability weighting and bounded SHIELD health score."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


MANDATORY_SUFFIXES = (
    "sq_snr", "sq_noise_floor", "sq_baseline_stability", "sq_dropout_rate",
    "sq_identical_sample_rate", "sq_longest_repeated_run_frac",
)
EXPERT_WEIGHTS = {"mandatory": 1.0, "stable": 0.8, "sensitive": 0.5}


@dataclass
class BRBModel:
    feature_names: list[str]
    selected_features: list[str]
    categories: dict[str, list[str]]
    means: list[float]
    standard_deviations: list[float]
    feature_reliabilities: list[float]
    expert_weights: list[float]
    composite_weights: list[float]
    healthy_quantile: float
    healthy_threshold: float
    healthy_scale: float
    calibration_windows: int
    upstream_commit: str
    adaptations: list[str]
    model_type: str = "brb_r"

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def from_json(cls, path: Path) -> "BRBModel":
        return cls(**json.loads(path.read_text(encoding="utf-8")))


@dataclass
class BinaryEventModel:
    """Healthy-envelope fallback for a binary event channel with zero variance."""

    baseline_value: float
    healthy_quantile: float
    healthy_threshold: float
    healthy_scale: float
    calibration_windows: int
    window_size: int
    upstream_commit: str
    adaptations: list[str]
    model_type: str = "binary_event_health"

    def to_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


def _reliability(values: np.ndarray) -> float:
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    count = ordered.size
    if count < 2:
        return 1.0
    prefix = np.concatenate(([0.0], np.cumsum(ordered)))
    indices = np.arange(count, dtype=np.float64)
    distances = (
        ordered * indices - prefix[:-1]
        + (prefix[-1] - prefix[1:]) - ordered * (count - indices - 1.0)
    ) / count
    maximum = float(np.max(distances))
    return 1.0 if maximum <= 1e-12 else float(np.clip(1.0 - np.mean(distances / maximum), 0.0, 1.0))


def _category_map(names: list[str], matrix: np.ndarray, n_segments: int,
                  n_stable: int, n_sensitive: int) -> dict[str, list[str]]:
    segments = [part for part in np.array_split(matrix, n_segments) if len(part)]
    segment_means = np.stack([np.mean(part, axis=0) for part in segments])
    ddof = 1 if len(segments) > 1 else 0
    cv = np.std(segment_means, axis=0, ddof=ddof) / (np.abs(np.mean(segment_means, axis=0)) + 1e-12)
    order = np.argsort(cv, kind="stable")
    stable = [names[index] for index in order[:n_stable]]
    sensitive = [names[index] for index in order[-n_sensitive:]] if n_sensitive else []
    mandatory = [name for name in names if name.endswith(MANDATORY_SUFFIXES)]
    selected = list(dict.fromkeys(mandatory + stable + sensitive))
    return {"stable": stable, "sensitive": sensitive, "mandatory": mandatory, "all": selected}


def fit_brb(feature_names: list[str], healthy: np.ndarray, *, healthy_quantile: float = 0.95,
            n_segments: int = 4, n_stable: int = 8, n_sensitive: int = 4,
            upstream_commit: str = "222dbb1b0d742c7e8ea9b719c841ac5b1d2f2a72") -> BRBModel:
    healthy = np.asarray(healthy, dtype=np.float64)
    if healthy.ndim != 2 or healthy.shape[0] < max(4, n_segments):
        raise ValueError("at least four healthy calibration windows are required")
    if healthy.shape[1] != len(feature_names):
        raise ValueError("feature names do not match calibration matrix")
    finite_columns = np.isfinite(healthy).all(axis=0)
    varying_columns = np.nanstd(healthy, axis=0) > 1e-12
    keep = finite_columns & varying_columns
    names = [name for name, selected in zip(feature_names, keep) if selected]
    values = healthy[:, keep]
    if not names:
        raise ValueError("calibration contains no finite varying features")
    categories = _category_map(names, values, n_segments, n_stable, n_sensitive)
    name_to_index = {name: index for index, name in enumerate(names)}
    indices = [name_to_index[name] for name in categories["all"]]
    selected = values[:, indices]
    means = np.mean(selected, axis=0)
    standard_deviations = np.std(selected, axis=0, ddof=1)
    standard_deviations = np.maximum(standard_deviations, 1e-12)
    reliabilities = np.asarray([_reliability(selected[:, index]) for index in range(selected.shape[1])])
    category_for = {}
    for category in ("sensitive", "stable", "mandatory"):
        for name in categories[category]:
            category_for[name] = category
    expert = np.asarray([EXPERT_WEIGHTS[category_for[name]] for name in categories["all"]])
    normalized_expert = expert / max(float(np.max(expert)), 1e-12)
    composite = normalized_expert / np.maximum(1.0 + normalized_expert - reliabilities, 1e-12)
    raw = np.sum(np.abs((selected - means) / standard_deviations) * composite, axis=1) / np.sum(composite)
    quantile = float(np.clip(healthy_quantile, 0.5, 0.999))
    threshold = float(np.quantile(raw, quantile))
    median = float(np.quantile(raw, 0.5))
    iqr = float(np.quantile(raw, 0.75) - np.quantile(raw, 0.25))
    scale = max(threshold - median, iqr, 1e-6)
    return BRBModel(
        feature_names=feature_names, selected_features=categories["all"], categories=categories,
        means=means.tolist(), standard_deviations=standard_deviations.tolist(),
        feature_reliabilities=reliabilities.tolist(), expert_weights=expert.tolist(),
        composite_weights=composite.tolist(), healthy_quantile=quantile,
        healthy_threshold=threshold, healthy_scale=scale,
        calibration_windows=int(healthy.shape[0]), upstream_commit=upstream_commit,
        adaptations=[
            "NumPy batch matrices replace upstream pandas DataFrames.",
            "Calibration windows are sampled from each Project SHIELD stream.",
            "The upstream BRB-r equations and healthy-envelope mapping are unchanged.",
        ],
    )


def score_brb(model: BRBModel, feature_names: list[str], features: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float64)
    lookup = {name: index for index, name in enumerate(feature_names)}
    try:
        indices = [lookup[name] for name in model.selected_features]
    except KeyError as error:
        raise ValueError(f"scoring features omit calibrated feature {error.args[0]}") from error
    selected = features[:, indices]
    means = np.asarray(model.means)
    standard_deviations = np.asarray(model.standard_deviations)
    weights = np.asarray(model.composite_weights)
    normalized = np.abs((selected - means) / standard_deviations)
    raw = np.sum(normalized * weights, axis=1) / np.sum(weights)
    excess = np.clip(raw - model.healthy_threshold, 0.0, None)
    health = 1.0 / (1.0 + excess / max(model.healthy_scale, 1e-12))
    sensitive = [index for index, name in enumerate(model.selected_features)
                 if name in model.categories["sensitive"]]
    degradation = (
        np.sum(normalized[:, sensitive] * weights[sensitive], axis=1) / np.sum(weights[sensitive])
        if sensitive else raw.copy()
    )
    return health.astype(np.float32), raw.astype(np.float32), degradation.astype(np.float32)


def fit_binary_event(windows: np.ndarray, *, healthy_quantile: float = 0.95,
                     upstream_commit: str = "222dbb1b0d742c7e8ea9b719c841ac5b1d2f2a72") -> BinaryEventModel:
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim == 3 and windows.shape[2] == 1:
        windows = windows[:, :, 0]
    if windows.ndim != 2 or len(windows) < 4:
        raise ValueError("at least four binary calibration windows are required")
    finite = windows[np.isfinite(windows)]
    unique, counts = np.unique(finite, return_counts=True)
    if not len(unique) or len(unique) > 2 or not np.all(np.isin(unique, (0.0, 1.0))):
        raise ValueError("binary event fallback requires values limited to 0 and 1")
    baseline = float(unique[int(np.argmax(counts))])
    rates = np.mean(windows != baseline, axis=1)
    quantile = float(np.clip(healthy_quantile, 0.5, 0.999))
    threshold = float(np.quantile(rates, quantile))
    iqr = float(np.quantile(rates, 0.75) - np.quantile(rates, 0.25))
    scale = max(threshold - float(np.median(rates)), iqr, 1.0 / windows.shape[1])
    return BinaryEventModel(
        baseline_value=baseline, healthy_quantile=quantile,
        healthy_threshold=threshold, healthy_scale=scale,
        calibration_windows=int(len(windows)), window_size=int(windows.shape[1]),
        upstream_commit=upstream_commit,
        adaptations=[
            "The source BRB-r requires varying continuous features.",
            "Project SHIELD vibration is a binary event channel and can have an all-zero healthy baseline.",
            "Its SHI therefore uses event fraction relative to the calibrated baseline mode.",
        ],
    )


def score_binary_event(model: BinaryEventModel, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    windows = np.asarray(windows, dtype=np.float64)
    if windows.ndim == 3 and windows.shape[2] == 1:
        windows = windows[:, :, 0]
    if windows.ndim != 2:
        raise ValueError("binary event windows must have shape (windows, samples[, 1])")
    raw = np.mean(windows != model.baseline_value, axis=1)
    excess = np.clip(raw - model.healthy_threshold, 0.0, None)
    health = 1.0 / (1.0 + excess / max(model.healthy_scale, 1e-12))
    return health.astype(np.float32), raw.astype(np.float32), raw.astype(np.float32)
