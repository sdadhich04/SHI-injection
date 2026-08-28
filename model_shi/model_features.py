"""Numerically equivalent, batched feature pipeline from shield-model.

Source commit: c094797c96923449b6075d8c483df37856b26712
Source file: https://github.com/samkorostov/shield-model/blob/main/features.py

The checked-in upstream source includes AR-Burg despite the README saying that
it is disabled. Model SHI follows the source code and enables AR-Burg by
default. The vectorized implementations preserve upstream numerical results.
"""

import numpy as np
import pywt
import warnings
from functools import lru_cache
from scipy import stats as sp_stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from spectrum import arburg


class MODWTFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, wavelet="db4", level=4, features=("energy", "variance")):
        self.wavelet = wavelet
        self.level = level
        self.features = features

    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        names = []
        for level in range(1, self.level + 1):
            if "energy" in self.features:
                names.append(f"detail_energy_L{level}")
            if "variance" in self.features:
                names.append(f"detail_variance_L{level}")
            if "mean_abs" in self.features:
                names.append(f"detail_mean_abs_L{level}")
        if "energy" in self.features:
            names.append(f"approx_energy_L{self.level}")
        if "variance" in self.features:
            names.append(f"approx_variance_L{self.level}")
        if "mean_abs" in self.features:
            names.append(f"approx_mean_abs_L{self.level}")
        return np.array(names)

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("MODWT input must have shape (windows, samples)")
        target_len = int(2 ** np.ceil(np.log2(max(X.shape[1], 2**self.level))))
        if X.shape[1] < target_len:
            X = np.pad(X, ((0, 0), (0, target_len - X.shape[1])), mode="symmetric")
        coeffs = pywt.swt(X, self.wavelet, level=self.level, axis=1)
        columns = []
        for c_a, c_d in coeffs:
            if "energy" in self.features:
                columns.append(np.sum(c_d**2, axis=1))
            if "variance" in self.features:
                columns.append(np.var(c_d, axis=1))
            if "mean_abs" in self.features:
                columns.append(np.mean(np.abs(c_d), axis=1))
        if "energy" in self.features:
            columns.append(np.sum(c_a**2, axis=1))
        if "variance" in self.features:
            columns.append(np.var(c_a, axis=1))
        if "mean_abs" in self.features:
            columns.append(np.mean(np.abs(c_a), axis=1))
        return np.column_stack(columns)

    def _extract(self, window):
        n = len(window)
        target_len = int(2 ** np.ceil(np.log2(max(n, 2**self.level))))
        if n < target_len:
            window = np.pad(window, (0, target_len - n), mode="symmetric")

        coeffs = pywt.swt(window, self.wavelet, level=self.level)
        feats = []
        for c_a, c_d in coeffs:
            if "energy" in self.features:
                feats.append(np.sum(c_d**2))
            if "variance" in self.features:
                feats.append(np.var(c_d))
            if "mean_abs" in self.features:
                feats.append(np.mean(np.abs(c_d)))

        if "energy" in self.features:
            feats.append(np.sum(c_a**2))
        if "variance" in self.features:
            feats.append(np.var(c_a))
        if "mean_abs" in self.features:
            feats.append(np.mean(np.abs(c_a)))
        return np.array(feats)


class TimeDomainFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return np.array(
            ["mean", "var", "rms", "skew", "kurtosis", "zero_crossing_rate"]
        )

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        means = np.mean(X, axis=1)
        variances = np.var(X, axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            skew = sp_stats.skew(X, axis=1)
            kurtosis = sp_stats.kurtosis(X, axis=1)
        skew = np.nan_to_num(skew)
        kurtosis = np.nan_to_num(kurtosis)
        crossing_rate = np.mean(np.diff(np.sign(X), axis=1) != 0, axis=1)
        return np.column_stack(
            [means, variances, np.sqrt(np.mean(X**2, axis=1)), skew, kurtosis, crossing_rate]
        )

    def _extract(self, window):
        crossing_rate = np.sum(np.diff(np.sign(window)) != 0) / (len(window) - 1)
        if np.var(window) <= np.finfo(np.float64).eps:
            skew = 0.0
            kurtosis = 0.0
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                skew = sp_stats.skew(window)
                kurtosis = sp_stats.kurtosis(window)
        return np.array(
            [
                np.mean(window),
                np.var(window),
                np.sqrt(np.mean(window**2)),
                skew,
                kurtosis,
                crossing_rate,
            ]
        )


class FrequencyDomainFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, fs=1000, bands=((0, 10), (10, 100), (100, 500))):
        self.fs = fs
        self.bands = bands

    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        bands = [f"band_energy_{lo}_{hi}hz" for lo, hi in self.bands]
        return np.array([*bands, "spectral_centroid", "spectral_flatness"])

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        sample_count = X.shape[1]
        frequencies = np.fft.rfftfreq(sample_count, d=1 / self.fs)
        power = np.abs(np.fft.rfft(X, axis=1)) ** 2
        total = np.sum(power, axis=1) + 1e-12
        columns = []
        for low, high in self.bands:
            mask = (frequencies >= low) & (frequencies < high)
            columns.append(np.sum(power[:, mask], axis=1) / total)
        columns.append(np.sum(power * frequencies[None, :], axis=1) / total)
        log_mean = np.mean(np.log(power + 1e-12), axis=1)
        columns.append(np.exp(log_mean) / (np.mean(power, axis=1) + 1e-12))
        return np.column_stack(columns)

    def _extract(self, window):
        sample_count = len(window)
        frequencies = np.fft.rfftfreq(sample_count, d=1 / self.fs)
        power = np.abs(np.fft.rfft(window)) ** 2
        total = np.sum(power) + 1e-12
        band_energies = []
        for low, high in self.bands:
            mask = (frequencies >= low) & (frequencies < high)
            band_energies.append(np.sum(power[mask]) / total)
        centroid = np.sum(frequencies * power) / total
        log_mean = np.mean(np.log(power + 1e-12))
        flatness = np.exp(log_mean) / (np.mean(power) + 1e-12)
        return np.array([*band_energies, centroid, flatness])


class ARBurgFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, order=6):
        self.order = order

    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return np.array(
            ["ar_mean_angle", "ar_std_angle", "ar_mean_magnitude", "ar_log_noise_var"]
        )

    def transform(self, X):
        return np.array([self._extract(window) for window in X])

    def _extract(self, window):
        try:
            if np.var(window) <= np.finfo(np.float64).eps:
                return np.zeros(4)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                coefficients, noise_variance, _ = arburg(window, self.order)
            roots = np.roots(np.concatenate([[1], -coefficients]))
            angles = np.abs(np.angle(roots))
            magnitudes = np.abs(roots)
            return np.array(
                [
                    np.mean(angles),
                    np.std(angles),
                    np.mean(magnitudes),
                    np.log(noise_variance + 1e-12),
                ]
            )
        except Exception:
            return np.zeros(4)


class StabilityFeature(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return np.array(["mean_squared_diff"])

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.mean(np.diff(X, axis=1) ** 2, axis=1, keepdims=True)


@lru_cache(maxsize=64)
def build_feature_pipeline(fs: float, include_ar: bool = True) -> FeatureUnion:
    transformers = [
        ("time", TimeDomainFeatures()),
        ("freq", FrequencyDomainFeatures(fs=fs)),
        ("stability", StabilityFeature()),
        ("modwt", MODWTFeatureExtractor(wavelet="db4", level=4)),
    ]
    if include_ar:
        transformers.append(("ar", ARBurgFeatures(order=6)))
    return FeatureUnion(transformers)


def extract_feature_batch(windows: np.ndarray, fs: float, include_ar: bool = True) -> np.ndarray:
    """Extract the upstream feature set from a 2-D window batch.

    The default follows upstream ``features.py``: 26 features including AR-Burg.
    """
    result = build_feature_pipeline(fs, include_ar=include_ar).transform(windows)
    return np.nan_to_num(result, nan=0.0, posinf=1e12, neginf=-1e12)


def get_pipeline_feature_names(is_imu: bool = False, include_ar: bool = True) -> list[str]:
    base_names = list(
        build_feature_pipeline(fs=1000.0, include_ar=include_ar).get_feature_names_out()
    )
    if not is_imu:
        return base_names
    return [f"{axis}__{name}" for axis in ("x", "y", "z") for name in base_names]
