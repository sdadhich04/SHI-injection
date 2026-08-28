import numpy as np
import pywt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from scipy import stats as sp_stats
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
        for lvl in range(1, self.level + 1):
            if "energy" in self.features:
                names.append(f"detail_energy_L{lvl}")
            if "variance" in self.features:
                names.append(f"detail_variance_L{lvl}")
            if "mean_abs" in self.features:
                names.append(f"detail_mean_abs_L{lvl}")
        if "energy" in self.features:
            names.append(f"approx_energy_L{self.level}")
        if "variance" in self.features:
            names.append(f"approx_variance_L{self.level}")
        if "mean_abs" in self.features:
            names.append(f"approx_mean_abs_L{self.level}")
        return np.array(names)

    def transform(self, X):
        return np.array([self._extract(window) for window in X])

    def _extract(self, window):
        n = len(window)
        target_len = int(2 ** np.ceil(np.log2(max(n, 2**self.level))))
        if n < target_len:
            window = np.pad(window, (0, target_len - n), mode="symmetric")

        coeffs = pywt.swt(window, self.wavelet, level=self.level)

        feats = []
        for cA, cD in coeffs:
            if "energy" in self.features:
                feats.append(np.sum(cD**2))
            if "variance" in self.features:
                feats.append(np.var(cD))
            if "mean_abs" in self.features:
                feats.append(np.mean(np.abs(cD)))

        if "energy" in self.features:
            feats.append(np.sum(cA**2))
        if "variance" in self.features:
            feats.append(np.var(cA))
        if "mean_abs" in self.features:
            feats.append(np.mean(np.abs(cA)))

        return np.array(feats)


class TimeDomainFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        return np.array(
            ["mean", "var", "rms", "skew", "kurtosis", "zero_crossing_rate"]
        )

    def transform(self, X):
        return np.array([self._extract(w) for w in X])

    def _extract(self, w):
        zc = np.sum(np.diff(np.sign(w)) != 0) / (len(w) - 1)
        return np.array(
            [
                np.mean(w),
                np.var(w),
                np.sqrt(np.mean(w**2)),
                sp_stats.skew(w),
                sp_stats.kurtosis(w),
                zc,
            ]
        )


class FrequencyDomainFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, fs=1000, bands=((0, 10), (10, 100), (100, 500))):
        self.fs = fs
        self.bands = bands

    def fit(self, X, y=None):
        return self

    def get_feature_names_out(self, input_features=None):
        band_names = [f"band_energy_{lo}_{hi}hz" for lo, hi in self.bands]
        return np.array([*band_names, "spectral_centroid", "spectral_flatness"])

    def transform(self, X):
        return np.array([self._extract(w) for w in X])

    def _extract(self, w):
        N = len(w)
        freqs = np.fft.rfftfreq(N, d=1 / self.fs)
        psd = np.abs(np.fft.rfft(w)) ** 2
        total = np.sum(psd) + 1e-12

        band_energies = []
        for lo, hi in self.bands:
            mask = (freqs >= lo) & (freqs < hi)
            band_energies.append(np.sum(psd[mask]) / total)

        centroid = np.sum(freqs * psd) / total

        log_mean = np.mean(np.log(psd + 1e-12))
        flatness = np.exp(log_mean) / (np.mean(psd) + 1e-12)

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
        return np.array([self._extract(w) for w in X])

    def _extract(self, w):
        try:
            ar_coeffs, noise_var, _ = arburg(w, self.order)
            roots = np.roots(np.concatenate([[1], -ar_coeffs]))
            angles = np.abs(np.angle(roots))
            magnitudes = np.abs(roots)
            return np.array(
                [
                    np.mean(angles),
                    np.std(angles),
                    np.mean(magnitudes),
                    np.log(noise_var + 1e-12),
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
        return np.array([[np.mean(np.diff(w) ** 2)] for w in X])


def build_feature_pipeline(fs: float) -> FeatureUnion:
    return FeatureUnion(
        [
            ("time", TimeDomainFeatures()),
            ("freq", FrequencyDomainFeatures(fs=fs)),
            ("stability", StabilityFeature()),
            ("modwt", MODWTFeatureExtractor(wavelet="db4", level=4)),
            ("ar", ARBurgFeatures(order=6)),
        ]
    )


def get_pipeline_feature_names(is_imu: bool = False) -> list[str]:
    """Returns ordered feature column names matching what pipeline.py produces."""
    base_names = list(build_feature_pipeline(fs=1000.0).get_feature_names_out())
    if not is_imu:
        return base_names
    return [f"{ax}__{name}" for ax in ("x", "y", "z") for name in base_names]
