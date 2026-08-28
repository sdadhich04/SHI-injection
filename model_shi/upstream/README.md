# Shield Model

Predictive fault detection for embedded sensor systems. The pipeline ingests raw sensor CSVs, synthesizes fault scenarios via injection, extracts time/frequency/wavelet features from sliding windows, and trains binary classifiers (RandomForest, XGBoost) to flag **healthy** vs **fault** windows up to 20 seconds before failure.

---

## Repository Layout

```
shield-model/
├── fault_injection.py        # Fault synthesis engine
├── features.py               # Feature extractors (sklearn transformers)
├── pipeline.py               # End-to-end data → feature matrix pipeline
├── train.ipynb               # Training notebook
├── requirements.txt          # Python dependencies
│
├── data/
│   ├── data/UNIT_0001_RUN_033/   # Raw sensor CSVs
│   └── metadata/sessions/        # Session metadata
│
├── processed_windows/        # Per-run feature chunks (.npy)
│   ├── vibration/            # Per-sensor subdirectory (single-sensor mode)
│   └── temperature/
│
├── rf.joblib                 # Saved RandomForest model
├── xgb.joblib                # Saved XGBoost model
├── scaler_scalar.joblib      # StandardScaler for scalar sensors
├── X_scalar.npy              # Feature matrix (last full run)
├── y_scalar.npy              # Labels (0=healthy, 1=pre-fault, 2=fault)
├── sensor_scalar.npy         # Sensor name per window row
└── feature_names_scalar.json # Ordered feature column names
```

---

## Components

### `fault_injection.py`

Synthesizes realistic sensor faults from clean recordings. Each fault is injected at a random onset index, and windows are labelled with a three-class scheme:

| Label | Meaning |
|-------|---------|
| `0` | Healthy |
| `1` | Pre-fault warning zone (configurable look-ahead, default 20 s) |
| `2` | Active fault |

**Active fault types** (enabled in `FAULT_CONFIGS`):

| Fault | Severity range | Description |
|-------|---------------|-------------|
| `bias_offset` | 0.5 – 3.0 | Constant DC shift added after onset |
| `drift` | 0.05 – 0.5 | Linear ramp added after onset |

Additional fault types are implemented but currently commented out: `noise_increase`, `dropout`, `latency`, `saturation`, `calibration_shift`.

**Key functions:**

- `load_signal(df)` — parses a 2-column `[timestamp_ms, value]` or 4-column `[timestamp_ms, x, y, z]` DataFrame and returns `(signal, fs)`.
- `inject_fault(signal, fault_type, onset_idx, severity, pre_fault_samples)` — injects a fault into a 1-D signal and returns `(corrupted, labels)`.
- `inject_fault_multiaxis(signal, ...)` — wraps `inject_fault` to handle both 1-D scalar and 2-D IMU `(n, 3)` signals.
- `generate_faulty_dataset(csv_paths, prediction_horizon_sec)` — generator that yields one scenario dict at a time to keep memory usage low.

---

### `features.py`

sklearn-compatible transformers that operate on `(n_windows, window_size)` arrays. All transformers implement `fit`, `transform`, and `get_feature_names_out`.

| Class | Features produced |
|-------|------------------|
| `TimeDomainFeatures` | mean, variance, RMS, skewness, kurtosis, zero-crossing rate |
| `FrequencyDomainFeatures(fs, bands)` | per-band energy fractions, spectral centroid, spectral flatness |
| `ARBurgFeatures(order)` | AR pole angles & magnitudes, log noise variance *(disabled by default)* |
| `MODWTFeatureExtractor(wavelet, level)` | per-level detail energy & variance + approximation energy & variance via stationary wavelet transform |
| `StabilityFeature` | mean squared first difference |

`build_feature_pipeline(fs)` returns a `FeatureUnion` of the active transformers, yielding **22 features per window** for scalar sensors and **66 features** for 3-axis IMU sensors.

`get_pipeline_feature_names(is_imu)` returns the ordered column name list that matches the pipeline output.

---

### `pipeline.py`

Orchestrates fault injection → windowing → feature extraction → scaling for an entire dataset directory.

**Windowing parameters** (top of file):

| Constant | Default | Meaning |
|----------|---------|---------|
| `WINDOW_SIZE` | 256 samples | Samples per window |
| `STRIDE` | 64 samples | Step between windows (75 % overlap) |

**Key functions:**

- `make_windows(sig_1d)` — zero-copy strided view over a 1-D signal.
- `process_entry(args)` — worker function run per job; loads one CSV, injects a fault, windows and extracts features, then writes `X_<tag>_NNNN.npy`, `y_<tag>_NNNN.npy`, `sensor_<tag>_NNNN.npy` to disk.
- `build_job_list(csv_paths, prediction_horizon_sec)` — reads only timestamps (cheap) to compute `fs` and build the flat job list (one healthy + one per active fault type, per CSV).
- `run_pipeline(csv_files, prediction_horizon_sec, n_workers, sensor_filter)` — full pipeline entry point; runs jobs in parallel, stacks chunks, fits `StandardScaler`s, and returns scaled matrices + metadata.

**CLI usage:**

```bash
python pipeline.py <data_dir> \
    [--prediction-horizon 20.0] \
    [--n-workers 4] \
    [--sensor <sensor_id>]
```

- `<data_dir>` — directory searched recursively for `*.csv` files.
- `--sensor` — restrict processing to a single sensor (e.g. `vibration`). Outputs go to `processed_windows/<sensor>/`.

---

### `train.ipynb`

Step-by-step training notebook:

1. **Install dependencies** — `pip install -r requirements.txt`
2. **Discover sensors** — lists available CSVs under `DATASET_PATH`.
3. **Run pipeline** — calls `run_pipeline`; set `SENSOR_ID` to a sensor name or `None` for all sensors.
4. **Save feature matrices** — persists `X_*.npy`, `y_*.npy`, `sensor_*.npy`, scalers, and feature name JSON files.
5. **Reload chunks** — loads only the pre-computed `.npy` chunks (skips re-extraction if already done), binarises labels to `0=healthy / 1=fault`.
6. **Train & evaluate** — stratified 80/20 split, fits `RandomForestClassifier` and `XGBClassifier`, prints `classification_report`.
7. **Save models** — dumps `rf.joblib` and `xgb.joblib`.

> **Note:** The current split is stratified random, not temporal. Because fault windows are contiguous, this may overestimate performance due to temporal leakage. A time-based split is recommended for rigorous evaluation.

---

## Data Format

### Sensor CSVs

Scalar sensors (vibration, temperature, pressure, …):
```
timestamp_ms,value
4614,0.0
4615,0.0
```

IMU sensors (accelerometer, gyroscope, magnetometer):
```
timestamp_ms,x,y,z
4614,0.765625,-0.078125,9.769531
```

### Session metadata (`data/metadata/sessions/sessions.csv`)

Records per-run context: sensor name, file, sampling rate, units, and health label for the session.

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

Place sensor CSVs under any directory structure. The pipeline discovers them recursively. The included sample is at `data/data/UNIT_0001_RUN_033/`.

### 3. Run the pipeline (CLI)

```bash
# Single sensor, 8 workers
python pipeline.py data/data/UNIT_0001_RUN_033 --sensor vibration --n-workers 8

# All sensors
python pipeline.py data/data/UNIT_0001_RUN_033 --n-workers 8
```

Outputs land in `processed_windows/` (or `processed_windows/<sensor>/` in single-sensor mode).

### 4. Train models

Open `train.ipynb` and run all cells. Adjust `SENSOR_ID` and `DATASET_PATH` at the top of the notebook to match your data.

### 5. Use a saved model

```python
import joblib, numpy as np
from features import build_feature_pipeline

scaler = joblib.load("scaler_scalar.joblib")
model  = joblib.load("xgb.joblib")   # or rf.joblib

# raw_windows: np.ndarray of shape (n_windows, 256)
pipeline = build_feature_pipeline(fs=1000.0)
X = scaler.transform(pipeline.transform(raw_windows))
predictions = model.predict(X)   # 0 = healthy, 1 = fault
```

---

## Label Schema

| Value | Meaning |
|-------|---------|
| `0` | Healthy |
| `1` | Pre-fault warning zone (default: 20 s look-ahead) |
| `2` | Active fault |

The training notebook collapses `1` and `2` into a single **fault** class for binary classification.

---

## Dependencies

Core: `numpy`, `pandas`, `scikit-learn`, `scipy`, `PyWavelets`, `xgboost`, `spectrum`

See `requirements.txt` for pinned versions.
