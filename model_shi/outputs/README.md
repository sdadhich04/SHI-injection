# Model SHI outputs

This directory is the default parent of Model SHI feature shards and
predictions. Only this README is version-controlled; generated binaries,
reports, plots, and models remain ignored by Git.

This document describes methodology and file semantics. It does not evaluate
or interpret the health of individual sensors.

## Pipeline

The current workflow is stored under `no_ar/` and uses 256-sample windows with
stride 64. The timestamp and signal stored for a row belong to the final sample
of its window. No-AR extraction produces 22 features for scalar sensors and 66
for three-axis sensors: time-domain, frequency-domain, stability, and
stationary-wavelet/MODWT-like features.

Software labels are `0=healthy`, `1=within 20 seconds before an injection
onset`, and `2=active injection`. Training collapses labels 1 and 2 to binary
fault class 1. Scenario codes are `0=ground_truth`, `1=random_0_25`, and
`2=random_0_5`.

Each sensor name, including every distinct `proc_*` channel, receives its own
`StandardScaler`, Random Forest, and XGBoost. Training uses a stratified random
80/20 split. Seeds 42–44 produce validation reports; seed-42 models are saved.
The present `models_no_ar` artifacts used at most 200,000 deterministically
sampled, label-aware rows per sensor.

The two SHIs are stored independently:

```text
shi_rf  = RandomForest P(class = healthy | scaled features)
shi_xgb = XGBoost      P(class = healthy | scaled features)
```

Both range from 0 to 1. They are classifier outputs and are not combined or
calibrated into a physical trust probability. Reported classification accuracy
uses `SHI >= 0.5` as the healthy decision.

## Layout

```text
outputs/no_ar/
├── training_features/
│   ├── training_plan.json
│   ├── feature_report_batch_XX_of_YY[_attempt_NN].json
│   └── <run>/<source-stem>/<sensor>.bin + job_report.json
├── software_predictions/
│   ├── software_prediction_plan.json
│   ├── software_report_batch_XX_of_YY[_attempt_NN].json
│   └── <run>/<source-stem>/<sensor>__<scenario>.bin + job_report.json
├── hardware_predictions/
│   ├── hardware_prediction_plan.json
│   ├── hardware_report_batch_XX_of_YY[_attempt_NN].json
│   └── <archive-stem>/<job-id>__<sensor>.bin/.json/.png
└── logs/
```

Plans freeze fingerprints, settings, and deterministic batch assignments.
Batch reports list successful, skipped, and failed jobs; `_attempt_NN` is a
later rerun. Software `job_report.json` and hardware same-stem `.json` files are
completion markers. A `.tmp` is incomplete. A zero-byte `.bin` contains no
complete 256-sample window and is skipped by the plotter.

Hardware job IDs disambiguate identically named files. Their JSON sidecars map
the ID to archive/member/sensor and store the protocol-derived intervention
timestamp. That timestamp is separate metadata, not inferred from SHI.

## Training feature record

Feature shards are headerless little-endian records:

| Field | Type |
|---|---:|
| `timestamp_ms` | `<u8` |
| `label` | `u1` |
| `scenario` | `u1` |
| `reserved` | `<u2` |
| `features[N]` | `<f4` × 22 or 66 |

They occupy 100 bytes for scalar sensors and 276 bytes for three-axis sensors.
They are model-training intermediates, not SHI predictions.

## Prediction record

Software and hardware `.bin` predictions use `PREDICTION_DTYPE` from
`model_shi/common.py`, a headerless little-endian 32-byte record:

| Field | Type | Methodological meaning |
|---|---:|---|
| `timestamp_ms` | `<u8` | Window endpoint timestamp |
| `shi_rf` | `<f4` | Random Forest healthy-class probability |
| `shi_xgb` | `<f4` | XGBoost healthy-class probability |
| `ground_truth_label` | `u1` | Software label 0/1/2; 255 for hardware |
| `scenario` | `u1` | Software scenario 0/1/2; 255 for hardware |
| `reserved` | `<u2` | Currently zero |
| `signal[3]` | `<f4` × 3 | Signal at the window endpoint |

Scalar sensors use `signal[0]`; unused axes are zero. The file size must be a
multiple of 32 and match the job metadata.

```python
import sys
import numpy as np

sys.path.insert(0, "model_shi")
from common import PREDICTION_DTYPE

records = np.memmap("path/to/prediction.bin",
                    dtype=PREDICTION_DTYPE, mode="r")
```

Plots display the endpoint signal and both SHIs. Software fault lines come from
transitions into label 2; hardware fault lines come from the JSON protocol
timestamp. Display downsampling does not alter the binary data.

The scaler and models live outside this output tree in
`models_no_ar/<sensor>/`. Preserve them with the results. Their
`model_metadata.json`, along with plans and job reports, is the authoritative
record of feature order and settings. Absolute report paths may need to be
resolved locally if the workspace is moved.
