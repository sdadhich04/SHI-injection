# Project SHIELD Sensor Health Index pipelines

This repository contains the SHI analysis side of Project SHIELD. It keeps four
methodologies separate so their scores and assumptions are not confused:

1. **Simple SHI**, a deterministic ground-truth-referenced distance score;
2. **Model SHI**, the RandomForest/XGBoost pipeline adapted from
   [`samkorostov/shield-model`](https://github.com/samkorostov/shield-model) at
   commit `c094797c96923449b6075d8c483df37856b26712`;
3. **Reconstructed SHIBench reference SHI**, a healthy-calibrated fusion of
   Mahalanobis, Isolation Forest, and EWMA detectors based on the methodology
   named—but not fully specified—in the available paper draft;
4. **Canonical BRB-r SHI**, the active reliability-weighted quality-vector
   implementation from `GilliamWong/SHIELD-Sensor-Modality`, adapted to stream
   the Project SHIELD software- and hardware-injection datasets.

Dataset generation is maintained separately in
[`sdadhich04/noise_injection_shield-`](https://github.com/sdadhich04/noise_injection_shield-).
No SHI pipeline modifies its input recordings.

## Repository layout

- `analyze_simple_shi.py`, `simple_shi_features.py`: software Simple SHI;
- `hardware_simple_shi.py`: baseline-calibrated hardware Simple SHI;
- `plot_*simple_shi*.py`: Simple SHI visualizations;
- `model_shi/`: learned features, training, inference, plots, and tests;
- `model_shi/upstream/`: the exact upstream reference files and commit marker;
- `shibench_reference_shi/`: reconstructed detector, hardware runner, plots,
  equations, parameter choices, and tests.
- `canonical_brb/`: canonical features and BRB-r equations, software/hardware
  batched runners, noise-dataset wrapper, binary plotting, tests, and detailed
  provenance documentation.

All datasets, binary predictions, trained models, plots, virtual environments,
and logs are excluded by `.gitignore`.

## Methodology: Simple SHI

Simple SHI uses 256-sample windows with stride 64. It extracts time-domain,
frequency, stability, and stationary-wavelet/MODWT-like features, with optional
AR-Burg coefficients. Software-injection results compare noisy-window features
with the aligned clean ground truth. Hardware results instead calibrate a
reference distribution from the protocol-defined baseline. The feature
distance is converted to a bounded score where 100 represents closest to the
reference and lower values indicate greater deviation.

This is an interpretable baseline, not a learned probability. Its hardware
result depends strongly on the selected baseline and cannot by itself
distinguish a genuine environmental change from a faulty sensing instrument.

## Methodology: Model SHI

Model SHI reproduces the upstream 256/64 window geometry and extracts 22 scalar
features per axis without AR-Burg: six time, five frequency, one stability, and
ten MODWT-like features. Three-axis sensors therefore use 66 features. Enabling
AR-Burg adds four features per axis (26/78 total) but is substantially slower.

Each sensor receives a `StandardScaler`, Random Forest, and XGBoost model.
Training uses clean ground-truth windows as healthy examples and only the
`random_0_25` and `random_0_5` variants as injected examples. Labels follow the
upstream healthy/pre-fault/active-fault convention and collapse to a binary
healthy-versus-fault target during model fitting. The two reported SHIs are:

```text
SHI_RF  = P_RF(class = healthy | window features)
SHI_XGB = P_XGB(class = healthy | window features)
```

Both range from 0 to 1 and remain separate so model disagreement is visible.
They are classifier estimates, not calibrated physical trust probabilities.
The random split of overlapping windows can overstate validation performance;
entire-run and independent hardware evaluation should be used for conclusions.

## Methodology: reconstructed SHIBench reference SHI

This third pipeline uses the same 22 no-AR-Burg features per axis, but requires
no injected-fault classifier training. It calibrates robust Mahalanobis,
Isolation Forest, and sequential EWMA-residual branches on a healthy segment,
maps each branch to health in `[0, 1]`, estimates reliability weights from the
baseline, and fuses the branches with a weighted geometric mean. It additionally
reports event-rate, liveness, and plausibility alarms without conflating them
with the continuous SHI.

The default healthy fifth percentile maps to the SHI alarm boundary `0.5`,
targeting 5% false positives on the calibration windows. This mapping and all
other defaults are reconstruction choices because the available draft omits
the exact formulas and parameters. See
[`shibench_reference_shi/README.md`](shibench_reference_shi/README.md) for the
equations, commands, artifacts, and limitations.

## Methodology: canonical BRB-r SHI

The canonical implementation is sourced from
[`GilliamWong/SHIELD-Sensor-Modality`](https://github.com/GilliamWong/SHIELD-Sensor-Modality)
at commit `222dbb1b0d742c7e8ea9b719c841ac5b1d2f2a72`. It extracts the upstream
time, Welch-spectrum, Allan-deviation, sym4 MODWT, and signal-quality features.
Features are selected as mandatory, temporally stable, or degradation-sensitive.
The upstream BRB-r equation combines expert importance with healthy-data
reliability, and a healthy calibration envelope maps the weighted feature
distance to SHI `[0, 1]`.

The original code is specific to a small six-axis IMU recollection. The adapter
uses bounded NumPy batches and deterministic calibration reservoirs so it can
process all SHIELD sensor types and 12-hour recordings without loading them
into memory. Models record the source commit and adaptations. Software output
contains ground truth plus `random_0_25` and `random_0_5`; hardware output uses
the same protocol-derived baseline and fault timing metadata as the existing
hardware tools. See [`canonical_brb/README.md`](canonical_brb/README.md) for
equations, commands, binary layouts, caveats, and output structure.

Noise injection remains a separate reusable stage, as in the source repository.
This avoids recomputing deterministic noise whenever SHI settings change and
lets all SHI methods assess identical injected data. The canonical directory
provides a wrapper for preparing the dataset and a one-command runner for all
software and hardware BRB-r batches.

```bash
python canonical_brb/run_all.py --all-batches --total-batches 10 --workers 4
```

## Setup

Clone both repositories beside one another:

```text
workspace/
├── noise_injection_shield/
└── SHI-injection/
```

Then create the SHI environment and run the tests:

```bash
cd SHI-injection
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt

NOISE_INJECTION_CODE=../noise_injection_shield \
PYTHONPATH="$PWD:$PWD/model_shi:../noise_injection_shield" \
  .venv/bin/python -m unittest discover -p 'test_*.py'

PYTHONPATH="$PWD:$PWD/model_shi" \
  .venv/bin/python -m unittest discover model_shi -p 'test_*.py'
```

## Running Model SHI without AR-Burg

The complete workflow is resumable and processes jobs in parallel. Point it to
the ignored datasets rather than copying data into Git:

```bash
NOISE_REPO=../noise_injection_shield \
DATASET_DIR=../noise_injection_shield/outputs/post_noise_injection_1 \
HARDWARE_DIR=../Data/hardware_injection \
TRAINING_ROW_CAP=200000 \
  ./model_shi/run_all_no_ar.sh
```

The runner builds features in batches, trains per-sensor models, performs
software and hardware inference, and generates signal/SHI/fault-time plots.
Completed artifacts are verified and skipped on safe reruns; failures are
recorded and can be retried without recomputing successful jobs.

## Known limitations

The present supervised models learn synthetic random multiplicative-noise
signatures. Physical bias, disconnection, clipping, dropout, thermal, vibration,
and EMI faults can lie outside that training distribution. Ground-truth event
timing and affected-sensor metadata must also be evaluated carefully. These
limitations are why Simple SHI and Model SHI are retained as explicit research
baselines rather than presented as deployment-ready health guarantees.
