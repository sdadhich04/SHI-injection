# Project SHIELD Sensor Health Index (SHI) pipelines

This repository contains analysis pipelines that score Project SHIELD sensor
recordings. It keeps several SHI approaches separate so their outputs and
assumptions can be compared without treating them as the same metric. The
repository expects Project SHIELD software-injection and hardware-injection
data to be supplied locally; those datasets and generated outputs are not
tracked here.

## What it does

- **Simple SHI** (`analyze_simple_shi.py`, `hardware_simple_shi.py`) extracts
  windowed signal features and scores deviation from either aligned clean data
  or a hardware-recording baseline.
- **Model SHI** (`model_shi/`) builds features, trains per-sensor Random Forest
  and XGBoost classifiers, performs software and hardware inference, and plots
  the resulting predictions.
- **Reconstructed SHIBench reference SHI** (`shibench_reference_shi/`) fuses
  Mahalanobis, Isolation Forest, and EWMA-based detector branches calibrated on
  a healthy segment.
- **Canonical BRB-r SHI** (`canonical_brb/`) runs a reliability-weighted,
  feature-based scorer on software-injected and hardware-injection data.
- **Tier-2 fused SHI** (`tier2_fused_shi/`) processes hardware-stress archives
  with a fused Isolation Forest, Mahalanobis, and EWMA score. Its separate
  binary event-rate branch is a documented adaptation because the recovered
  source does not include that branch's exact equation.

The scripts batch work, write binary prediction artifacts plus JSON metadata,
and include plotting and unit-test modules. They do not modify input
recordings.

## Tools and inputs

The code is Python and uses the packages listed in
[`requirements.txt`](requirements.txt): NumPy, SciPy, scikit-learn, XGBoost,
PyWavelets, Spectrum, Matplotlib, and Joblib. The runners expect locally
available Project SHIELD data, including software-injection outputs from the
separate `noise_injection_shield` repository and hardware-injection archives.
The source identifies those archives as hardware inputs but does not document a
specific device model.

Create an environment and install the dependencies:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The commands below are written for a POSIX shell, as is the included Model SHI
orchestrator. Set the data paths for your local checkout before running:

```bash
# Run the Model SHI workflow (feature building, training, inference, and plots).
NOISE_REPO=../noise_injection_shield \
DATASET_DIR=../noise_injection_shield/outputs/post_noise_injection_1 \
HARDWARE_DIR=../Data/hardware_injection \
  ./model_shi/run_all_no_ar.sh

# Inspect a Tier-2 hardware run plan without processing data.
.venv/bin/python tier2_fused_shi/run_hardware.py \
  --input-dir ../Data/hardware_injection --plan-only --total-batches 10

# Run all canonical BRB-r software and hardware batches.
.venv/bin/python canonical_brb/run_all.py \
  --all-batches --total-batches 10 --workers 4
```

Run the included tests after setting the repository paths used by the modules:

```bash
NOISE_INJECTION_CODE=../noise_injection_shield \
PYTHONPATH="$PWD:$PWD/model_shi:../noise_injection_shield" \
  .venv/bin/python -m unittest discover -p 'test_*.py'
```

## Credits and provenance

`model_shi/upstream/` preserves the referenced upstream model files and commit
marker. `canonical_brb/` records its source as
[`GilliamWong/SHIELD-Sensor-Modality`](https://github.com/GilliamWong/SHIELD-Sensor-Modality),
and the Model SHI code records its referenced upstream commit in
`model_shi/upstream/UPSTREAM_COMMIT`. See the method-specific READMEs for
equations, parameters, output layouts, and limitations.
