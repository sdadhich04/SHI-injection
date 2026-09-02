# Canonical BRB-r SHI

This directory adapts the active health-score methodology from
[`GilliamWong/SHIELD-Sensor-Modality`](https://github.com/GilliamWong/SHIELD-Sensor-Modality)
at commit `222dbb1b0d742c7e8ea9b719c841ac5b1d2f2a72` to the Project SHIELD
software- and hardware-injection datasets.

## Method

Each signal is divided into 256-sample windows at a 64-sample stride. Per axis,
the canonical time-domain, Welch-spectrum, Allan-deviation, sym4 MODWT, and
signal-quality features are extracted. A healthy calibration reservoir is used
to rank temporally stable and sensitive features and estimate each feature's
reliability from pairwise observation distances. Mandatory signal-quality
features receive expert weight 1.0, stable features 0.8, and sensitive features
0.5. The upstream BRB-r composite weight is

```text
delta_bar_i = delta_i / max(delta)
omega_i     = delta_bar_i / (1 + delta_bar_i - reliability_i)
raw         = sum(omega_i * abs(z_i)) / sum(omega_i)
```

The healthy 95th percentile defines the calibration envelope. Windows inside
it receive SHI 1; outside it decay toward zero using the upstream bounded
mapping. Thus 1 means within the calibrated healthy envelope and 0 means very
far outside it. This is a health index, not a calibrated probability.

The equations and feature definitions are canonical. Dataset integration is
necessarily adapted: NumPy batches replace pandas DataFrames, calibration uses
a deterministic bounded reservoir for long recordings, and the general SHIELD
256/64 geometry replaces the source repository's fixed 3 s/1.5 s IMU-only
recollection geometry. Every saved model records the source commit and these
adaptations.

Project SHIELD's `vibration` channel is binary rather than a continuous
waveform. Some healthy calibration intervals contain only zeros, making every
continuous feature constant and BRB-r undefined. Those streams use an explicit
`binary_event_health` fallback: the baseline mode is calibrated from healthy
windows and the fraction of opposite-state samples is mapped through the same
healthy-envelope form. One event in an otherwise all-zero 256-sample baseline
maps to SHI 0.5. The model JSON and job report identify this fallback; it is not
reported as BRB-r. Continuous sensors always retain canonical BRB-r.

## Why injection and SHI remain separate

The upstream repository defines reusable fault-injection functions and then
evaluates health separately. Materializing injected recordings is the better
choice here: injection is inexpensive compared with repeated canonical feature
extraction, and the same deterministic injections can be reused across BRB-r,
Random Forest, and Simple SHI experiments. Combining the stages would save some
disk reads once but would regenerate identical noise whenever SHI parameters
change. `run_noise_injection.py` provides one entry point while preserving the
separation.

## Prepare software-injection data

The existing dataset is already prepared. For a new dataset, first create the
noise batch plan, then ground truth, and finally each noise batch:

```bash
python canonical_brb/run_noise_injection.py --plan-only --batch-hours 4
python canonical_brb/run_noise_injection.py --stage ground-truth
python canonical_brb/run_noise_injection.py --stage noise --batch-hours 4 --batch 1
```

The wrapper selects only ZIP files immediately below `Data/`; hardware ZIPs in
`Data/hardware_injection/` are intentionally excluded. The maintained noise
generator currently creates constant and random variants, but canonical BRB-r
software inference reads only `random_0_25` and `random_0_5`.

## Run in batches

Create or inspect deterministic plans without processing:

```bash
python canonical_brb/run_software.py --plan-only --total-batches 10
python canonical_brb/run_hardware.py --plan-only --total-batches 10
```

Run one batch:

```bash
python canonical_brb/run_software.py --batch 1 --total-batches 10 --workers 4
python canonical_brb/run_hardware.py --batch 1 --total-batches 10 --workers 4
```

Run every batch and both datasets:

```bash
python canonical_brb/run_all.py --all-batches --total-batches 10 --workers 4
```

Allan-deviation features are canonical and enabled by default. Add `--no-adev`
for a faster exploratory run; use a different output directory because changing
the feature configuration intentionally invalidates the existing plan.

Jobs are parallelized across files/sensor streams. Writes are atomic, completed
jobs are verified and skipped, and each batch report lists all successes,
skips, and failures. A failed job does not stop other jobs or later batches.
If execution is interrupted, a rerun moves that job's partial files under
`outputs/canonical_brb/.../_interrupted/` before retrying it. Nothing partial is
deleted, and completed jobs remain untouched.

## Plot results

```bash
python canonical_brb/plot.py software \
  --input-dir outputs/canonical_brb/software \
  --output-dir outputs/canonical_brb/software_plots

python canonical_brb/plot.py hardware \
  --input-dir outputs/canonical_brb/hardware \
  --output-dir outputs/canonical_brb/hardware_plots
```

Software plots shade windows containing injected samples. Hardware plots use
the protocol-derived fault timestamp stored in each job report. A hardware
recording marked `initial_proxy` has no known healthy section; its initial
segment is only a relative reference and must not be interpreted as validated
healthy calibration.

## Output layout

```text
outputs/canonical_brb/
├── software/
│   ├── batch_plan.json
│   ├── batch_NNN_report.json
│   └── RUN/.../source/
│       ├── SENSOR__model.json
│       ├── SENSOR__ground_truth.bin
│       ├── SENSOR__random_0_25.bin
│       ├── SENSOR__random_0_5.bin
│       └── job_report.json
└── hardware/
    ├── batch_plan.json
    ├── batch_NNN_report.json
    └── ARCHIVE/JOB_ID/
        ├── model.json
        ├── predictions.bin
        └── job_report.json
```

Software records contain timestamp, SHI, raw health distance, sensitive-feature
degradation score, injected fraction, ground-truth signal, and observed signal.
Hardware records contain timestamp, the same three score fields, and signal.
JSON is used for plans, provenance, calibration parameters, and failure reports;
high-volume timelines remain fixed-width little-endian `.bin` files.
