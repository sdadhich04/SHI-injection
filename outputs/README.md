# Simple SHI outputs

This directory is the default parent of generated Simple SHI artifacts. Only
this README is version-controlled; generated binaries, reports, plots, and
datasets remain ignored by Git.

This document describes file structure and methodology. It does not evaluate
or interpret the health of any recorded sensor.

## Method

`analyze_simple_shi.py` compares aligned ground-truth and software-injected
signals in 256-sample windows with stride 64. A record's timestamp and signal
values belong to the final sample of its window.

The default pipeline extracts 22 features per axis: six time-domain, five
frequency-domain, one stability, and ten stationary-wavelet/MODWT-like
features. Three-axis feature vectors are concatenated. AR-Burg is disabled in
the default batched run.

For each feature, a scale is calculated across the corresponding complete
clean source:

```text
scale = max(sample_std(clean), 0.10 * mean(abs(clean)), numerical_floor)
normalized_difference = min(abs(injected - clean) / scale, 10)
feature_distance = RMS(normalized_difference)
SHI = 100 * exp(-feature_distance)
```

Windows containing no changed samples are assigned `SHI = 100`. This SHI is a
deterministic ground-truth-referenced distance on a 0–100 scale, not a learned
probability. The batched pipeline processes only `random_0_25` and
`random_0_5`.

## Layout

```text
outputs/simple_shi_batches/
├── batch_plan.json
├── processing_report_batch_XX_of_YY[_attempt_NN].json
└── <run>/<source-file-stem>/
    ├── job_report.json
    ├── <sensor>__random_0_25.bin
    ├── <sensor>__random_0_5.bin
    └── matching PNGs when requested
```

`batch_plan.json` freezes the input fingerprint and batch assignment. Batch
reports list successful, skipped, and failed jobs. Later reruns receive
`_attempt_NN` rather than overwriting history. `job_report.json` is the
completion marker for a source and records settings, rows, bytes, timestamp
bounds, and summaries. A `.tmp` is incomplete and must not be consumed.

Names beginning with `proc_` are separate processed channels inherited from
the input. `_12h` is also part of an input run name; it is not added by SHI.

## Binary record

Every `.bin` is a headerless sequence of little-endian 44-byte records using
`BINARY_DTYPE` from `analyze_simple_shi.py`:

| Field | Type | Methodological meaning |
|---|---:|---|
| `timestamp_ms` | `<u8` | Window endpoint timestamp |
| `shi` | `<f4` | 0–100 score from the formula above |
| `feature_distance` | `<f4` | Distance before exponential mapping |
| `injected_fraction` | `<f4` | Fraction of window samples changed from ground truth |
| `ground_truth[3]` | `<f4` × 3 | Clean signal at the window endpoint |
| `observed[3]` | `<f4` × 3 | Injected signal at the window endpoint |

Scalar sensors use array element zero; unused axes are zero. A valid file size
is a positive multiple of 44 and must agree with its `job_report.json`.

```python
import numpy as np
from analyze_simple_shi import BINARY_DTYPE

records = np.memmap("path/to/result.bin", dtype=BINARY_DTYPE, mode="r")
```

`plot_simple_shi_bin.py` reads this format directly. It plots the clean and
observed endpoint signals, the SHI, and shading derived from
`injected_fraction`; it does not convert or modify the binary.

Absolute paths inside reports describe the generation machine. After moving a
workspace, use the report's local directory and filename. The authoritative
configuration is `processing_config`, not the directory name alone.
