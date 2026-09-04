# Tier-2 hardware fused SHI

This directory applies the fused SHI used for the draft paper's Tier-2 physical
stress experiments to the ZIP archives in `Data/hardware_injection`. Inputs are
opened read-only and are never converted, renamed, or modified.

## Method and provenance

The continuous branch is a direct port of the feature front end and `_fit_one` /
`_score` equations recovered in `shield-tier1-shi/src/pipeline.py` and
`shield-tier1-shi/src/v3_pipeline.py`:

- 3-second windows with 50% overlap;
- per axis: nine time-domain, eight Symlet-4 level-5 SWT/MODWT-energy, and
  three signal-quality features;
- calibration standardization and removal of features with standard deviation
  at or below `1e-6`;
- Isolation Forest (200 trees, contamination 0.05, seed 0), empirical
  covariance Mahalanobis distance, and EWMA of standardized feature norm
  (`lambda = 0.1`);
- each anomaly component is mapped from calibration median to 0 and calibration
  99th percentile to 1, clipped to `[0, 1]`;
- equal-weight anomaly fusion, `SHI = 1 - mean(component anomaly)`, and a
  sensor-specific alarm threshold at the calibration SHI fifth percentile;
- a detection is persistent after two consecutive windows below the threshold.

This is the predecessor fused SHI that the draft says was used for both the
Tier-1 grid and Tier-2 experiments. It is not canonical BRB-r, which is a later
reliability-weighted formulation and remains under `canonical_brb/`.

The draft describes Tier 2 as a two-branch monitor: continuous SHI plus an
event-rate rule for sparse/binary channels. Neither the draft nor the recovered
source contains the event-rate equation. `EventRateModel` therefore implements
a clearly labelled adaptation for binary vibration: the fraction of state
changes per window is compared with its healthy calibration median, and the
absolute deviation is mapped from calibration median to calibration P95. This
branch is reported separately as `event_health`; it does not alter fused SHI.
An event alarm occurs when that clipped health reaches zero (a deviation at or
beyond the calibration P95 boundary).

## Calibration and fault timing

The runner reuses the hardware archive discovery and protocol rules already
used by the other SHI pipelines. It estimates the sampling rate from timestamps
and uses within-recording calibration because the draft reports that transferred
healthy-session calibration failed on three of four continuous connector
channels.

- A1 shaker, A3 EMI, loose connector, and partial occlusion use the initial
  protocol-defined pre-fault interval (normally five minutes).
- A2 thermal-cold uses the recovery tail retrospectively because exposure is
  already active at recording start.
- supply droop, IMU bias, and accelerated aging have no healthy interval in the
  same recording. Their initial minute is only an exploratory stability proxy
  and every output is marked with that warning.

Protocol-derived fault timestamps and their confidence are metadata, not
automatic fault detections. The plots show them as vertical reference markers.

## Running

From the `SHI-injection` repository root, first verify the complete job plan:

```bash
/home/nicbe/Shield/noise_injection/venv/bin/python \
  tier2_fused_shi/run_hardware.py --plan-only --total-batches 10
```

Run one batch, all batches, or safely rerun after an interruption:

```bash
/home/nicbe/Shield/noise_injection/venv/bin/python \
  tier2_fused_shi/run_hardware.py --batch 1 --total-batches 10 --workers 4

/home/nicbe/Shield/noise_injection/venv/bin/python \
  tier2_fused_shi/run_hardware.py --all-batches --total-batches 10 --workers 4
```

Successful jobs are verified and skipped. An incomplete job directory is moved
under `_interrupted/` before that job is retried, so partial artifacts are
preserved rather than overwritten. Failures do not stop other jobs and are
listed in the per-attempt batch report and `latest_run_summary.json`.

Generate plots after processing:

```bash
/home/nicbe/Shield/noise_injection/venv/bin/python \
  tier2_fused_shi/plot_hardware.py --workers 2
```

The complete sequential BRB plotting -> Tier-2 processing -> Tier-2 plotting
workflow is provided by `run_tmux_sequence.sh`.

## Output layout

```text
outputs/tier2_fused_shi/
├── hardware/
│   ├── batch_plan.json
│   ├── format.json
│   ├── manifest.json
│   ├── batch_001_report.json
│   ├── latest_run_summary.json
│   ├── _interrupted/                         # preserved partial attempts
│   └── <source-archive>/<job-id>/
│       ├── predictions.bin                   # 48-byte fixed records
│       ├── model.joblib                      # fitted within-run detector
│       ├── model_metadata.json               # parameters and calibration
│       └── job_report.json                   # status, timing, alarms, outputs
├── hardware_plots/<source-archive>/<job-id>/current.png
├── summaries/
│   ├── tier2_jobs.csv                       # one fault-aligned row per successful job
│   ├── tier2_summary.json                   # exact-CI aggregates, kept in separate strata
│   ├── tier2_failures.csv                   # detector and aggregation-audit failures
│   ├── tier2_event_branch.csv               # per-job event-branch counts
│   └── tier2_calibration_health.csv         # calibration-only EWMA/SHI diagnostic
└── logs/tier2_sequence_<timestamp>.log
```

Each binary prediction record stores timestamp, fused SHI, the three continuous
component health scores, event health, the continuous threshold, alarm flags,
and up to three signal axes. `format.json` is the machine-readable schema. All
health fields use `[0, 1]`, with 1 meaning most calibration-like and 0 meaning
most anomalous under that component. This scale is methodological output, not a
calibrated probability that a physical sensor is healthy.

## Fault-aligned aggregation

The per-job `first_detection` and `alarm_rates` cover the whole recording and
must not be quoted as onset-aligned latency or detection rate. Generate the
publication-facing aggregation from the deposited predictions instead:

```bash
/home/nicbe/Shield/noise_injection/venv/bin/python scripts/tier2_metrics.py
/home/nicbe/Shield/noise_injection/venv/bin/python scripts/tier2_calibration_health.py
```

The aggregation gives calibration precedence when protocol calibration overlaps
the nominal fault period, then partitions all remaining windows into pre-fault
and post-fault regions. Detection is the first persistent continuous breach at
or after onset and outside calibration. It verifies the 48-byte record size,
the partition count, and all three whole-recording alarm rates against each job
report before it emits aggregate tables. Exact Clopper-Pearson intervals are
used for proportions. Confidence classes, onset-at-start versus delayed-onset
jobs, and physical fault subtypes are kept separate.

The deposited run has a known coverage limitation: the Accelerated Aging data
in the Experiment B archive exists only as raw binary streams. The discovery
logic prefers converted CSVs whenever an archive contains any converted CSV,
so those binary streams were not added to this run's manifest. Consequently,
the current summaries contain no Accelerated Aging result; producing one would
require a detector rerun and is outside aggregation.

## Tests

```bash
PYTHONPATH="$PWD" /home/nicbe/Shield/noise_injection/venv/bin/python \
  -m unittest tier2_fused_shi.test_tier2_fused_shi -v
```

The tests cover the feature schema, score ranges, binary record size, the
event-rate edge case, and a synthetic ZIP -> binary output -> PNG plot flow.

Run the aggregation tests, including validation of every deposited job:

```bash
PYTHONPATH="$PWD" /home/nicbe/Shield/noise_injection/venv/bin/python \
  -m unittest scripts.test_tier2_metrics -v
```
