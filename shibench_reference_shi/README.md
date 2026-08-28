# Reconstructed SHIBench reference SHI

This directory implements a **configurable reconstruction**, not an exact
reproduction, of the reference SHI described in `alperen-paper-v3 (2).docx`.
The available draft names the detector branches and evaluation principles, but
does not contain the equations or parameter values from its referenced
methodology sections. Consequently, every material choice below is explicit in
the code and is saved with each result so it can be revised later.

The reconstruction combines the three continuous branches named by the draft:

1. robust Mahalanobis distance (Ledoit-Wolf covariance);
2. Isolation Forest anomaly score;
3. the norm of the residual from a sequential EWMA feature estimate.

It also implements the draft's separate event-rate rule and complementary
liveness/range checks. The feature vector is the project's existing no-AR-Burg
family: 22 time, frequency, stability, and MODWT-like features per signal axis.

## Score construction

The model is fitted only on a recording's protocol-defined healthy baseline.
Features are standardized using that baseline. For each continuous branch
`j`, let `a_j` be its anomaly score, `m_j` the healthy median, and `t_j` the
healthy 95th percentile. It is converted to health by

```text
h_j(a) = 1 / (1 + exp(log(9) * (a - t_j) / max(t_j - m_j, eps)))
```

Thus `h_j(t_j) = 0.5` and `h_j(m_j) = 0.9`. Branch reliability is the inverse
standard deviation of its healthy health scores, with a floor and a 3:1 cap.
The uncalibrated fusion is the reliability-weighted geometric mean:

```text
H_raw = exp(sum_j w_j * log(max(h_j, 1e-6)))
```

`H_raw` is calibrated once more so its healthy fifth percentile maps to 0.5
and its healthy median maps to 0.9. The reported SHI is therefore on the
paper's 0-to-1 health scale: values near 1 are baseline-like and values below
0.5 raise the continuous alarm. On the calibration samples, the default target
false-positive rate is 5%; this is not a guarantee on later data.

The event branch learns the baseline 99.5th percentile of sample-to-sample
jumps, counts how often that threshold is exceeded in each window, and maps
the rate to a separate 0-to-1 health score. In `auto` mode it is enabled for
vibration and sparse/discrete channels. Liveness flags windows with at least
98% identical transitions. The plausibility flag uses a robust baseline
envelope of 20 scale units around the median. These checks remain separate
flags rather than silently changing the continuous SHI.

## Hardware calibration

Calibration is performed independently within each recording:

- `pre_fault`: from recording start to the documented intervention time;
- `tail_recovery`: from the final healthy/recovery interval;
- `initial_proxy`: the initial interval when no known healthy segment exists.

`initial_proxy` is marked with a warning because it may already contain the
physical fault. This strategy is intentionally conservative about the paper's
reported cross-session/device transfer limitation, but it can hide a persistent
fault present throughout calibration.

## Run

From the `SHI-injection` repository, with the data repository available beside
it:

```bash
export PYTHONPATH="$PWD:$PWD/model_shi:../noise_injection_shield"
export NOISE_INJECTION_CODE=../noise_injection_shield

python -m shibench_reference_shi.run_hardware \
  --input-dir ../Data/hardware_injection \
  --output-dir outputs/shibench_reference_shi/hardware \
  --total-batches 10 --plan-only

for batch in {1..10}; do
  python -m shibench_reference_shi.run_hardware \
    --input-dir ../Data/hardware_injection \
    --output-dir outputs/shibench_reference_shi/hardware \
    --total-batches 10 --batch "$batch" --workers 2 || true
done

python -m shibench_reference_shi.plot_hardware \
  outputs/shibench_reference_shi/hardware
```

Jobs are balanced by uncompressed input size and isolated from one another.
Successful jobs are safely skipped on rerun; failures are retained in a batch
report. Use the same output directory only with exactly the same plan and
settings. Filtering options (`--archive-contains`, `--member-contains`, and
repeatable `--sensor`) allow focused trials. `--fault-start-seconds` overrides
the protocol-derived intervention time for selected data.

The main parameters can be changed without editing code:

```text
--calibration-fpr 0.05
--ewma-alpha 0.15
--isolation-estimators 64
--minimum-calibration-windows 20
--event-branch auto|all|none
--plausibility-mad-multiplier 20
```

Use a new output directory when changing parameters, preserving prior results
for comparison.

## Artifacts

Each sensor stream produces:

- `.bin`: fixed-width, little-endian prediction records;
- `.json`: input/protocol details, all thresholds and parameters, reliability
  weights, calibration summary, alarm counts, and paths;
- `.calibration.joblib`: the fitted within-recording detector;
- `.png`: signal, component health, fused SHI, and documented fault start.

The binary NumPy dtype is `REFERENCE_DTYPE` in `core.py` and occupies 48 bytes:
timestamp, fused SHI, three component health scores, event health, flag byte,
reserved bytes, and up to three signal axes. Flag bits are continuous `1`,
event `2`, liveness `4`, and plausibility `8`.

## Interpretation limits

This reconstruction is useful for experiments and parameter studies, not yet a
validated trust probability. Within-recording calibration, overlapping
windows, uncertain intervention timestamps, and incomplete baseline segments
must be considered when reporting performance. The most important next step is
to compare pre/post-fault SHI distributions and detection delays across fault
types, then replace reconstructed formulas or defaults if the paper authors
provide the missing reference implementation.
