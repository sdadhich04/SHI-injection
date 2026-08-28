#!/usr/bin/env bash
# Run the complete Model SHI workflow on this computer without AR-Burg.
# Safe to rerun: completed jobs/models are verified and skipped.

set -uo pipefail

trap 'echo; echo "Interrupted by user; completed outputs were preserved."; exit 130' INT TERM

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT" || exit 1

PYTHON="${PYTHON:-.venv/bin/python}"
NOISE_REPO="${NOISE_REPO:-../noise_injection_shield}"
WORKERS="${WORKERS:-4}"
HARDWARE_WORKERS="${HARDWARE_WORKERS:-2}"
FEATURE_BATCHES="${FEATURE_BATCHES:-5}"
SOFTWARE_BATCHES="${SOFTWARE_BATCHES:-5}"
HARDWARE_BATCHES="${HARDWARE_BATCHES:-10}"
TRAINING_ROW_CAP_EXPLICIT="${TRAINING_ROW_CAP+x}"
TRAINING_ROW_CAP="${TRAINING_ROW_CAP:-0}"
PLOT_RESULTS="${PLOT_RESULTS:-1}"
HARDWARE_ONLY="${HARDWARE_ONLY:-0}"

DATASET_DIR="${DATASET_DIR:-$NOISE_REPO/outputs/post_noise_injection_1}"
HARDWARE_DIR="${HARDWARE_DIR:-../Data/hardware_injection}"
FEATURES_DIR="${FEATURES_DIR:-model_shi/outputs/no_ar/training_features}"
MODELS_DIR="${MODELS_DIR:-model_shi/models_no_ar}"
SOFTWARE_OUTPUT_DIR="${SOFTWARE_OUTPUT_DIR:-model_shi/outputs/no_ar/software_predictions}"
HARDWARE_OUTPUT_DIR="${HARDWARE_OUTPUT_DIR:-model_shi/outputs/no_ar/hardware_predictions}"
LOG_DIR="${LOG_DIR:-model_shi/outputs/no_ar/logs}"

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_all_no_ar_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

export NOISE_INJECTION_CODE="$NOISE_REPO"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/model_shi:$NOISE_REPO${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/model-shi-mpl}"

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python environment not found or not executable: $PYTHON"
  exit 2
fi
if [[ ! -d "$DATASET_DIR/ground_truth" ]]; then
  echo "ERROR: software dataset not found: $DATASET_DIR"
  exit 2
fi
if [[ ! -d "$HARDWARE_DIR" ]]; then
  echo "ERROR: hardware dataset not found: $HARDWARE_DIR"
  exit 2
fi

# A resumed run must use the same training cap as its existing models. When the
# caller did not explicitly choose a cap, inherit it from the first model.
if [[ -z "$TRAINING_ROW_CAP_EXPLICIT" ]]; then
  existing_metadata="$(find "$MODELS_DIR" -mindepth 2 -maxdepth 2 -name model_metadata.json -print -quit 2>/dev/null)"
  if [[ -n "$existing_metadata" ]]; then
    TRAINING_ROW_CAP="$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["training_config"]["max_rows_per_sensor"])' "$existing_metadata")"
  fi
fi

echo "Model SHI complete run WITHOUT AR-Burg"
echo "Started: $(date --iso-8601=seconds)"
echo "Workers: $WORKERS"
echo "Hardware workers: $HARDWARE_WORKERS"
echo "Training row cap per sensor: $TRAINING_ROW_CAP (0 means every row)"
echo "Plots enabled: $PLOT_RESULTS"
echo "Hardware-only inference: $HARDWARE_ONLY"
echo "Log: $LOG_FILE"

echo
echo "[1/5] Building the deterministic training plan"
"$PYTHON" model_shi/build_training_features.py \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$FEATURES_DIR" \
  --total-batches "$FEATURE_BATCHES" \
  --no-ar-burg --plan-only || exit 1

feature_failed=0
for ((batch = 1; batch <= FEATURE_BATCHES; batch++)); do
  echo
  echo "[1/5] Training features batch $batch/$FEATURE_BATCHES"
  "$PYTHON" model_shi/build_training_features.py \
    --dataset-dir "$DATASET_DIR" \
    --output-dir "$FEATURES_DIR" \
    --total-batches "$FEATURE_BATCHES" \
    --batch "$batch" --workers "$WORKERS" --no-ar-burg || feature_failed=1
done
if ((feature_failed)); then
  echo "ERROR: at least one training-feature job failed."
  echo "Rerun this same script: completed jobs will be skipped and failures retried."
  exit 1
fi

echo
echo "[2/5] Training per-sensor RandomForest and XGBoost models"
"$PYTHON" model_shi/train_models.py \
  --features-dir "$FEATURES_DIR" \
  --models-dir "$MODELS_DIR" \
  --max-rows-per-sensor "$TRAINING_ROW_CAP" \
  --exclude-sensor imu \
  --repeats 3 --workers "$WORKERS" || exit 1

prediction_failed=0
if [[ "$HARDWARE_ONLY" == "1" ]]; then
  echo
  echo "[3/5] Skipping software inference because HARDWARE_ONLY=1"
else
  echo
  echo "[3/5] Building the software-inference plan"
  "$PYTHON" model_shi/predict_synthetic.py \
    --dataset-dir "$DATASET_DIR" \
    --models-dir "$MODELS_DIR" \
    --output-dir "$SOFTWARE_OUTPUT_DIR" \
    --total-batches "$SOFTWARE_BATCHES" --plan-only || exit 1

  for ((batch = 1; batch <= SOFTWARE_BATCHES; batch++)); do
    echo
    echo "[3/5] Software inference batch $batch/$SOFTWARE_BATCHES"
    "$PYTHON" model_shi/predict_synthetic.py \
      --dataset-dir "$DATASET_DIR" \
      --models-dir "$MODELS_DIR" \
      --output-dir "$SOFTWARE_OUTPUT_DIR" \
      --total-batches "$SOFTWARE_BATCHES" \
      --batch "$batch" --workers "$WORKERS" || prediction_failed=1
  done
fi

echo
echo "[4/5] Building the hardware-inference plan"
"$PYTHON" model_shi/predict_hardware.py \
  --input-dir "$HARDWARE_DIR" \
  --models-dir "$MODELS_DIR" \
  --output-dir "$HARDWARE_OUTPUT_DIR" \
  --total-batches "$HARDWARE_BATCHES" --plan-only || exit 1

for ((batch = 1; batch <= HARDWARE_BATCHES; batch++)); do
  echo
  echo "[4/5] Hardware inference batch $batch/$HARDWARE_BATCHES"
  "$PYTHON" model_shi/predict_hardware.py \
    --input-dir "$HARDWARE_DIR" \
    --models-dir "$MODELS_DIR" \
    --output-dir "$HARDWARE_OUTPUT_DIR" \
    --total-batches "$HARDWARE_BATCHES" \
    --batch "$batch" --workers "$HARDWARE_WORKERS" || prediction_failed=1
done

echo
echo "[5/5] Plotting completed predictions"
if [[ "$PLOT_RESULTS" == "1" ]]; then
  if [[ "$HARDWARE_ONLY" != "1" ]]; then
    "$PYTHON" model_shi/plot_model_shi.py "$SOFTWARE_OUTPUT_DIR" || prediction_failed=1
  fi
  "$PYTHON" model_shi/plot_model_shi.py "$HARDWARE_OUTPUT_DIR" || prediction_failed=1
else
  echo "Skipped because PLOT_RESULTS=$PLOT_RESULTS"
fi

echo
echo "Finished: $(date --iso-8601=seconds)"
echo "Log: $LOG_FILE"
echo "Models: $MODELS_DIR"
echo "Software predictions: $SOFTWARE_OUTPUT_DIR"
echo "Hardware predictions: $HARDWARE_OUTPUT_DIR"
if ((prediction_failed)); then
  echo "Some prediction or plotting jobs failed. Rerun this script to retry unfinished work."
  exit 1
fi
echo "All stages completed successfully."
