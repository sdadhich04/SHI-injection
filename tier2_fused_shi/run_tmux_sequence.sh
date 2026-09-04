#!/usr/bin/env bash
# Run BRB plotting, Tier-2 fused processing, then Tier-2 plotting sequentially.

set -u -o pipefail

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/../.." && pwd)"
SESSION_NAME="${SESSION_NAME:-shield-shi-tier2}"
DRY_RUN="${DRY_RUN:-0}"

if [[ "${1:-}" != "--inside" && -z "${TMUX:-}" && "$DRY_RUN" != "1" ]]; then
    if ! command -v tmux >/dev/null 2>&1; then
        echo "tmux is not installed or not on PATH" >&2
        exit 2
    fi
    if tmux has-session -t "=$SESSION_NAME" 2>/dev/null; then
        echo "tmux session '$SESSION_NAME' already exists." >&2
        echo "Attach with: tmux attach -t $SESSION_NAME" >&2
        exit 2
    fi
    printf -v INNER_COMMAND '%q %q %q' /bin/bash "$SCRIPT_PATH" --inside
    tmux new-session -d -s "$SESSION_NAME" "$INNER_COMMAND"
    echo "Started tmux session: $SESSION_NAME"
    echo "Attach: tmux attach -t $SESSION_NAME"
    echo "The log path is printed at the top of the tmux session."
    exit 0
fi

if [[ "${1:-}" == "--inside" ]]; then
    shift
fi
if [[ $# -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 2
fi

PYTHON="${SHIELD_PYTHON:-$WORKSPACE_ROOT/noise_injection/venv/bin/python}"
TOTAL_BATCHES="${TOTAL_BATCHES:-10}"
PROCESS_WORKERS="${PROCESS_WORKERS:-4}"
PLOT_WORKERS="${PLOT_WORKERS:-2}"
MAX_PLOT_POINTS="${MAX_PLOT_POINTS:-5000}"
BRB_INPUT="${BRB_INPUT:-$REPO_ROOT/outputs/canonical_brb/hardware}"
BRB_PLOTS="${BRB_PLOTS:-$REPO_ROOT/outputs/canonical_brb/hardware_plots}"
TIER2_INPUT="${TIER2_INPUT:-$WORKSPACE_ROOT/Data/hardware_injection}"
TIER2_OUTPUT="${TIER2_OUTPUT:-$REPO_ROOT/outputs/tier2_fused_shi/hardware}"
TIER2_PLOTS="${TIER2_PLOTS:-$REPO_ROOT/outputs/tier2_fused_shi/hardware_plots}"
LOG_DIR="$REPO_ROOT/outputs/tier2_fused_shi/logs"
mkdir -p "$LOG_DIR"
LOG_PATH="$LOG_DIR/tier2_sequence_$(date +%Y%m%d_%H%M%S).log"

if [[ "$DRY_RUN" != "1" ]]; then
    exec > >(tee -a "$LOG_PATH") 2>&1
fi

echo "Sequence started: $(date --iso-8601=seconds)"
echo "Repository: $REPO_ROOT"
echo "Python: $PYTHON"
echo "Log: $LOG_PATH"

if [[ ! -x "$PYTHON" ]]; then
    echo "Python environment is missing or not executable: $PYTHON" >&2
    exit 2
fi
if [[ ! -d "$TIER2_INPUT" ]]; then
    echo "Tier-2 input directory is missing: $TIER2_INPUT" >&2
    exit 2
fi
if [[ ! -d "$BRB_INPUT" ]]; then
    echo "Canonical BRB output directory is missing: $BRB_INPUT" >&2
    exit 2
fi
if ! find "$BRB_INPUT" -type f -name predictions.bin -print -quit | grep -q .; then
    echo "No canonical BRB predictions.bin files found under: $BRB_INPUT" >&2
    exit 2
fi

if [[ "$DRY_RUN" != "1" ]]; then
    "$PYTHON" -c "import joblib, matplotlib, numpy, pywt, scipy, sklearn" || exit 2
    if command -v flock >/dev/null 2>&1; then
        exec 9>"$LOG_DIR/tier2_sequence.lock"
        if ! flock -n 9; then
            echo "Another Tier-2 sequence already holds $LOG_DIR/tier2_sequence.lock" >&2
            exit 2
        fi
    fi
fi

cd "$REPO_ROOT" || exit 2
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/shield-shi-mpl}"
mkdir -p "$MPLCONFIGDIR"

run_stage() {
    local label="$1"
    shift
    echo
    echo "[$(date --iso-8601=seconds)] START $label"
    printf 'Command:'
    printf ' %q' "$@"
    printf '\n'
    if [[ "$DRY_RUN" == "1" ]]; then
        echo "DRY RUN: not executed"
        return 0
    fi
    "$@"
    local status=$?
    echo "[$(date --iso-8601=seconds)] END $label (exit=$status)"
    return "$status"
}

overall=0
run_stage "1/3 canonical BRB hardware plotting" \
    "$PYTHON" canonical_brb/plot.py hardware \
    --input-dir "$BRB_INPUT" --output-dir "$BRB_PLOTS" \
    --max-points "$MAX_PLOT_POINTS" || overall=1

run_stage "2/3 Tier-2 fused SHI processing" \
    "$PYTHON" tier2_fused_shi/run_hardware.py \
    --input-dir "$TIER2_INPUT" --output-dir "$TIER2_OUTPUT" \
    --all-batches --total-batches "$TOTAL_BATCHES" \
    --workers "$PROCESS_WORKERS" || overall=1

# Plot whatever completed successfully even if individual Tier-2 jobs failed.
run_stage "3/3 Tier-2 fused SHI plotting" \
    "$PYTHON" tier2_fused_shi/plot_hardware.py \
    --input-dir "$TIER2_OUTPUT" --output-dir "$TIER2_PLOTS" \
    --workers "$PLOT_WORKERS" --max-points "$MAX_PLOT_POINTS" || overall=1

echo
echo "Sequence finished: $(date --iso-8601=seconds)"
echo "Overall exit status: $overall"
echo "Log: $LOG_PATH"
echo "BRB plots: $BRB_PLOTS"
echo "Tier-2 data: $TIER2_OUTPUT"
echo "Tier-2 plots: $TIER2_PLOTS"
exit "$overall"

