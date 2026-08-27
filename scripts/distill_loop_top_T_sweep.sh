#!/bin/bash
# Training loop for the temperature sweep (α=0, β=1, variable T) on top tagging.
#
# Wraps distill_train_top_sweep.sh with α=0.0, β=1.0 fixed.
# Each salloc grabs 4 nodes × 4 GPUs for 4 hours; resubmits on timeout.
#
# Usage (in a tmux or screen session):
#   bash distill_loop_top_T_sweep.sh 1        # T=1
#   bash distill_loop_top_T_sweep.sh 8        # T=8
#   bash distill_loop_top_T_sweep.sh 4 r1     # T=4, rep 1

set -euo pipefail

DISTILL_T=${1:?Usage: $0 T [rep_suffix]}
REP_SUFFIX=${2:-}

TAG_SUFFIX=""
[ -n "$REP_SUFFIX" ] && TAG_SUFFIX="_${REP_SUFFIX}"

export ALPHA=0.0
export BETA=1.0
export DISTILL_T
export SAVE_TAG="distill_top_small_scratch_a00_b10_T${DISTILL_T}${TAG_SUFFIX}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_sweep/${SAVE_TAG}
mkdir -p "$LOG_DIR"

MAX_LOOPS=50
LOOP=0

echo "T-sweep run: ALPHA=$ALPHA BETA=$BETA T=$DISTILL_T SAVE_TAG=$SAVE_TAG"

while [ "$LOOP" -lt "$MAX_LOOPS" ]; do
    LOOP=$((LOOP + 1))
    TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
    LOG_FILE="$LOG_DIR/session_${LOOP}_${TIMESTAMP}.out"
    echo "[${TIMESTAMP}] === Session ${LOOP} / ${MAX_LOOPS} (${SAVE_TAG}) ===" | tee -a "$LOG_DIR/summary.log"

    salloc \
        -C gpu \
        -q interactive \
        -t 240 \
        --nodes 4 \
        --ntasks-per-node 4 \
        --gpus-per-node 4 \
        -A m3246 \
        bash "$SCRIPT_DIR/distill_train_top_sweep.sh" \
        2>&1 | tee "$LOG_FILE"
    EXIT="${PIPESTATUS[0]}"

    MSG="[$(date '+%Y-%m-%d %H:%M:%S')] Session ${LOOP} exited with code ${EXIT}"
    echo "$MSG" | tee -a "$LOG_DIR/summary.log"

    if [ "$EXIT" -eq 0 ]; then
        echo "Training completed: ${SAVE_TAG}. Stopping loop." | tee -a "$LOG_DIR/summary.log"
        exit 0
    fi

    echo "Non-zero exit (likely time limit or preemption). Resubmitting in 15s..." \
        | tee -a "$LOG_DIR/summary.log"
    sleep 15
done

echo "Reached MAX_LOOPS=${MAX_LOOPS} for ${SAVE_TAG}." | tee -a "$LOG_DIR/summary.log"
exit 1
