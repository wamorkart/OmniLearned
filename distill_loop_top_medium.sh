#!/bin/bash
# Loop wrapper for distill_train_top_medium.sh (TAKD stage 1: large->medium).
# Mirrors distill_loop_top_T_sweep.sh's resubmit-on-timeout pattern, just for
# a single fixed a00_b10_T4 medium run instead of a T sweep.
#
# Usage (in a tmux or screen session):
#   bash distill_loop_top_medium.sh

set -euo pipefail

SAVE_TAG=distill_top_medium_scratch_a00_b10_T4

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_sweep/${SAVE_TAG}
mkdir -p "$LOG_DIR"

MAX_LOOPS=50
LOOP=0

echo "TAKD stage 1 (large->medium): SAVE_TAG=$SAVE_TAG"

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
        bash "$SCRIPT_DIR/distill_train_top_medium.sh" \
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
