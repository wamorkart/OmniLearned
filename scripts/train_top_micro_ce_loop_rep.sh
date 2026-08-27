#!/bin/bash
# CE-only training loop for a single micro-student replicate run.
#
# Usage:
#   bash train_top_micro_ce_loop_rep.sh train_top_micro_ce_scratch_r1
#
# Same hyperparams/architecture as the KD micro reps (distill_loop_top_micro_rep.sh)
# but with --distill and teacher flags stripped out (see train_top_micro_ce.sh).
# Each salloc grabs 4 nodes x 4 GPUs for 4 hours; resubmits on timeout, exits
# 0 when the 50-epoch run completes.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAVE_TAG="${1:?Usage: $0 <save_tag>  e.g. train_top_micro_ce_scratch_r1}"
export SAVE_TAG

LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/train_top_micro_ce
mkdir -p "$LOG_DIR"

MAX_LOOPS=20
LOOP=0

while [ "$LOOP" -lt "$MAX_LOOPS" ]; do
    LOOP=$((LOOP + 1))
    TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
    LOG_FILE="$LOG_DIR/${SAVE_TAG}_session_${LOOP}_${TIMESTAMP}.out"
    echo "[${TIMESTAMP}] === ${SAVE_TAG} Session ${LOOP} / ${MAX_LOOPS} ===" | tee -a "$LOG_DIR/${SAVE_TAG}_summary.log"

    salloc \
        -C gpu \
        -q interactive \
        -t 240 \
        --nodes 4 \
        --ntasks-per-node 4 \
        --gpus-per-node 4 \
        -A m3246 \
        bash "$SCRIPT_DIR/train_top_micro_ce.sh" \
        2>&1 | tee "$LOG_FILE"
    EXIT="${PIPESTATUS[0]}"

    MSG="[$(date '+%Y-%m-%d %H:%M:%S')] ${SAVE_TAG} Session ${LOOP} exited with code ${EXIT}"
    echo "$MSG" | tee -a "$LOG_DIR/${SAVE_TAG}_summary.log"

    if [ "$EXIT" -eq 0 ]; then
        echo "Training completed. Stopping loop." | tee -a "$LOG_DIR/${SAVE_TAG}_summary.log"
        exit 0
    fi

    echo "Non-zero exit (likely time limit or preemption). Resubmitting in 15s..." \
        | tee -a "$LOG_DIR/${SAVE_TAG}_summary.log"
    sleep 15
done

echo "Reached MAX_LOOPS=${MAX_LOOPS}. Edit MAX_LOOPS and rerun to continue." \
    | tee -a "$LOG_DIR/${SAVE_TAG}_summary.log"
exit 1
