#!/bin/bash
# Resubmit loop for the atlas_flav ftag TEST-split evaluation, 4 nodes x 4
# GPUs on the Perlmutter interactive queue. Keeps resubmitting salloc jobs
# until the eval exits 0 or MAX_LOOPS is hit. Same pattern as
# train_loop_atlas_flav_ftag_m.sh.
#
# Run inside a screen session so the loop survives disconnects:
#   screen -dmS eval_atlas_flav_ftag_m bash scripts/eval_loop_atlas_flav_ftag_m.sh
#   screen -r eval_atlas_flav_ftag_m    # to reattach
#
# One 1/6 test chunk (~4M jets) on 16 GPUs with bf16 finishes well inside one
# 240-min interactive session, so MAX_LOOPS=3 is only preemption headroom.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/eval_loop_atlas_flav_ftag_m
mkdir -p "$LOG_DIR"

MAX_LOOPS=${MAX_LOOPS:-3}
LOOP=0

while [ "$LOOP" -lt "$MAX_LOOPS" ]; do
    LOOP=$((LOOP + 1))
    TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
    LOG_FILE="$LOG_DIR/session_${LOOP}_${TIMESTAMP}.out"
    echo "[${TIMESTAMP}] === Session ${LOOP} / ${MAX_LOOPS} ===" | tee -a "$LOG_DIR/summary.log"

    set +e
    salloc \
        -C gpu \
        -q interactive \
        -t 240 \
        --nodes 4 \
        --ntasks-per-node 4 \
        --gpus-per-node 4 \
        -A m3246 \
        bash "$SCRIPT_DIR/eval_atlas_flav_ftag_m_interactive.sh" \
        2>&1 | tee "$LOG_FILE"
    EXIT="${PIPESTATUS[0]}"
    set -e

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Session ${LOOP} exited with code ${EXIT}" \
        | tee -a "$LOG_DIR/summary.log"

    if [ "$EXIT" -eq 0 ]; then
        echo "Evaluation completed. Stopping loop." | tee -a "$LOG_DIR/summary.log"
        exit 0
    fi

    echo "Non-zero exit (time limit, preemption, or crash). Resubmitting in 15s..." \
        | tee -a "$LOG_DIR/summary.log"
    sleep 15
done

echo "Reached MAX_LOOPS=${MAX_LOOPS} without a clean exit. Check logs." \
    | tee -a "$LOG_DIR/summary.log"
exit 1
