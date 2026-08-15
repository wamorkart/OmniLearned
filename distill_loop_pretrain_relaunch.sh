#!/bin/bash
# Short VALIDATION loop for the full-pretrain-dataset distillation relaunch
# (2026-08-05) -- see distill_train_pretrain_relaunch.sh for the tag/epoch
# rationale. Same 4-node/16-GPU interactive shape as distill_loop.sh, just a
# small MAX_LOOPS since --epoch 10 (~3.85h at the measured 1377s/epoch) should
# finish inside a single 4h session; a few extra sessions of headroom in case
# of preemption before it needs a human to look at the result.
#
# Run inside a screen session so the loop survives terminal disconnects:
#   screen -dmS distill_pretrain_relaunch bash distill_loop_pretrain_relaunch.sh
#
# set +e/-e around the salloc|tee pipeline: with `set -e` + `pipefail` (both
# on below), a non-zero salloc exit makes the whole pipeline non-zero and -e
# kills the script right there, before EXIT is read -- so the resubmit logic
# never runs. This exact bug in the original distill_loop.sh silently killed
# the outer loop on the Jul 3 NCCL crash and let the run sit dead, unnoticed,
# for weeks. Fixed here from the start (and backported to distill_loop.sh).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_pretrain_relaunch
mkdir -p "$LOG_DIR"

MAX_LOOPS=3
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
        bash "$SCRIPT_DIR/distill_train_pretrain_relaunch.sh" \
        2>&1 | tee "$LOG_FILE"
    EXIT="${PIPESTATUS[0]}"
    set -e

    MSG="[$(date '+%Y-%m-%d %H:%M:%S')] Session ${LOOP} exited with code ${EXIT}"
    echo "$MSG" | tee -a "$LOG_DIR/summary.log"

    if [ "$EXIT" -eq 0 ]; then
        echo "Validation run completed (epoch 10 reached). Stopping loop." | tee -a "$LOG_DIR/summary.log"
        exit 0
    fi

    echo "Non-zero exit (likely time limit, preemption, or crash). Resubmitting in 15s..." \
        | tee -a "$LOG_DIR/summary.log"
    sleep 15
done

echo "Reached MAX_LOOPS=${MAX_LOOPS} without a clean exit -- check logs in $LOG_DIR." \
    | tee -a "$LOG_DIR/summary.log"
exit 1
