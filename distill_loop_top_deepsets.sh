#!/bin/bash
# Continuous training loop for the DeepSets top-tagging KD student
# (distill_top_deepsets_small_scratch_a05_T4_archfix0804), 4 nodes x 4 GPUs
# on the Perlmutter interactive queue. Keeps resubmitting salloc jobs until
# training finishes (exit 0) or MAX_LOOPS is hit.
#
# Run inside a screen session so the loop survives terminal disconnects:
#   screen -dmS distill_deepsets bash distill_loop_top_deepsets.sh
#   screen -r distill_deepsets   # to reattach
#
# Each salloc grabs 4 nodes x 4 GPUs for 4 hours (Perlmutter interactive
# queue max walltime). distill_train_top_deepsets.sh runs with --resuming,
# so each new session picks the checkpoint back up where the last one left
# off. When an allocation expires or crashes (non-zero exit), the loop
# resubmits after a short pause; when training finishes all epochs (exit 0),
# it stops.
#
# The mislabeled prior run (see NOTE in distill_train_top_deepsets.sh) did
# --epoch 50 in ~2h46m on the same 4-node/16-GPU shape, well inside one
# 240-min salloc. MAX_LOOPS below has generous margin for preemption/crash
# restarts, not because one session is expected to be insufficient.
#
# CAUTION: a prior 16-node pretrain run's outer resubmit loop silently died
# after an NCCL crash (no surviving tmux/screen session) and went unnoticed
# for weeks (see distill-lazy-teacher-progress memory). Check summary.log
# periodically (or `screen -r distill_deepsets`) to confirm the loop is
# still alive, not just that a job is queued/running.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_deepsets
mkdir -p "$LOG_DIR"

MAX_LOOPS=8
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
        bash "$SCRIPT_DIR/distill_train_top_deepsets.sh" \
        2>&1 | tee "$LOG_FILE"
    EXIT="${PIPESTATUS[0]}"
    set -e

    MSG="[$(date '+%Y-%m-%d %H:%M:%S')] Session ${LOOP} exited with code ${EXIT}"
    echo "$MSG" | tee -a "$LOG_DIR/summary.log"

    if [ "$EXIT" -eq 0 ]; then
        echo "Training completed. Stopping loop." | tee -a "$LOG_DIR/summary.log"
        exit 0
    fi

    echo "Non-zero exit (likely time limit, preemption, or crash). Resubmitting in 15s..." \
        | tee -a "$LOG_DIR/summary.log"
    sleep 15
done

echo "Reached MAX_LOOPS=${MAX_LOOPS}. Edit MAX_LOOPS and rerun to continue." \
    | tee -a "$LOG_DIR/summary.log"
exit 1
