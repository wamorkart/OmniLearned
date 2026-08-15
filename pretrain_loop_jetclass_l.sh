#!/bin/bash
# Continuous pretraining loop for the large PET model on JetClass, 16 nodes x
# 4 GPUs (64 GPUs) on the Perlmutter interactive queue. Keeps resubmitting
# salloc jobs until training finishes (exit 0) or MAX_LOOPS is hit.
#
# Run inside a tmux/screen session so the loop survives terminal disconnects:
#   tmux new-session -s pretrain_jc_l 'bash pretrain_loop_jetclass_l.sh'
#
# Each salloc grabs 16 nodes x 4 GPUs for 4 hours (Perlmutter interactive
# queue max walltime). pretrain_train_jetclass_l.sh runs with --resuming, so
# each new session picks the checkpoint back up where the last one left off.
# When an allocation expires or crashes (non-zero exit), the loop resubmits
# after a short pause; when training finishes all epochs (exit 0), it stops.
#
# CAUTION: a prior 16-node pretrain run's outer resubmit loop silently died
# after an NCCL crash (no surviving tmux/screen session) and went unnoticed
# for weeks (see distill-lazy-teacher-progress memory). Check summary.log
# periodically (or `tmux attach -t pretrain_jc_l`) to confirm the loop is
# still alive, not just that a job is queued/running.
#
# --epoch 3125 (4 passes over JetClass) x ~499s/epoch (measured) needs ~108
# sessions minimum at ~29 epochs/session; MAX_LOOPS below has ~20% margin.
# Safety: MAX_LOOPS caps at 130 sessions (~8320 node-hours at 4 nodes/4h).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/pretrain_loop_jetclass_l
mkdir -p "$LOG_DIR"

MAX_LOOPS=130
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
        bash "$SCRIPT_DIR/pretrain_train_jetclass_l.sh" \
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
