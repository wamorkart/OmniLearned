#!/bin/bash
# Continuous training loop for a DeepSets top-tagging KD run. Keeps
# resubmitting 4-node x 4-GPU interactive salloc jobs until training exits 0
# or MAX_LOOPS is hit. distill_train_top_deepsets.sh runs with --resuming, so
# each session picks up where the last left off.
#
#   CONFIG=<name> [SAVE_TAG=...] [MAX_LOOPS=N] \
#     screen -dmS distill_deepsets_<name> bash distill_loop_top_deepsets.sh
#   screen -r distill_deepsets_<name>   # to reattach
#
# CONFIG is passed straight through to distill_train_top_deepsets.sh (see its
# header for the table; default a05). SAVE_TAG overrides the checkpoint tag
# for replicate runs. Always launch inside screen so the loop survives a
# disconnect.
#
# CAUTION: a prior outer resubmit loop silently died after an NCCL crash with
# no surviving screen session and went unnoticed for weeks (see
# distill-lazy-teacher-progress memory). Check summary.log (or reattach)
# periodically to confirm the loop itself is alive, not just that a job is
# queued.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONFIG="${CONFIG:-${1:-a05}}"
export CONFIG
[ -n "${SAVE_TAG:-}" ] && export SAVE_TAG

LABEL="${SAVE_TAG:-$CONFIG}"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_deepsets_$LABEL
mkdir -p "$LOG_DIR"

MAX_LOOPS="${MAX_LOOPS:-8}"
LOOP=0

while [ "$LOOP" -lt "$MAX_LOOPS" ]; do
    LOOP=$((LOOP + 1))
    TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
    LOG_FILE="$LOG_DIR/session_${LOOP}_${TIMESTAMP}.out"
    echo "[${TIMESTAMP}] === $LABEL Session ${LOOP} / ${MAX_LOOPS} ===" | tee -a "$LOG_DIR/summary.log"

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

    MSG="[$(date '+%Y-%m-%d %H:%M:%S')] $LABEL Session ${LOOP} exited with code ${EXIT}"
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
