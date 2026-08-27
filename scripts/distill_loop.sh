#!/bin/bash
# Continuous distillation training loop for Perlmutter interactive queue.
#
# Run inside a screen session so the loop survives terminal disconnects:
#   screen -dmS distill_pretrain bash /path/to/distill_loop.sh
#
# Each salloc grabs 4 nodes × 4 GPUs (16 GPUs) for 4 hours.
# When the allocation expires (non-zero exit), the loop resubmits.
# When training finishes all epochs (exit 0), the loop stops.
#
# Safety: MAX_LOOPS caps at 90 sessions (~360 node-hours, ~15 days back-to-back)
# to comfortably cover a 10-day unattended window with margin for queue waits.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop
mkdir -p "$LOG_DIR"

MAX_LOOPS=90
LOOP=0

while [ "$LOOP" -lt "$MAX_LOOPS" ]; do
    LOOP=$((LOOP + 1))
    TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
    LOG_FILE="$LOG_DIR/session_${LOOP}_${TIMESTAMP}.out"
    echo "[${TIMESTAMP}] === Session ${LOOP} / ${MAX_LOOPS} ===" | tee -a "$LOG_DIR/summary.log"

    # salloc runs bash distill_train.sh on the submit node;
    # the srun inside distill_train.sh steps into the 4-node allocation.
    #
    # set +e/-e around the pipeline: with `set -e` + `pipefail` (both on
    # above), a non-zero salloc exit makes the whole pipeline non-zero and
    # -e kills the script right here, before EXIT is even read -- so the
    # resubmit logic below never runs. This exact bug (see
    # pretrain_loop_jetclass_l.sh) silently killed the loop on the Jul 3
    # NCCL crash and let the run sit dead, unnoticed, for weeks.
    set +e
    salloc \
        -C gpu \
        -q interactive \
        -t 240 \
        --nodes 4 \
        --ntasks-per-node 4 \
        --gpus-per-node 4 \
        -A m3246 \
        bash "$SCRIPT_DIR/distill_train.sh" \
        2>&1 | tee "$LOG_FILE"
    EXIT="${PIPESTATUS[0]}"
    set -e

    MSG="[$(date '+%Y-%m-%d %H:%M:%S')] Session ${LOOP} exited with code ${EXIT}"
    echo "$MSG" | tee -a "$LOG_DIR/summary.log"

    if [ "$EXIT" -eq 0 ]; then
        echo "Training completed. Stopping loop." | tee -a "$LOG_DIR/summary.log"
        exit 0
    fi

    echo "Non-zero exit (likely time limit or preemption). Resubmitting in 15s..." \
        | tee -a "$LOG_DIR/summary.log"
    sleep 15
done

echo "Reached MAX_LOOPS=${MAX_LOOPS}. Edit MAX_LOOPS and rerun to continue." \
    | tee -a "$LOG_DIR/summary.log"
exit 1
