#!/bin/bash
# Continuous loop for the OmniFold pythia-vs-herwig unfolding run. Wraps
# train_omnifold_pythia_herwig.sh with salloc auto-resubmit on timeout --
# each Step1/Step2 fit is independently resumable (src/omnilearned/omnifold.py
# checks for a last-checkpoint / a completed training_*.json before starting
# a step), so a session that times out mid-step picks back up cleanly on the
# next resubmit, same pattern as the pretrain-KD loops in this repo.
#
# Run in a screen session per [[feedback-launch-trainings-in-screen]]:
#   screen -dmS omnifold_pythia_herwig bash distill_loop_omnifold.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_omnifold
mkdir -p "$LOG_DIR"

SAVE_TAG=omnifold_pythia_herwig
MAX_LOOPS=40
LOOP=0

echo "OmniFold loop: $SAVE_TAG"

while [ "$LOOP" -lt "$MAX_LOOPS" ]; do
    LOOP=$((LOOP + 1))
    TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
    LOG_FILE="$LOG_DIR/session_${LOOP}_${TIMESTAMP}.out"
    echo "[${TIMESTAMP}] === Session ${LOOP} / ${MAX_LOOPS} (${SAVE_TAG}) ===" | tee -a "$LOG_DIR/summary.log"

    salloc \
        -C gpu \
        -q interactive \
        -t 240 \
        --nodes 1 \
        --ntasks-per-node 4 \
        --gpus-per-node 4 \
        -A m3246 \
        bash "$SCRIPT_DIR/train_omnifold_pythia_herwig.sh" \
        2>&1 | tee "$LOG_FILE"
    EXIT="${PIPESTATUS[0]}"

    MSG="[$(date '+%Y-%m-%d %H:%M:%S')] Session ${LOOP} exited with code ${EXIT}"
    echo "$MSG" | tee -a "$LOG_DIR/summary.log"

    if [ "$EXIT" -eq 0 ]; then
        echo "Unfolding completed: ${SAVE_TAG}. Stopping loop." | tee -a "$LOG_DIR/summary.log"
        exit 0
    fi

    echo "Non-zero exit (likely time limit or preemption). Resubmitting in 15s..." \
        | tee -a "$LOG_DIR/summary.log"
    sleep 15
done

echo "Reached MAX_LOOPS=${MAX_LOOPS} for ${SAVE_TAG}." | tee -a "$LOG_DIR/summary.log"
exit 1
