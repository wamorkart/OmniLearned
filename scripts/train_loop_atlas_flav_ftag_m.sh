#!/bin/bash
# Continuous training loop for the atlas_flav ftag fine-tune
# (fine_tune_atlas_flav_m), 4 nodes x 4 GPUs on the Perlmutter interactive
# queue. Keeps resubmitting salloc jobs until training finishes (exit 0) or
# MAX_LOOPS is hit. Modeled on distill_loop_top_deepsets.sh.
#
# Run inside a screen session so the loop survives terminal disconnects:
#   screen -dmS train_atlas_flav_ftag_m bash train_loop_atlas_flav_ftag_m.sh
#   screen -r train_atlas_flav_ftag_m   # to reattach

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/train_loop_atlas_flav_ftag_m
mkdir -p "$LOG_DIR"

# Measured throughput ~3.8s/iter (medium model + --interaction + dual-head
# ftag loss, much heavier than this project's other trainings) -> the full
# 30-epoch schedule needs ~60hr wall time. MAX_LOOPS=8 (32hr) was sized
# before that was known and would exhaust partway through; raised to 20
# (~80hr headroom) so a fresh launch of this script can actually finish.
# Safe to edit while a loop from an OLDER version of this file is live: bash
# already has the old value loaded into its running process and won't
# re-read this file mid-run -- this only takes effect on the next fresh
# invocation (e.g. once that older run exhausts its own resubmits).
MAX_LOOPS=20
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
        bash "$SCRIPT_DIR/train_atlas_flav_ftag_m.sh" \
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
