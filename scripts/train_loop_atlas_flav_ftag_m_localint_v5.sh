#!/bin/bash
# Continuous training loop for the atlas_flav ftag fine-tune "v5" run
# (fine_tune_atlas_flav_m_localint_v5 = upstream train.sh reference command
# + --local-interaction, dataset on /pscratch), 4 nodes x 4 GPUs = 16 GPUs on
# the Perlmutter interactive queue. Keeps resubmitting salloc jobs until
# training finishes (exit 0) or MAX_LOOPS is hit. Same pattern as
# train_loop_atlas_flav_ftag_m_localint_ref16.sh, pointed at the v5
# per-session script.
#
# Run inside a screen session so the loop survives terminal disconnects:
#   screen -dmS train_atlas_flav_ftag_m_localint_v5 bash scripts/train_loop_atlas_flav_ftag_m_localint_v5.sh
#   screen -r train_atlas_flav_ftag_m_localint_v5   # to reattach

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/train_loop_atlas_flav_ftag_m_localint_v5
mkdir -p "$LOG_DIR"

# ~3.8s/iter x 2000 iter x 30 epoch ~= 63hr wall; each interactive session is
# 240 min, so ~16-20 sessions are needed. MAX_LOOPS=20 (~80hr headroom).
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
        bash "$SCRIPT_DIR/train_atlas_flav_ftag_m_localint_v5_interactive.sh" \
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
