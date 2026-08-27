#!/bin/bash
# Continuous loop for fine-tuning the JetClass-KD small student on top tagging.
# Wraps fine_tune_top_distill_jetclass_pretrain_l.sh with salloc auto-resubmit on timeout.
#
# Run in a screen session:
#   screen -S top_ft_jc_pretrain_l bash distill_loop_top_jetclass_pretrain_l_finetune.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_jetclass_pretrain_l_finetune
mkdir -p "$LOG_DIR"

SAVE_TAG=fine_tune_top_distill_jetclass_s_pretrain_l_a05_T4
MAX_LOOPS=50
LOOP=0

echo "Fine-tune loop: $SAVE_TAG"

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
        bash "$SCRIPT_DIR/fine_tune_top_distill_jetclass_pretrain_l.sh" \
        2>&1 | tee "$LOG_FILE"
    EXIT="${PIPESTATUS[0]}"

    MSG="[$(date '+%Y-%m-%d %H:%M:%S')] Session ${LOOP} exited with code ${EXIT}"
    echo "$MSG" | tee -a "$LOG_DIR/summary.log"

    if [ "$EXIT" -eq 0 ]; then
        echo "Training completed: ${SAVE_TAG}. Stopping loop." | tee -a "$LOG_DIR/summary.log"
        exit 0
    fi

    echo "Non-zero exit (likely time limit or preemption). Resubmitting in 15s..." \
        | tee -a "$LOG_DIR/summary.log"
    sleep 15
done

echo "Reached MAX_LOOPS=${MAX_LOOPS} for ${SAVE_TAG}." | tee -a "$LOG_DIR/summary.log"
exit 1
