#!/bin/bash
# Resubmit loop for the arXiv:2512.17011 comparison job (#1-#4). One node,
# 1 GPU is enough (single-process analysis scripts) but interactive QOS
# allocates a whole node. Short job; MAX_LOOPS is just preemption headroom.
#
#   screen -dmS paper_compare bash scripts/paper_compare_loop.sh
#   screen -r paper_compare
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/paper_compare
mkdir -p "$LOG_DIR"

MAX_LOOPS=${MAX_LOOPS:-3}
LOOP=0
while [ "$LOOP" -lt "$MAX_LOOPS" ]; do
    LOOP=$((LOOP + 1))
    TS=$(date '+%Y-%m-%d_%H-%M-%S')
    LOG="$LOG_DIR/session_${LOOP}_${TS}.out"
    echo "[$TS] === session $LOOP / $MAX_LOOPS ===" | tee -a "$LOG_DIR/summary.log"

    set +e
    salloc -C gpu -q interactive -t 90 --nodes 1 --ntasks-per-node 1 \
        --gpus-per-node 4 -A m3246 \
        bash "$SCRIPT_DIR/paper_compare_job.sh" 2>&1 | tee "$LOG"
    EXIT="${PIPESTATUS[0]}"
    set -e

    echo "[$(date '+%F %T')] session $LOOP exit $EXIT" | tee -a "$LOG_DIR/summary.log"
    if [ "$EXIT" -eq 0 ]; then
        echo "done." | tee -a "$LOG_DIR/summary.log"
        exit 0
    fi
    echo "non-zero exit, resubmitting in 15s..." | tee -a "$LOG_DIR/summary.log"
    sleep 15
done
echo "reached MAX_LOOPS=$MAX_LOOPS without clean exit." | tee -a "$LOG_DIR/summary.log"
exit 1
