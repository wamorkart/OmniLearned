#!/bin/bash
# Loop driver for the alpha/beta sweep on top-tagging distillation.
#
# Usage (run inside tmux so it survives disconnects):
#   tmux new-session -s distill_top_a025_b075 \
#     'bash distill_loop_top_sweep.sh 0.25 0.75 4'
#
# Arguments: ALPHA BETA [DISTILL_T]   (T defaults to 4)
#
# Each salloc grabs 4 nodes x 4 GPUs for 4 hours.
# Loop resubmits on non-zero exit (time limit / preemption).
# Stops cleanly on exit 0 (all epochs done).
# MAX_LOOPS=20 caps at 80 node-hours per run.

set -euo pipefail

ALPHA=${1:?Usage: $0 ALPHA BETA [T]}
BETA=${2:?Usage: $0 ALPHA BETA [T]}
DISTILL_T=${3:-4}

# Build save tag: str(float) with decimal stripped, e.g. 0.5->"05", 0.25->"025", 1.0->"10"
export SAVE_TAG=$(python3 -c "
a, b, t = '$ALPHA', '$BETA', '$DISTILL_T'
def fmt(x): return str(float(x)).replace('.', '')
print(f'distill_top_small_scratch_a{fmt(a)}_b{fmt(b)}_T{int(float(t))}')
")
export ALPHA BETA DISTILL_T

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_sweep/${SAVE_TAG}
mkdir -p "$LOG_DIR"

MAX_LOOPS=50
LOOP=0

echo "Starting sweep run: ALPHA=$ALPHA BETA=$BETA T=$DISTILL_T SAVE_TAG=$SAVE_TAG"

while [ "$LOOP" -lt "$MAX_LOOPS" ]; do
    LOOP=$((LOOP + 1))
    TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
    LOG_FILE="$LOG_DIR/session_${LOOP}_${TIMESTAMP}.out"
    echo "[${TIMESTAMP}] === Session ${LOOP} / ${MAX_LOOPS} (${SAVE_TAG}) ===" | tee -a "$LOG_DIR/summary.log"

    salloc \
        -C gpu \
        -q interactive \
        -t 240 \
        --nodes 4 \
        --ntasks-per-node 4 \
        --gpus-per-node 4 \
        -A m3246 \
        bash "$SCRIPT_DIR/distill_train_top_sweep.sh" \
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
