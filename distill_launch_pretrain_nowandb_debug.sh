#!/bin/bash
# One-shot launcher for the no-wandb A/B test (see
# distill_train_pretrain_merged_nowandb_debug.sh for rationale). No resubmit
# loop -- this is a single observation, not a real training run: if it hangs,
# we want to inspect and stop, not auto-retry.
#
# Run inside a screen session so it survives terminal disconnects:
#   screen -dmS distill_nowandb_debug bash distill_launch_pretrain_nowandb_debug.sh

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_nowandb_debug
mkdir -p "$LOG_DIR"

TIMESTAMP=$(date '+%Y-%m-%d_%H-%M-%S')
LOG_FILE="$LOG_DIR/session_${TIMESTAMP}.out"
echo "[${TIMESTAMP}] === no-wandb A/B debug run ===" | tee -a "$LOG_DIR/summary.log"

salloc \
    -C gpu \
    -q interactive \
    -t 90 \
    --nodes 4 \
    --ntasks-per-node 4 \
    --gpus-per-node 4 \
    -A m3246 \
    bash "$SCRIPT_DIR/distill_train_pretrain_merged_nowandb_debug.sh" \
    2>&1 | tee "$LOG_FILE"
EXIT="${PIPESTATUS[0]}"

MSG="[$(date '+%Y-%m-%d %H:%M:%S')] no-wandb debug run exited with code ${EXIT}"
echo "$MSG" | tee -a "$LOG_DIR/summary.log"

if [ "$EXIT" -eq 0 ]; then
    echo "SURVIVED cleanly (no hang) -- wandb is implicated as the trigger." \
        | tee -a "$LOG_DIR/summary.log"
else
    echo "Hung/crashed too (exit ${EXIT}) -- wandb is NOT the (sole) cause, check $LOG_FILE." \
        | tee -a "$LOG_DIR/summary.log"
fi
