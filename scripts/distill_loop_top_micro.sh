#!/bin/bash
# Resubmit loop: PET2-micro top-tagging KD student (config top_micro_a05).
#   screen -dmS distill_top_micro bash scripts/distill_loop_top_micro.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOP_LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_micro
export MAX_LOOPS="${MAX_LOOPS:-20}"
exec "$SCRIPT_DIR/lib/resubmit_loop.sh" bash "$SCRIPT_DIR/run_train.sh" top_micro_a05
