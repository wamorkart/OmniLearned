#!/bin/bash
# Resubmit loop: TAKD stage 1, PET2-medium assistant, pure KD (top_medium_a00_b10).
#   screen -dmS distill_top_medium bash scripts/distill_loop_top_medium.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOP_LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_sweep/distill_top_medium_scratch_a00_b10_T4
export MAX_LOOPS="${MAX_LOOPS:-50}"
exec "$SCRIPT_DIR/lib/resubmit_loop.sh" bash "$SCRIPT_DIR/run_train.sh" top_medium_a00_b10
