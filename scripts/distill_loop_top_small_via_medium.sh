#!/bin/bash
# Resubmit loop: TAKD stage 3, medium assistant -> PET2-small (top_small_via_medium_a00_b10).
#   screen -dmS distill_small_via_medium bash scripts/distill_loop_top_small_via_medium.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOP_LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_sweep/distill_top_small_via_medium_a00_b10_T4
export MAX_LOOPS="${MAX_LOOPS:-50}"
exec "$SCRIPT_DIR/lib/resubmit_loop.sh" bash "$SCRIPT_DIR/run_train.sh" top_small_via_medium_a00_b10
