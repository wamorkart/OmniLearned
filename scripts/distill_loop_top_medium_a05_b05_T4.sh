#!/bin/bash
# Resubmit loop: TAKD stage 1 variant, PET2-medium assistant a=b=0.5 (top_medium_a05_b05).
#   screen -dmS distill_medium_a05_b05 bash scripts/distill_loop_top_medium_a05_b05_T4.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOP_LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_sweep/distill_top_medium_scratch_a05_b05_T4
export MAX_LOOPS="${MAX_LOOPS:-50}"
exec "$SCRIPT_DIR/lib/resubmit_loop.sh" bash "$SCRIPT_DIR/run_train.sh" top_medium_a05_b05
