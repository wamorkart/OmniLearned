#!/bin/bash
# Resubmit loop: minimal-MLP top-tagging KD student (config top_mlp_a05).
#   screen -dmS distill_top_mlp bash scripts/distill_loop_top_mlp.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOP_LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_mlp
export MAX_LOOPS="${MAX_LOOPS:-8}"
exec "$SCRIPT_DIR/lib/resubmit_loop.sh" bash "$SCRIPT_DIR/run_train.sh" top_mlp_a05
