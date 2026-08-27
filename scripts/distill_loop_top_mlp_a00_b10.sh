#!/bin/bash
# Resubmit loop: pure-KD minimal-MLP student (config top_mlp_a00_b10).
#   screen -dmS distill_top_mlp_a00_b10 bash scripts/distill_loop_top_mlp_a00_b10.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOP_LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_mlp_a00_b10
export MAX_LOOPS="${MAX_LOOPS:-8}"
exec "$SCRIPT_DIR/lib/resubmit_loop.sh" bash "$SCRIPT_DIR/run_train.sh" top_mlp_a00_b10
