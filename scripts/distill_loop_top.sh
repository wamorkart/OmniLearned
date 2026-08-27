#!/bin/bash
# Resubmit loop: PET2-small top-tagging KD student (config top_small_a05).
# Thin shim over lib/resubmit_loop.sh + train.sh. Launch inside screen/tmux:
#   screen -dmS distill_top bash scripts/distill_loop_top.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LOOP_LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top
export MAX_LOOPS="${MAX_LOOPS:-20}"
exec "$SCRIPT_DIR/lib/resubmit_loop.sh" bash "$SCRIPT_DIR/run_train.sh" top_small_a05
