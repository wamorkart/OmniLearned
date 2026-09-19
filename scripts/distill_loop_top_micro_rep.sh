#!/bin/bash
# Resubmit loop for one PET2-micro top KD replicate (config top_micro_a05).
#   bash scripts/distill_loop_top_micro_rep.sh distill_top_micro_scratch_a05_T4_r1
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAVE_TAG="${1:?Usage: $0 <save_tag>  e.g. distill_top_micro_scratch_a05_T4_r1}"
export SAVE_TAG LOOP_TAG="$SAVE_TAG"
export LOOP_LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_micro
export MAX_LOOPS="${MAX_LOOPS:-20}"
exec "$SCRIPT_DIR/lib/resubmit_loop.sh" bash "$SCRIPT_DIR/run_train.sh" top_micro_a05
