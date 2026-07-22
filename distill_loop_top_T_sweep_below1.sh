#!/bin/bash
# Sequential low-T (<1) extension of the pure-KD (alpha=0, beta=1) temperature
# sweep on top tagging. Reuses distill_loop_top_T_sweep.sh (teacher=fine_tune_top_l)
# for each T in turn, since only one interactive-queue slot is available for
# this scan (the other is running the pretrain distillation).
#
# Run in a screen session:
#   screen -dmS distill_lowT_sweep bash distill_loop_top_T_sweep_below1.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_sweep
mkdir -p "$LOG_DIR"

T_VALUES=(0.5 0.25 0.125)

for T in "${T_VALUES[@]}"; do
    TAG="distill_top_small_scratch_a00_b10_T${T}"
    LOGFILE="$LOG_DIR/${TAG}/summary.log"
    if [ -f "$LOGFILE" ] && grep -q "Training completed:" "$LOGFILE"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] T=$T (${TAG}) already done, skipping"
        continue
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting T=$T (${TAG})"
    bash "$SCRIPT_DIR/distill_loop_top_T_sweep.sh" "$T"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished T=$T (${TAG})"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Low-T scan (T<1) complete: ${T_VALUES[*]}"
