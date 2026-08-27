#!/bin/bash
# Evaluate all three small-teacher-KD DeepSets-small reps
# (distill_top_deepsets_small_scratch_teachS_a05_T4, _r2, _r3 -- see
# fpga-deepsets-distillation-progress memory) on the top-tagging test split,
# one salloc slot at a time, then aggregate accuracy/AUC mean+-std across the
# 3 reps with compute_metrics_top.py (repeated --tag).
#
# Run inside a screen session so it survives disconnects:
#   screen -dmS eval_deepsets_teachs_spread bash eval_deepsets_teachs_spread.sh
#   screen -r eval_deepsets_teachs_spread   # to reattach

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/eval_deepsets_teachs_spread
mkdir -p "$LOG_DIR"

TAGS=(
    distill_top_deepsets_small_scratch_teachS_a05_T4
    distill_top_deepsets_small_scratch_teachS_a05_T4_r2
    distill_top_deepsets_small_scratch_teachS_a05_T4_r3
)

for TAG in "${TAGS[@]}"; do
    echo "[$(date '+%F %T')] Evaluating ${TAG}"
    SAVE_TAG="$TAG" salloc \
        -C gpu \
        -q interactive \
        -t 60 \
        --nodes 4 \
        --ntasks-per-node 4 \
        --gpus-per-node 4 \
        -A m3246 \
        bash "$SCRIPT_DIR/evaluate_top_distill_deepsets.sh" \
        > "$LOG_DIR/${TAG}_eval.out" 2>&1
    echo "[$(date '+%F %T')] ${TAG} eval exited with code $?" | tee -a "$LOG_DIR/summary.log"
done

echo "[$(date '+%F %T')] All evals complete. Computing metrics + spread." | tee -a "$LOG_DIR/summary.log"

/global/homes/t/twamorka/omnilearned-clean/env/bin/python "$SCRIPT_DIR/compute_metrics_top.py" \
    --indir /pscratch/sd/t/twamorka/omnilearned/eval/top_distill_deepsets/ \
    --tag "${TAGS[0]}" --tag "${TAGS[1]}" --tag "${TAGS[2]}" \
    | tee -a "$LOG_DIR/metrics.log"
