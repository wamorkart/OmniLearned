#!/bin/bash
# Evaluate all three CE-only DeepSets-small "from scratch" reps
# (train_top_deepsets_small_ce_scratch, _r2, _r3 -- see
# fpga-deepsets-distillation-progress memory) on the top-tagging test split,
# one salloc slot at a time, then aggregate accuracy/AUC mean+-std across the
# 3 reps with compute_metrics_top.py --spread.
#
# IMPORTANT: salloc can fail to get resources ("Unable to allocate
# resources: Connection timed out" after sitting PENDING) and STILL exit 0,
# so $? alone cannot tell success from failure here (caught this 2026-08-25:
# 3 straight salloc attempts across this script and eval_deepsets_teachs_spread.sh
# all timed out/got revoked with Start=None in sacct, yet all logged "exited
# with code 0"). Instead, verify the real completion signal: the 4 per-rank
# npz output files for this tag must exist AND be newer than this attempt's
# start time. Retry (fresh salloc) up to MAX_RETRIES times if not.
#
# Run inside a screen session so it survives disconnects:
#   screen -dmS eval_deepsets_ce_spread bash eval_deepsets_ce_spread.sh
#   screen -r eval_deepsets_ce_spread   # to reattach

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/eval_deepsets_ce_spread
OUTPUT_DIR=/pscratch/sd/t/twamorka/omnilearned/eval/top_distill_deepsets
mkdir -p "$LOG_DIR"

TAGS=(
    train_top_deepsets_small_ce_scratch
    train_top_deepsets_small_ce_scratch_r2
    train_top_deepsets_small_ce_scratch_r3
)
MAX_RETRIES=5

for TAG in "${TAGS[@]}"; do
    ATTEMPT=1
    while :; do
        START_EPOCH=$(date +%s)
        echo "[$(date '+%F %T')] Evaluating ${TAG} (attempt ${ATTEMPT}/${MAX_RETRIES})"
        SAVE_TAG="$TAG" salloc \
            -C gpu \
            -q interactive \
            -t 60 \
            --nodes 4 \
            --ntasks-per-node 4 \
            --gpus-per-node 4 \
            -A m3246 \
            bash "$SCRIPT_DIR/evaluate_top_distill_deepsets.sh" \
            > "$LOG_DIR/${TAG}_eval_attempt${ATTEMPT}.out" 2>&1
        SALLOC_EXIT=$?

        NUM_FRESH=$(find "$OUTPUT_DIR" -maxdepth 1 -name "outputs_${TAG}_top_test_rank*.npz" -newermt "@${START_EPOCH}" 2>/dev/null | wc -l)
        if [ "$NUM_FRESH" -ge 1 ]; then
            echo "[$(date '+%F %T')] ${TAG} eval CONFIRMED complete (salloc exit ${SALLOC_EXIT}, ${NUM_FRESH} fresh rank files)" \
                | tee -a "$LOG_DIR/summary.log"
            break
        fi

        echo "[$(date '+%F %T')] ${TAG} eval attempt ${ATTEMPT} produced NO fresh output (salloc exit ${SALLOC_EXIT}) -- salloc likely failed to allocate; see ${TAG}_eval_attempt${ATTEMPT}.out" \
            | tee -a "$LOG_DIR/summary.log"

        if [ "$ATTEMPT" -ge "$MAX_RETRIES" ]; then
            echo "[$(date '+%F %T')] ${TAG} FAILED after ${MAX_RETRIES} attempts -- giving up, spread will be computed without it if unresolved" \
                | tee -a "$LOG_DIR/summary.log"
            break
        fi
        ATTEMPT=$((ATTEMPT + 1))
        sleep 15
    done
done

echo "[$(date '+%F %T')] All evals attempted. Computing metrics + spread." | tee -a "$LOG_DIR/summary.log"

/global/homes/t/twamorka/omnilearned-clean/env/bin/python "$SCRIPT_DIR/compute_metrics_top.py" \
    --indir "$OUTPUT_DIR" \
    --tag "${TAGS[0]}" --tag "${TAGS[1]}" --tag "${TAGS[2]}" \
    | tee -a "$LOG_DIR/metrics.log"
