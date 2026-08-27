#!/bin/bash
# Waits for the resumed r2/r4 no-interaction micro-KD training loops to finish,
# then evaluates both on the top-tagging test set and builds the comparison table.
set -uo pipefail

cd /global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned

LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_micro_noint
OUT=/pscratch/sd/t/twamorka/omnilearned/results/r2_r4_pipeline.log
PY=/global/homes/t/twamorka/omnilearned-clean/env/bin/python

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$OUT"; }

log "=== Pipeline started, waiting for r2 and r4 training to complete ==="
while true; do
    r2_done=$(grep -c "Training completed. Stopping loop." \
        "$LOG_DIR/distill_top_micro_noint_scratch_a05_T4_r2_summary.log" 2>/dev/null || echo 0)
    r4_done=$(grep -c "Training completed. Stopping loop." \
        "$LOG_DIR/distill_top_micro_noint_scratch_a05_T4_r4_summary.log" 2>/dev/null || echo 0)
    r2_failed=$(grep -c "Reached MAX_LOOPS" \
        "$LOG_DIR/distill_top_micro_noint_scratch_a05_T4_r2_summary.log" 2>/dev/null || echo 0)
    r4_failed=$(grep -c "Reached MAX_LOOPS" \
        "$LOG_DIR/distill_top_micro_noint_scratch_a05_T4_r4_summary.log" 2>/dev/null || echo 0)

    if [ "$r2_failed" -ge 1 ] || [ "$r4_failed" -ge 1 ]; then
        log "ERROR: a training loop hit MAX_LOOPS without completing. r2_failed=$r2_failed r4_failed=$r4_failed. Aborting pipeline."
        exit 1
    fi
    if [ "$r2_done" -ge 1 ] && [ "$r4_done" -ge 1 ]; then
        log "Both r2 and r4 training complete."
        break
    fi
    sleep 60
done

log "=== Evaluating r2 ==="
salloc -C gpu -q interactive -t 60 --nodes 4 --ntasks-per-node 4 --gpus-per-node 4 -A m3246 \
    bash -c "SAVE_TAG=distill_top_micro_noint_scratch_a05_T4_r2 bash scripts/evaluate_top_distill_micro_noint.sh" \
    >> "$OUT" 2>&1
R2_EVAL_EXIT=$?
log "r2 eval exit code: $R2_EVAL_EXIT"

log "=== Evaluating r4 ==="
salloc -C gpu -q interactive -t 60 --nodes 4 --ntasks-per-node 4 --gpus-per-node 4 -A m3246 \
    bash -c "SAVE_TAG=distill_top_micro_noint_scratch_a05_T4_r4 bash scripts/evaluate_top_distill_micro_noint.sh" \
    >> "$OUT" 2>&1
R4_EVAL_EXIT=$?
log "r4 eval exit code: $R4_EVAL_EXIT"

log "=== Computing metrics and comparison table ==="
"$PY" compare_noint_r2_r4.py >> "$OUT" 2>&1
log "=== Pipeline complete ==="
