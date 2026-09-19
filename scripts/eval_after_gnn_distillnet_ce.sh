#!/bin/bash
# One-off driver: wait for the CE-only distillnet-GNN control training to finish, then
# evaluate it on the test split and score it with compute_metrics_top.py
# (alongside the KD distillnet-GNN run it is the control for).
#
#   screen -dmS eval_gnn_ce bash scripts/eval_after_gnn_distillnet_ce.sh
#
# Completion signal: the resubmit loop writes "Training completed" to its
# summary.log when `omnilearned train` exits 0 (all 50 epochs done). We also
# accept the presence of training_<tag>.json (written at the very end of
# train.py) as a backstop, and bail out if the loop hits MAX_LOOPS first.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TAG=train_top_deepsets_distillnet_gnn1_k64_ce_scratch
BASELINE_TAG=distill_top_deepsets_distillnet_scratch_a05_T4_gnn1_k64
EVAL_CONFIG=top_deepsets_distillnet_gnn_ce
EVAL_DIR=/pscratch/sd/t/twamorka/omnilearned/eval/top_distill_deepsets
CKPT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints
LOOP_LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_loop_top_deepsets_distillnet_gnn_ce
SUMMARY="$LOOP_LOG_DIR/summary.log"
TRAIN_JSON="$CKPT_DIR/training_${TAG}.json"

OUT_LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/eval_after_gnn_distillnet_ce
mkdir -p "$OUT_LOG_DIR"
RESULTS="$OUT_LOG_DIR/results.log"

PY=/global/homes/t/twamorka/omnilearned-clean/env/bin/python

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$OUT_LOG_DIR/driver.log"; }

log "driver start; waiting for training to finish (tag=$TAG)"

# --- 1. wait for completion ------------------------------------------------
MAX_WAIT_MIN=900          # 15 h ceiling
POLL_SEC=300
waited=0
while :; do
    if [ -f "$SUMMARY" ] && grep -q "Training completed" "$SUMMARY"; then
        log "summary.log reports training complete"; break
    fi
    if [ -f "$TRAIN_JSON" ]; then
        log "found $TRAIN_JSON (train.py finished)"; break
    fi
    if [ -f "$SUMMARY" ] && grep -q "Reached MAX_LOOPS" "$SUMMARY"; then
        log "ABORT: loop hit MAX_LOOPS before finishing 50 epochs. Not evaluating."
        exit 1
    fi
    if [ "$waited" -ge "$MAX_WAIT_MIN" ]; then
        log "ABORT: waited ${MAX_WAIT_MIN} min without a completion signal."
        exit 1
    fi
    sleep "$POLL_SEC"
    waited=$(( waited + POLL_SEC / 60 ))
done

# small settle so the final best_model_*.pt write is flushed
sleep 30
log "checkpoints present:"
ls -la "$CKPT_DIR"/*"${TAG}"* 2>&1 | tee -a "$OUT_LOG_DIR/driver.log"

# --- 2. evaluate on the test split --------------------------------------
TS=$(date '+%Y-%m-%d_%H-%M-%S')
log "launching eval (salloc 1 node x 4 GPU, run_eval.sh $EVAL_CONFIG)"
cd "$REPO_DIR"
salloc \
    -C gpu -q interactive -t 60 \
    --nodes 1 --ntasks-per-node 4 --gpus-per-node 4 \
    -A m3246 \
    bash scripts/run_eval.sh "$EVAL_CONFIG" \
    2>&1 | tee "$OUT_LOG_DIR/eval_session_${TS}.out"
EVAL_EXIT="${PIPESTATUS[0]}"
log "eval salloc exited $EVAL_EXIT"

NPZ_COUNT=$(ls "$EVAL_DIR"/outputs_"${TAG}"_top_test_rank*.npz 2>/dev/null | wc -l)
log "found $NPZ_COUNT per-rank npz for $TAG"
if [ "$NPZ_COUNT" -eq 0 ]; then
    log "ABORT: eval produced no output npz."
    exit 1
fi

# --- 3. score it (GNN run + baseline, separately) -------------------------
{
    echo "===================================================================="
    echo "  distillnet-GNN CE-only eval  --  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  tag:      $TAG"
    echo "  baseline: $BASELINE_TAG"
    echo "===================================================================="
} | tee -a "$RESULTS"

log "scoring CE-only GNN run"
"$PY" tools/metrics/compute_metrics_top.py --indir "$EVAL_DIR" --tag "$TAG" \
    2>&1 | tee -a "$RESULTS"

if ls "$EVAL_DIR"/outputs_"${BASELINE_TAG}"_top_test_rank*.npz >/dev/null 2>&1; then
    log "scoring KD distillnet-GNN for comparison"
    "$PY" tools/metrics/compute_metrics_top.py --indir "$EVAL_DIR" --tag "$BASELINE_TAG" \
        2>&1 | tee -a "$RESULTS"
else
    log "baseline npz not found in $EVAL_DIR -- skipping baseline score"
fi

log "DONE. Results in $RESULTS"
