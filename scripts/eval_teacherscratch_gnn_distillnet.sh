#!/bin/bash
# One-off driver: evaluate the already-trained distillnet-GNN KD student that
# was distilled from the from-scratch large teacher (top_l_scratch) on the top
# test split, and score it next to the pretrained-teacher arm for comparison.
#
#   screen -dmS eval_gnn_ts bash scripts/eval_teacherscratch_gnn_distillnet.sh
#
# Training for this tag finished 2026-09-04 (loop summary: "Training completed"),
# so unlike eval_after_gnn_distillnet.sh this driver does not wait -- it just
# checks the checkpoint is on disk, runs the eval, and scores it.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TAG=distill_top_deepsets_distillnet_scratch_a05_T4_gnn1_k64_teacherscratch
BASELINE_TAG=distill_top_deepsets_distillnet_scratch_a05_T4_gnn1_k64   # pretrained-teacher arm
EVAL_CONFIG=top_deepsets_distillnet_gnn_teacherscratch
EVAL_DIR=/pscratch/sd/t/twamorka/omnilearned/eval/top_distill_deepsets
CKPT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints
CKPT="$CKPT_DIR/best_model_${TAG}.pt"

OUT_LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/eval_teacherscratch_gnn_distillnet
mkdir -p "$OUT_LOG_DIR"
RESULTS="$OUT_LOG_DIR/results.log"

PY=/global/homes/t/twamorka/omnilearned-clean/env/bin/python

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$OUT_LOG_DIR/driver.log"; }

log "driver start (tag=$TAG)"

if [ ! -f "$CKPT" ]; then
    log "ABORT: checkpoint not found: $CKPT"
    exit 1
fi
log "checkpoint present:"
ls -la "$CKPT_DIR"/*"${TAG}"* 2>&1 | tee -a "$OUT_LOG_DIR/driver.log"

# --- evaluate on the test split -----------------------------------------
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

# --- score it (teacherscratch arm + pretrained-teacher baseline) -------
{
    echo "===================================================================="
    echo "  distillnet-GNN KD eval  --  from-scratch-teacher arm"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  tag:      $TAG"
    echo "  baseline: $BASELINE_TAG  (pretrained teacher, 93.98%)"
    echo "===================================================================="
} | tee -a "$RESULTS"

log "scoring teacherscratch arm"
"$PY" tools/metrics/compute_metrics_top.py --indir "$EVAL_DIR" --tag "$TAG" \
    2>&1 | tee -a "$RESULTS"

if ls "$EVAL_DIR"/outputs_"${BASELINE_TAG}"_top_test_rank*.npz >/dev/null 2>&1; then
    log "scoring pretrained-teacher baseline for comparison"
    "$PY" tools/metrics/compute_metrics_top.py --indir "$EVAL_DIR" --tag "$BASELINE_TAG" \
        2>&1 | tee -a "$RESULTS"
else
    log "baseline npz not found in $EVAL_DIR -- skipping baseline score"
fi

log "DONE. Results in $RESULTS"
