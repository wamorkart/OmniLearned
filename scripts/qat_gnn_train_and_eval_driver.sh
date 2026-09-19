#!/bin/bash
# Detached-screen driver: run the 8-bit QAT fine-tune of the distillnet+GNN
# top-tagging student on an interactive salloc, then (on clean exit) evaluate
# it on the 404k top test split and score it against the float baseline.
#
# Launch:  screen -dmS qat_gnn bash scripts/qat_gnn_train_and_eval_driver.sh
#
# One-shot: qat_deepsets.py always warm-starts from the float --tag, so if the
# salloc is preempted the rerun just restarts the 15-epoch fine-tune cleanly.
set -u
cd /global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned

LOGDIR=/pscratch/sd/t/twamorka/omnilearned/logs/qat_gnn_distillnet
mkdir -p "$LOGDIR"
DRIVER_LOG="$LOGDIR/driver.log"
RESULTS_LOG="$LOGDIR/results.log"
SAVE_TAG=qat_top_deepsets_distillnet_gnn_a05_T4_8bit
CKPT=/pscratch/sd/t/twamorka/omnilearned/checkpoints/best_model_${SAVE_TAG}.pt

log() { echo "[$(date '+%F %T')] $*" | tee -a "$DRIVER_LOG"; }

log "=== QAT-GNN driver start (save-tag $SAVE_TAG) ==="

log "launching QAT training salloc (4 nodes x 4 GPU, -q interactive, 240 min)"
salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
       --gpus-per-node 4 -A m3246 \
       bash scripts/qat_train_deepsets_distillnet_gnn_8bit.sh \
       >> "$DRIVER_LOG" 2>&1
rc=$?
log "QAT training salloc exited $rc"

if [[ ! -f "$CKPT" ]]; then
    log "ERROR: expected checkpoint $CKPT not found -- stopping (no eval)."
    exit 1
fi
log "QAT checkpoint present: $(ls -l "$CKPT")"

log "launching eval salloc (1 node x 4 GPU, -q interactive, 60 min)"
salloc -C gpu -q interactive -t 60 --nodes 1 --ntasks-per-node 4 \
       --gpus-per-node 4 -A m3246 \
       bash scripts/qat_deepsets_gnn_eval.sh \
       >> "$DRIVER_LOG" 2>&1
log "eval salloc exited $?"

{
    echo "===================================================================="
    echo "  distillnet+GNN 8-bit QAT eval  --  $(date '+%F %T')"
    echo "  QAT tag : $SAVE_TAG"
    echo "  float baseline: distill_top_deepsets_distillnet_scratch_a05_T4_gnn1_k64"
    echo "                  93.98% acc / AUC 0.9859 / rej50 449.7 / rej30 1979.5"
    echo "  (plain distillnet 8-bit QAT for reference: 92.58% / 0.9785 / 178.1 / 698.7)"
    echo "===================================================================="
} >> "$RESULTS_LOG"
# the eval script prints its metrics to the driver log; copy the last block over
tail -20 "$DRIVER_LOG" >> "$RESULTS_LOG"

log "=== driver done. Metrics in $RESULTS_LOG ==="
