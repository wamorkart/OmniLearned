#!/bin/bash
# End-to-end pipeline for the hls4ml-friendly plain distillnet top-tagging
# student (fixed-N=64, ReLU, no in-graph mask). Runs the whole chain from one
# screen session, one interactive slot at a time, each stage skipped if its
# output already exists (so a killed run just resumes):
#
#   1. float KD training      distill_loop_top.sh top_deepsets_distillnet_fpga
#                             (4-node resubmit loop, exits 0 after 50 epochs)
#   2. float test eval        run_eval.sh + compute_metrics_top.py
#   3. 8-bit QAT              qat_train_deepsets_distillnet_fpga_8bit.sh (4 node)
#   4. QAT test eval          qat_deepsets_fpga_eval.sh (1 GPU)
#   5. QONNX export           qat_deepsets_export_qonnx.py  (CPU salloc)
#   6. hls4ml QONNX ingestion hls4ml_convert_qonnx.py       (CPU salloc)
#
#   screen -dmS fpga_distillnet bash scripts/fpga_distillnet_pipeline.sh
#
# Results table appended to $RESULTS; per-stage progress in $DRIVER_LOG.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

FLOAT_TAG=distill_top_deepsets_distillnet_fpga_a05_T4
QAT_TAG=qat_top_deepsets_distillnet_fpga_a05_T4_8bit
BASELINE_TAG=distill_top_deepsets_distillnet_scratch_a05_T4   # plain (masked, GELU) float

CKPT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints
EVAL_DIR=/pscratch/sd/t/twamorka/omnilearned/eval/top_distill_deepsets
QONNX_DIR=/pscratch/sd/t/twamorka/omnilearned/qonnx/fpga
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/fpga_distillnet_pipeline
mkdir -p "$LOG_DIR" "$QONNX_DIR"
DRIVER_LOG="$LOG_DIR/driver.log"
RESULTS="$LOG_DIR/results.log"

CLEAN_PY=/global/homes/t/twamorka/omnilearned-clean/env/bin/python
FPGA_PY=/global/homes/t/twamorka/omnilearned-fpga/env/bin/python

log() { echo "[$(date '+%F %T')] $*" | tee -a "$DRIVER_LOG"; }
res() { echo "$*" | tee -a "$RESULTS"; }

log "=== fpga distillnet pipeline start (float=$FLOAT_TAG qat=$QAT_TAG) ==="

# ------------------------------------------------------------------ 1. train
TRAIN_JSON="$CKPT_DIR/training_${FLOAT_TAG}.json"
if [ -f "$TRAIN_JSON" ]; then
    log "[1/6] float training already done ($TRAIN_JSON) -- skip"
else
    log "[1/6] launching float KD training loop (distill_loop_top.sh top_deepsets_distillnet_fpga)"
    bash "$SCRIPT_DIR/distill_loop_top.sh" top_deepsets_distillnet_fpga \
        2>&1 | tee -a "$LOG_DIR/stage1_train.out"
    rc="${PIPESTATUS[0]}"
    log "[1/6] training loop exited $rc"
    if [ "$rc" -ne 0 ] || [ ! -f "$TRAIN_JSON" ]; then
        log "ABORT: float training did not complete (no $TRAIN_JSON)."
        exit 1
    fi
fi
sleep 20  # let the final best_model_*.pt write flush

# ------------------------------------------------------------- 2. float eval
FLOAT_NPZ=$(ls "$EVAL_DIR"/outputs_"${FLOAT_TAG}"_top_test_rank*.npz 2>/dev/null | wc -l)
if [ "$FLOAT_NPZ" -gt 0 ]; then
    log "[2/6] float eval npz already present ($FLOAT_NPZ) -- skip eval salloc"
else
    log "[2/6] float eval: salloc 1 node x 4 GPU, run_eval.sh top_deepsets_distillnet_fpga"
    salloc -C gpu -q interactive -t 45 --nodes 1 --ntasks-per-node 4 --gpus-per-node 4 -A m3246 \
        bash scripts/run_eval.sh top_deepsets_distillnet_fpga \
        2>&1 | tee "$LOG_DIR/stage2_eval.out"
    log "[2/6] eval salloc exited ${PIPESTATUS[0]}"
fi
FLOAT_NPZ=$(ls "$EVAL_DIR"/outputs_"${FLOAT_TAG}"_top_test_rank*.npz 2>/dev/null | wc -l)
if [ "$FLOAT_NPZ" -eq 0 ]; then
    log "ABORT: float eval produced no npz."
    exit 1
fi
res ""
res "================ fpga distillnet pipeline  $(date '+%F %T') ================"
res "-- float KD (fixed-N=64, ReLU, no mask) --"
"$CLEAN_PY" tools/metrics/compute_metrics_top.py --indir "$EVAL_DIR" --tag "$FLOAT_TAG" \
    2>&1 | tee -a "$RESULTS"
if ls "$EVAL_DIR"/outputs_"${BASELINE_TAG}"_top_test_rank*.npz >/dev/null 2>&1; then
    res "-- baseline: plain distillnet (masked-mean, GELU) --"
    "$CLEAN_PY" tools/metrics/compute_metrics_top.py --indir "$EVAL_DIR" --tag "$BASELINE_TAG" \
        2>&1 | tee -a "$RESULTS"
fi

# -------------------------------------------------------------------- 3. QAT
QAT_CKPT="$CKPT_DIR/best_model_${QAT_TAG}.pt"
if [ -f "$QAT_CKPT" ]; then
    log "[3/6] QAT checkpoint already present ($QAT_CKPT) -- skip"
else
    log "[3/6] 8-bit QAT: salloc 4 node x 4 GPU, qat_train_deepsets_distillnet_fpga_8bit.sh"
    salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 --gpus-per-node 4 -A m3246 \
        bash scripts/qat_train_deepsets_distillnet_fpga_8bit.sh \
        2>&1 | tee "$LOG_DIR/stage3_qat.out"
    log "[3/6] QAT salloc exited ${PIPESTATUS[0]}"
fi
if [ ! -f "$QAT_CKPT" ]; then
    log "ABORT: QAT produced no checkpoint ($QAT_CKPT)."
    exit 1
fi

# --------------------------------------------------------------- 4. QAT eval
log "[4/6] QAT eval: salloc 1 node x 1 GPU, qat_deepsets_fpga_eval.sh"
salloc -C gpu -q interactive -t 30 --nodes 1 --ntasks-per-node 1 --gpus-per-node 1 -A m3246 \
    bash scripts/qat_deepsets_fpga_eval.sh \
    2>&1 | tee "$LOG_DIR/stage4_qat_eval.out"
log "[4/6] QAT eval salloc exited ${PIPESTATUS[0]}"
res ""
res "-- 8-bit QAT (fpga distillnet) -- (metrics from stage4_qat_eval.out) --"
grep -E "Accuracy|AUC|1/FPR|acc |auc " "$LOG_DIR/stage4_qat_eval.out" 2>/dev/null | tee -a "$RESULTS"

# --------------------------------------------------- 5+6. QONNX export + hls4ml
log "[5-6/6] QONNX export + hls4ml ingestion: salloc -C cpu, fpga_export_and_hls4ml.sh"
salloc -C cpu -q interactive -t 60 -N 1 -A m3246 \
    bash scripts/fpga_export_and_hls4ml.sh \
    2>&1 | tee "$LOG_DIR/stage56_qonnx_hls4ml.out"
log "[5-6/6] CPU salloc exited ${PIPESTATUS[0]}"

log "=== pipeline done. Results: $RESULTS ; hls4ml log: $LOG_DIR/stage56_qonnx_hls4ml.out ==="
