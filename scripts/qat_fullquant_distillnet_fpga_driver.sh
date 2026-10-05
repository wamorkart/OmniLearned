#!/bin/bash
# Full-quant 8-bit QAT of the plain (no-GNN) fixed-N=64 ReLU distillnet student,
# for the paper's quantization table, then everything the table row needs:
#   1. QAT   (4 node x 4 GPU, ~2 h)  qat_train_deepsets_distillnet_fpga_8bit.sh
#   2. eval  (1 GPU)                 qat_deepsets_eval.py on the full test set,
#                                    QONNX export + PyTorch-vs-QONNX parity,
#                                    float32 + 8-bit PTQ (both, inputs-only) of
#                                    the same float checkpoint
# Finished stages are skipped on rerun.
#
# Launch:  screen -dmS qat_fullquant env -u OMNILEARNED_ENV bash scripts/qat_fullquant_distillnet_fpga_driver.sh

REPO=/global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned
PY=/global/homes/t/twamorka/omnilearned-fpga/env/bin/python
CKPT=/pscratch/sd/t/twamorka/omnilearned/checkpoints
LOGDIR=/pscratch/sd/t/twamorka/omnilearned/logs/qat_fullquant_distillnet_fpga
FLOAT_TAG=distill_top_deepsets_distillnet_fpga_a05_T4
QAT_TAG=qat_top_deepsets_distillnet_fpga_a05_T4_8bit_fullQuant
mkdir -p "$LOGDIR"
cd "$REPO" || exit 1
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGDIR/driver.log"; }

if [ ! -f "$CKPT/best_model_${QAT_TAG}.pt" ]; then
    log "[1/2] QAT: salloc 4 node x 4 GPU"
    salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 --gpus-per-node 4 -A m3246 \
        bash scripts/qat_train_deepsets_distillnet_fpga_8bit.sh > "$LOGDIR/qat.log" 2>&1
    [ -f "$CKPT/best_model_${QAT_TAG}.pt" ] || { log "QAT produced no checkpoint, see qat.log"; exit 1; }
fi
log "[1/2] QAT done: $CKPT/best_model_${QAT_TAG}.pt"

# Single process, no MASTER_ADDR (qat_deepsets_eval.py's own instruction).
log "[2/2] eval + export + float/PTQ: salloc 1 GPU"
salloc -C gpu -q interactive -t 90 --nodes 1 --ntasks-per-node 1 --gpus-per-node 1 -A m3246 bash -c "
    cd $REPO
    $PY tools/quantize/qat_deepsets_eval.py --tag $QAT_TAG > $LOGDIR/eval_qat.log 2>&1
    $PY tools/quantize/qat_deepsets_export_qonnx.py --tag $QAT_TAG > $LOGDIR/export_qonnx.log 2>&1
    $PY tools/quantize/ptq_deepsets.py --tag $FLOAT_TAG --size distillnet --act-layer relu \
        --deepsets-fixed-n 64 --bits 8 > $LOGDIR/ptq_both.log 2>&1
    $PY tools/quantize/ptq_deepsets.py --tag $FLOAT_TAG --size distillnet --act-layer relu \
        --deepsets-fixed-n 64 --bits 8 --weights-float > $LOGDIR/ptq_inputs_only.log 2>&1
"
log "[2/2] done"
grep -A6 "full-quant QAT" "$LOGDIR/eval_qat.log" | tee -a "$LOGDIR/driver.log"
grep -i "max |" "$LOGDIR/export_qonnx.log" | tee -a "$LOGDIR/driver.log"
grep -A4 "=== Summary" "$LOGDIR/ptq_both.log" | tee -a "$LOGDIR/driver.log"
