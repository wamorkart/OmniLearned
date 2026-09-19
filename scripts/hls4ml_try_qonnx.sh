#!/bin/bash
# Attempt QONNX -> hls4ml ingestion for the DeepSets top-tagging QAT students.
# NO Vivado/Vitis HLS on this system -> C-sim only, no synthesis numbers.
# The point is to learn how far hls4ml's QONNX frontend gets on (a) the plain
# distillnet MLP and (b) the distillnet+GNN graph, and exactly which op it
# dies on.
#
# Launch detached:
#   screen -dmS hls4ml_qonnx bash scripts/hls4ml_try_qonnx.sh
# Watch:
#   tail -f /pscratch/sd/t/twamorka/omnilearned/logs/hls4ml_qonnx/run.log
#
# CPU-only (python + g++). Uses the omnilearned-fpga env python by ABSOLUTE
# path (has Brevitas/QONNX/hls4ml).
set -u
cd /global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned
module load pytorch 2>/dev/null

PY=/global/homes/t/twamorka/omnilearned-fpga/env/bin/python
QDIR=/pscratch/sd/t/twamorka/omnilearned/qonnx
LOGDIR=/pscratch/sd/t/twamorka/omnilearned/logs/hls4ml_qonnx
mkdir -p "$LOGDIR"
LOG="$LOGDIR/run.log"

export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OMP_PROC_BIND=false
exec > >(tee -a "$LOG") 2>&1

echo "############################################################"
echo "# hls4ml QONNX ingestion attempt  --  $(date '+%F %T')"
echo "############################################################"

# ---- 1. make sure the plain-distillnet QAT QONNX export exists too --------
PLAIN_CLEAN="$QDIR/plain/qat_top_deepsets_distillnet_a05_T4_8bit_clean.onnx"
if [[ ! -f "$PLAIN_CLEAN" ]]; then
    echo "[$(date '+%T')] exporting plain distillnet QAT -> QONNX"
    $PY tools/quantize/qat_deepsets_export_qonnx.py \
        --tag qat_top_deepsets_distillnet_a05_T4_8bit \
        --size distillnet --bits 8 \
        --num-interaction-layers 0 --interaction-k 0 \
        --batch 32 --out-dir "$QDIR/plain"
else
    echo "[$(date '+%T')] plain distillnet QONNX already present"
fi

# ---- 2. hls4ml attempt: plain distillnet (easy case) ---------------------
echo
echo "[$(date '+%T')] >>> hls4ml attempt: PLAIN distillnet"
$PY tools/quantize/hls4ml_convert_qonnx.py \
    --onnx "$PLAIN_CLEAN" \
    --out-dir "$LOGDIR/hls_plain" \
    --backend Vitis --io-type io_stream

# ---- 3. hls4ml attempt: distillnet + GNN (hard case) --------------------
echo
echo "[$(date '+%T')] >>> hls4ml attempt: distillnet + GNN"
$PY tools/quantize/hls4ml_convert_qonnx.py \
    --onnx "$QDIR/qat_top_deepsets_distillnet_gnn_a05_T4_8bit_clean.onnx" \
    --out-dir "$LOGDIR/hls_gnn" \
    --backend Vitis --io-type io_stream

echo
echo "[$(date '+%T')] === done. Full log: $LOG ==="
