#!/bin/bash
# Stage 5+6 of fpga_distillnet_pipeline.sh, split out so the nested salloc
# quoting stays sane. CPU-only: QONNX export of the 8-bit QAT fpga distillnet
# checkpoint, then hls4ml QONNX ingestion. omnilearned-fpga env python by
# ABSOLUTE path (Brevitas/QONNX/hls4ml). NO Vivado/Vitis on this system ->
# hls4ml gets as far as parse/config/compile + C-sim, no synthesis numbers.
#
#   salloc -C cpu -q interactive -t 60 -N 1 -A m3246 bash scripts/fpga_export_and_hls4ml.sh
set -uo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"
module load pytorch 2>/dev/null
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OMP_PROC_BIND=false

FPGA_PY=/global/homes/t/twamorka/omnilearned-fpga/env/bin/python
QAT_TAG=qat_top_deepsets_distillnet_fpga_a05_T4_8bit
QONNX_DIR=/pscratch/sd/t/twamorka/omnilearned/qonnx/fpga
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/fpga_distillnet_pipeline
CLEAN_ONNX="$QONNX_DIR/${QAT_TAG}_clean.onnx"
mkdir -p "$QONNX_DIR" "$LOG_DIR"

if [ -f "$CLEAN_ONNX" ]; then
    echo "[export] $CLEAN_ONNX already present -- skip"
else
    echo "[export] qat_deepsets_export_qonnx.py -> $QONNX_DIR"
    "$FPGA_PY" tools/quantize/qat_deepsets_export_qonnx.py \
        --tag "$QAT_TAG" --size distillnet --bits 8 \
        --act-layer relu --deepsets-fixed-n 64 \
        --num-interaction-layers 0 --interaction-k 0 \
        --batch 64 --out-dir "$QONNX_DIR" || { echo "[export] FAILED"; exit 1; }
fi

echo
echo "[hls4ml] hls4ml_convert_qonnx.py on $CLEAN_ONNX"
"$FPGA_PY" tools/quantize/hls4ml_convert_qonnx.py \
    --onnx "$CLEAN_ONNX" \
    --out-dir "$LOG_DIR/hls" \
    --backend Vitis --io-type io_stream
echo "[hls4ml] exit $?"
