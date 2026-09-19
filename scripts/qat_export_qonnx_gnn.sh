#!/bin/bash
# Export the distillnet+GNN 8-bit QAT top-tagging checkpoint to QONNX and
# run the cleanup / shape-inference / onnxruntime-parity checks in
# qat_deepsets_export_qonnx.py.
#
# CPU-only graph work (no GPU, no DDP) -- run it in a short CPU interactive
# salloc so it does not touch the GPU interactive allowance, e.g.:
#   salloc -C cpu -q interactive -t 20 -N 1 -A m3246 \
#          bash scripts/qat_export_qonnx_gnn.sh
#
# Uses the omnilearned-fpga env python by ABSOLUTE path (has Brevitas/QONNX;
# conda activate + bare python has hit ModuleNotFoundError here before).
module load pytorch
cd /global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned
/global/homes/t/twamorka/omnilearned-fpga/env/bin/python \
    tools/quantize/qat_deepsets_export_qonnx.py \
    --tag qat_top_deepsets_distillnet_gnn_a05_T4_8bit \
    --size distillnet --bits 8 \
    --num-interaction-layers 1 --interaction-k 64
