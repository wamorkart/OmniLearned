#!/bin/bash
# Eval the 8-bit QAT hls4ml-friendly plain distillnet student on the top test
# split. Rebuilds the fpga body (--act-layer relu --deepsets-fixed-n 64) with
# 8-bit QuantLinear BEFORE restoring the checkpoint. omnilearned-fpga env
# python by ABSOLUTE path (conda activate + bare python has hit
# ModuleNotFoundError here before).
#
# Run in a 1-node GPU interactive salloc, e.g.:
#   salloc -C gpu -q interactive -t 30 --nodes 1 --ntasks-per-node 1 \
#          --gpus-per-node 1 -A m3246 bash scripts/qat_deepsets_fpga_eval.sh
module load pytorch
# NOTE: do NOT set MASTER_ADDR here -- see qat_deepsets_eval.sh.
cd /global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned
/global/homes/t/twamorka/omnilearned-fpga/env/bin/python tools/quantize/qat_deepsets_eval.py \
    --tag qat_top_deepsets_distillnet_fpga_a05_T4_8bit --size distillnet --bits 8 \
    --act-layer relu --deepsets-fixed-n 64
