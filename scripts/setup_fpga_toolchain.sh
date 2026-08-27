#!/bin/bash
# Installs the PyTorch-native FPGA QAT/synthesis toolchain (Brevitas for
# quantization-aware training in PyTorch, QONNX as the interchange format,
# hls4ml for HLS synthesis via its QONNX frontend), chosen over QKeras/
# TensorFlow specifically to reuse the existing PyTorch DeepSets model/
# training code without a parallel Keras reimplementation. No Vivado/Vitis
# HLS is available on Perlmutter (module spider finds neither), so hls4ml
# here is for C-simulation/bit-accuracy checks and generating a portable HLS
# project -- real resource/latency synthesis (csynth) needs to happen
# wherever Vivado/Vitis HLS access exists.
#
# Installed into a CLONE of the live omnilearned-clean/env, not the env
# itself: hls4ml commonly pulls in TensorFlow as a dependency, which tends
# to pin an older numpy than the current torch 2.12 stack wants -- doing
# that pip resolution in the same env a 4-node DDP training job is actively
# running against (distill_deepsets_wd005 screen session) risks the next
# salloc-loop resubmission picking up a downgraded/incompatible numpy.
# Cloning first keeps this fully isolated; the live env is untouched.
#
# Run inside a screen session:
#   screen -dmS fpga_toolchain_install bash setup_fpga_toolchain.sh
#   screen -r fpga_toolchain_install   # to reattach

set -euo pipefail

module load conda

FPGA_ENV=/global/homes/t/twamorka/omnilearned-fpga/env
if [ ! -d "$FPGA_ENV" ]; then
    echo "=== $(date '+%F %T') cloning omnilearned-clean/env -> omnilearned-fpga/env ==="
    mkdir -p /global/homes/t/twamorka/omnilearned-fpga
    conda create --yes --prefix "$FPGA_ENV" --clone /global/homes/t/twamorka/omnilearned-clean/env
fi
conda activate "$FPGA_ENV"

echo "=== $(date '+%F %T') installing brevitas, qonnx, hls4ml ==="
pip install --no-input brevitas qonnx hls4ml

echo "=== $(date '+%F %T') verifying imports ==="
python3 -c "
import brevitas, qonnx, hls4ml
print('brevitas', brevitas.__version__)
print('qonnx', qonnx.__version__)
print('hls4ml', hls4ml.__version__)
"

echo "=== $(date '+%F %T') checking for Vivado/Vitis HLS (expected: none) ==="
command -v vivado_hls >/dev/null 2>&1 && echo "vivado_hls found" || echo "vivado_hls: not found (expected)"
command -v vitis_hls >/dev/null 2>&1 && echo "vitis_hls found" || echo "vitis_hls: not found (expected)"

echo "=== $(date '+%F %T') done ==="
