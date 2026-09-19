# hls4ml-friendly plain DeepSets/PFN "distillnet"-width top-tagging student.
# Same KD recipe as top_deepsets_distillnet (a0.5/b0.5/T4, teacher
# fine_tune_top_l, wd 0.5, 50 ep, lr 5e-4) -- the only changes are the two
# EXTRA_FLAGS that make the exported ONNX graph ingestible by hls4ml:
#   --act-layer relu       GELU has no ONNX/hls4ml lowering
#   --deepsets-fixed-n 64  fixed 64 leading-pT slots, NO in-graph validity
#                          mask, plain mean pool -> drops the Gelu / Equal /
#                          Not / Cast / masked-ReduceSum blocker ops
# Baseline to beat: plain distillnet float = 92.85% acc / AUC 0.9800 /
# 1/FPR@50% 199.1 (accepts a small hit from dropping the mask + truncating).
ARCH=deep-sets
SIZE=distillnet
INTERACTION=0
LOCAL_INTERACTION=0
EXTRA_FLAGS="--act-layer relu --deepsets-fixed-n 64"
SAVE_TAG=distill_top_deepsets_distillnet_fpga_a05_T4
