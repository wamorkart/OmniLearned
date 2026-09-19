# Test-split eval for the hls4ml-friendly plain distillnet student
# (counterpart of train config top_deepsets_distillnet_fpga). The arch flags
# MUST match training exactly so evaluate.py rebuilds the same body before
# restoring the checkpoint.
#
# Score it the same way as the rest of the DeepSets family:
#   /global/homes/t/twamorka/omnilearned-clean/env/bin/python tools/metrics/compute_metrics_top.py \
#     --indir /pscratch/sd/t/twamorka/omnilearned/eval/top_distill_deepsets/ \
#     --tag distill_top_deepsets_distillnet_fpga_a05_T4
ARCH=deep-sets
SIZE=distillnet
INTERACTION=0
LOCAL_INTERACTION=0
EXTRA_FLAGS="--act-layer relu --deepsets-fixed-n 64"
SAVE_TAG=distill_top_deepsets_distillnet_fpga_a05_T4
OUTDIR=/pscratch/sd/t/twamorka/omnilearned/eval/top_distill_deepsets/
