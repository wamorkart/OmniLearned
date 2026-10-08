# Plain DeepSets "distillnet"-width student (no GNN block) with ReLU instead
# of GELU (GELU has no ONNX/hls4ml lowering), on ATLAS flavour tagging
# (4 classes: 0=light 1=c 2=b 3=tau). KD from the v7 ftag fine-tune
# (docs/ATLAS_FLAV_RESULTS.md) with the top-tagging distillnet recipe
# (a0.5/b0.5/T4, wd 0.5, lr 5e-4, 50 ep). Classifier mode: the student has no
# per-track origin head, so only the jet-flavour logits are distilled.
#
# Keeps the padding mask (no --deepsets-fixed-n), unlike
# top_deepsets_distillnet_fpga.sh.
#
# Requires teacher logits built via scripts/save_teacher_logits_atlas_flav.sh.
DATASET=atlas_flav
DATA_PATH=/pscratch/sd/t/twamorka/omnilearned/datasets/
NUM_CLASSES=4
NUM_FEAT=4
ARCH=deep-sets
SIZE=distillnet
INTERACTION=0
LOCAL_INTERACTION=0
TEACHER_TAG=fine_tune_atlas_flav_m_localint_v7
EXTRA_FLAGS="--use-add --num-add 17 --conditional --num-cond 4 --act-layer relu"
SAVE_TAG=distill_atlas_flav_deepsets_distillnet_relu_a05_T4
