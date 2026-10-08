# DeepSets "distillnet" student + one GNN message-passing block (K=64),
# distilled from the v7 atlas_flav ftag fine-tune
# (fine_tune_atlas_flav_m_localint_v7, docs/ATLAS_FLAV_RESULTS.md).
#
# Identical to atlas_flav_deepsets_distillnet_relu_ftag.sh (ftag schedule,
# ReLU, track-origin head) except for the GNN block, so the pair isolates
# what the GNN adds. Supersedes atlas_flav_deepsets_distillnet_gnn.sh,
# which used the top-tagging schedule. atlas_flav jets hold at most 40
# tracks, so K=64 never truncates.
#
# Requires teacher logits built via scripts/save_teacher_logits_atlas_flav.sh.
DATASET=atlas_flav
DATA_PATH=/pscratch/sd/t/twamorka/omnilearned/datasets/
MODE=ftag
NUM_CLASSES=4
NUM_FEAT=4
ARCH=deep-sets
SIZE=distillnet
INTERACTION=0
LOCAL_INTERACTION=0
BATCH=512
ITERATIONS=2000
EPOCH=30
LR=5e-4
WD=0.1
TEACHER_TAG=fine_tune_atlas_flav_m_localint_v7
EXTRA_FLAGS="--use-add --num-add 17 --num-gen-classes 8 --conditional --num-cond 4 --act-layer relu --num-interaction-layers 1 --interaction-k 64"
SAVE_TAG=distill_atlas_flav_deepsets_distillnet_relu_ftag_a05_T4_gnn1_k64
