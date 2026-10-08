# Plain DeepSets "distillnet" student (no GNN block, ReLU, padding mask kept)
# distilled from the v7 atlas_flav ftag fine-tune
# (fine_tune_atlas_flav_m_localint_v7, docs/ATLAS_FLAV_RESULTS.md).
#
# --mode ftag adds DeepSets' training-only per-track origin head (8 classes),
# so, as for the PET2-small student, the loss is
#   a * (0.1*CE + track_origin) + b * KD.
# Evaluate with --mode classifier; the track head is not needed at inference.
#
# Schedule is the atlas_flav one (bs512, 2000 it, 30 ep, wd 0.1) but at
# lr 5e-4 instead of 5e-5, since the student is ~250x smaller than PET2-small.
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
EXTRA_FLAGS="--use-add --num-add 17 --num-gen-classes 8 --conditional --num-cond 4 --act-layer relu"
SAVE_TAG=distill_atlas_flav_deepsets_distillnet_relu_ftag_a05_T4
