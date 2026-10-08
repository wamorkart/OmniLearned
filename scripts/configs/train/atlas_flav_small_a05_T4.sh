# PET2 small student (~2.8M params) distilled from the v7 atlas_flav ftag
# fine-tune (fine_tune_atlas_flav_m_localint_v7, docs/ATLAS_FLAV_RESULTS.md).
#
# Schedule is the upstream atlas_flav from-scratch small recipe
# (scripts/train.sh: --epoch 30 --lr 5e-5 --wd 0.1 --iterations 2000
# --batch 512 --mode ftag --num-gen-classes 8), NOT the top-tagging KD
# defaults. --local-interaction kept on to match the teacher.
#
# --mode ftag keeps the per-track origin loss. Note that in ftag mode
# get_loss scales the jet CE by 0.1, so the total is
#   a * (0.1*CE + track_origin) + b * KD.
#
# Requires teacher logits built via scripts/save_teacher_logits_atlas_flav.sh.
DATASET=atlas_flav
DATA_PATH=/pscratch/sd/t/twamorka/omnilearned/datasets/
MODE=ftag
NUM_CLASSES=4
SIZE=small
INTERACTION=1
LOCAL_INTERACTION=1
BATCH=512
ITERATIONS=2000
EPOCH=30
LR=5e-5
WD=0.1
TEACHER_TAG=fine_tune_atlas_flav_m_localint_v7
EXTRA_FLAGS="--use-add --num-add 17 --num-gen-classes 8 --conditional --num-cond 4"
SAVE_TAG=distill_atlas_flav_small_a05_T4
