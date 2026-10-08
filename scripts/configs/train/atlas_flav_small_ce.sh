# No-KD control for atlas_flav_small_a05_T4.sh: same PET2 small student and
# upstream atlas_flav from-scratch small recipe, trained on hard labels only
# (no teacher), to measure what KD from the v7 teacher adds.
#
# --mode ftag keeps the per-track origin loss; loss = 0.1*CE + track_origin.
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
DISTILL=0
EXTRA_FLAGS="--use-add --num-add 17 --num-gen-classes 8 --conditional --num-cond 4"
SAVE_TAG=train_atlas_flav_small_ce_scratch
