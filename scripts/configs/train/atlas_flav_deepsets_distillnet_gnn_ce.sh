# CE-only control for atlas_flav_deepsets_distillnet_gnn.sh: same student and
# recipe trained on hard labels with no teacher, to measure what KD adds.
DATASET=atlas_flav
DATA_PATH=/pscratch/sd/t/twamorka/omnilearned/datasets/
NUM_CLASSES=4
NUM_FEAT=4
ARCH=deep-sets
SIZE=distillnet
INTERACTION=0
LOCAL_INTERACTION=0
DISTILL=0
EXTRA_FLAGS="--use-add --num-add 17 --conditional --num-cond 4 --num-interaction-layers 1 --interaction-k 64"
SAVE_TAG=train_atlas_flav_deepsets_distillnet_gnn1_k64_ce_scratch
