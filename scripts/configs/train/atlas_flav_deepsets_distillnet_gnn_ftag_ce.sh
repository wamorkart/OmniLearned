# No-KD control for atlas_flav_deepsets_distillnet_gnn_ftag.sh: same student
# and schedule, trained from scratch on the labels only.
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
DISTILL=0
EXTRA_FLAGS="--use-add --num-add 17 --num-gen-classes 8 --conditional --num-cond 4 --act-layer relu --num-interaction-layers 1 --interaction-k 64"
SAVE_TAG=train_atlas_flav_deepsets_distillnet_relu_ftag_gnn1_k64_ce_scratch
