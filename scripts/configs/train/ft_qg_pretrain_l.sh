# CE-only fine-tune of the large pretrained model (pretrain_l) on the
# quark/gluon dataset, 2-class classifier.
# (was fine_tune_qg_pretrain_l.sh, from the distill_dev branch)
#
# Interaction ON, local-interaction OFF; iterations 300, 10 epochs, batch 32,
# wd 10.0, lr 1e-6. The save-tag encodes those (int / i300 / e10) so it stays
# comparable to the original distill_dev checkpoints.
DISTILL=0
FINETUNE=1
PRETRAIN_TAG=pretrain_l
DATASET=qg
NUM_CLASSES=2
SIZE=large
INTERACTION=1
LOCAL_INTERACTION=0
ITERATIONS=300
BATCH=32
EPOCH=10
WD=10.0
LR=1e-6
EXTRA_FLAGS="--use-pid"
SAVE_TAG=fine_tune_pretrain_l_qg_int_i300_e10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
