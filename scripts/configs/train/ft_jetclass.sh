# CE-only fine-tune of pretrain_s -> 10-class JetClass classifier.
# (was fine_tune_jetclass.sh)
DISTILL=0
FINETUNE=1
PRETRAIN_TAG=pretrain_s
DATASET=jetclass
NUM_CLASSES=10
NUM_FEAT=9
EPOCH=100
LR=5e-5
LR_FACTOR=10
SAVE_TAG=fine_tune_jetclass_s_pretrain_s
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
