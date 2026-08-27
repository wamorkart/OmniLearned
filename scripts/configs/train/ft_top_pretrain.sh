# Fine-tune the KD-pretrained PET2-small student on top tagging.
# (was fine_tune_top_distill_pretrain.sh)
DISTILL=0
FINETUNE=1
PRETRAIN_TAG=distill_pretrain_s_scratch_a05_T4
LR=5e-5
LR_FACTOR=10
SAVE_TAG=fine_tune_top_distill_pretrain_s_a05_T4
