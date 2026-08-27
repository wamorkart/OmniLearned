# Fine-tune the 52-GPU pure-KD (a=0, b=1, T=4, full500, v2) pretrained student on top tagging.
# (was fine_tune_top_distill_pretrain_full500_reg52_a00_b10_v2.sh)
DISTILL=0
FINETUNE=1
PRETRAIN_TAG=distill_pretrain_s_scratch_a00_b10_T4_full500_reg52_v2
LR=5e-5
LR_FACTOR=10
SAVE_TAG=fine_tune_top_distill_pretrain_s_a00_b10_T4_full500_reg52_v2
