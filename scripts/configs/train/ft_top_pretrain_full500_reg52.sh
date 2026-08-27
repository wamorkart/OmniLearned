# Fine-tune the 52-GPU KD-pretrained (a=b=0.5, T=4, full500) student on top tagging.
# (was fine_tune_top_distill_pretrain_full500_reg52.sh)
DISTILL=0
FINETUNE=1
PRETRAIN_TAG=distill_pretrain_s_scratch_a05_b05_T4_full500_reg52
LR=5e-5
LR_FACTOR=10
SAVE_TAG=fine_tune_top_distill_pretrain_s_a05_b05_T4_full500_reg52
