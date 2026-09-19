# Fine-tune the 52-GPU KD-pretrained student on top tagging with the
# fine_tune_top_s baseline HPs (lr 5e-6, lr-factor 5, wd 0.1, warmup 1, ep 10).
# (was fine_tune_top_distill_pretrain_full500_reg52_matchedhp.sh)
DISTILL=0
FINETUNE=1
PRETRAIN_TAG=distill_pretrain_s_scratch_a05_b05_T4_full500_reg52
EPOCH=10
LR=5e-6
LR_FACTOR=5.0
WARMUP_EPOCH=1
WD=0.1
SAVE_TAG=fine_tune_top_distill_pretrain_s_a05_b05_T4_full500_reg52_matchedhp
