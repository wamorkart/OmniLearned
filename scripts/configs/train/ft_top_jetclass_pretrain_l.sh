# Fine-tune the JetClass-KD small student on top tagging.
# (was fine_tune_top_distill_jetclass_pretrain_l.sh)
DISTILL=0
FINETUNE=1
PRETRAIN_TAG=distill_jetclass_s_pretrain_l_a05_T4_100epochs
LR=5e-5
LR_FACTOR=10
SAVE_TAG=fine_tune_top_distill_jetclass_s_pretrain_l_a05_T4
