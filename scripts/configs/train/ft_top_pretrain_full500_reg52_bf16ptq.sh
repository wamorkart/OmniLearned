# Fine-tune the bf16-PTQ'd 52-GPU KD-pretrained student on top tagging.
# (was fine_tune_top_distill_pretrain_full500_reg52_bf16ptq.sh)
DISTILL=0
FINETUNE=1
PRETRAIN_TAG=distill_pretrain_s_scratch_a05_b05_T4_full500_reg52_bf16ptq
LR=5e-5
LR_FACTOR=10
EXTRA_FLAGS="--use-amp --amp-dtype bf16"
SAVE_TAG=fine_tune_top_distill_pretrain_s_a05_b05_T4_full500_reg52_bf16ptq
