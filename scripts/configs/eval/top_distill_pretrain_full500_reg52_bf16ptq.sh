# Pretrain-KD -> bf16 PTQ -> top fine-tune lineage. Compare vs
# top_distill_pretrain_full500_reg52 (same lineage, no PTQ step).
# (was evaluate_top_distill_pretrain_full500_reg52_bf16ptq.sh)
SAVE_TAG=fine_tune_top_distill_pretrain_s_a05_b05_T4_full500_reg52_bf16ptq
