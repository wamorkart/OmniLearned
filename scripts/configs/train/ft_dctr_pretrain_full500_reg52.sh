# Fine-tune the 52-GPU KD-pretrained student on the dctr reweighting task.
# --conditional --num-cond 4 (jet-level kinematics), --num-feat 5 (dctr's
# per-particle array is (N,51,5)). (was fine_tune_dctr_distill_pretrain_full500_reg52.sh)
DISTILL=0
FINETUNE=1
PRETRAIN_TAG=distill_pretrain_s_scratch_a05_b05_T4_full500_reg52
DATASET=dctr
NUM_FEAT=5
LR=5e-5
LR_FACTOR=10
EXTRA_FLAGS="--conditional --num-cond 4"
SAVE_TAG=fine_tune_dctr_distill_pretrain_s_a05_b05_T4_full500_reg52
