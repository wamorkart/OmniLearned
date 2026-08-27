# PET2-micro student WITHOUT the pairwise interaction matrix (FLOPs ablation).
# (was distill_train_top_micro_noint_rep.sh, which required $DISTILL_SAVE_TAG;
# replicates now: SAVE_TAG=..._r1 scripts/train.sh top_micro_noint_a05)
SIZE=micro
INTERACTION=0
LOCAL_INTERACTION=0
SAVE_TAG=distill_top_micro_noint_scratch_a05_T4
