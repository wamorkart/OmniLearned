# DeepSets/PFN top-tagging student, weight-decay ablation: wd 0.05 instead of
# the 0.5 default. One-variable change from top_deepsets_a05 (KD a0.5/b0.5/T4,
# teacher fine_tune_top_l, size=small).
#
# --wandb ON (inherited from _defaults.sh).
# Was: distill_train_top_deepsets.sh CONFIG=wd005.
ARCH=deep-sets
INTERACTION=0
LOCAL_INTERACTION=0
WD=0.05
SAVE_TAG=distill_top_deepsets_small_scratch_a05_wd005_T4
