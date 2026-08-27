# DeepSets/PFN top-tagging student, pure KD (DistillNet-style): alpha=0,
# beta=1, drop the ground-truth CE term and train purely to mimic the
# fine_tune_top_l teacher. T=4, size=small, wd 0.5.
#
# One-variable change from top_deepsets_a05 (a/b weighting). Matches the
# PET2-student T-sweep winner (AUC 0.9879 @ a00/b10) and DistillNet
# (arXiv:2311.12551). See EXPERIMENTS_deepsets_kd.md for the mislabeled-PET2
# retrain history.
#
# --wandb ON (inherited from _defaults.sh).
# Was: distill_train_top_deepsets.sh CONFIG=a00_b10.
ARCH=deep-sets
INTERACTION=0
LOCAL_INTERACTION=0
ALPHA=0.0
BETA=1.0
SAVE_TAG=distill_top_deepsets_small_scratch_a00_b10_T4
