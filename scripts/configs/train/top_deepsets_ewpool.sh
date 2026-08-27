# DeepSets/PFN top-tagging student with pT-weighted (energy-weighted) pooling
# instead of plain masked mean-pool. One-variable change from top_deepsets_a05
# (adds --energy-weighted-pool; KD a0.5/b0.5/T4, teacher fine_tune_top_l,
# size=small, wd 0.5).
#
# --wandb ON (inherited from _defaults.sh).
# Was: distill_train_top_deepsets.sh CONFIG=ewpool.
ARCH=deep-sets
INTERACTION=0
LOCAL_INTERACTION=0
EXTRA_FLAGS="--energy-weighted-pool"
SAVE_TAG=distill_top_deepsets_small_scratch_a05_ewpool_T4
