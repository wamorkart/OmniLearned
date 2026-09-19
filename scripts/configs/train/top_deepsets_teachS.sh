# DeepSets/PFN top-tagging student distilled from the SMALL fine-tuned teacher
# (fine_tune_top_s) instead of the large one -- teacher-capacity ablation.
# One-variable change from top_deepsets_a05 (TEACHER_TAG). KD a0.5/b0.5/T4,
# size=small, wd 0.5.
#
# TEACHER_DIR auto-derives to $TEACHER_ROOT/companion_fine_tune_top_s in
# run_train.sh, so the companion logits for the small teacher must exist there.
#
# --wandb ON (inherited from _defaults.sh).
# Was: distill_train_top_deepsets.sh CONFIG=teachS.
ARCH=deep-sets
INTERACTION=0
LOCAL_INTERACTION=0
TEACHER_TAG=fine_tune_top_s
SAVE_TAG=distill_top_deepsets_small_scratch_teachS_a05_T4
