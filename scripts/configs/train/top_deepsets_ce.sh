# DeepSets/PFN top-tagging student, CE-only no-teacher control: DISTILL=0
# drops --distill and every --teacher-*/--distill-* arg (train.py gates the
# whole KD path behind `if distill:`), so plain cross-entropy, no teacher
# companion files touched. Isolates "KD vs plain CE" from "architecture
# capacity" in the DeepSets-vs-PET2-small accuracy gap.
#
# size=small, wd 0.5, otherwise the shared base recipe.
# --wandb ON (inherited from _defaults.sh).
# Was: distill_train_top_deepsets.sh CONFIG=ce.
ARCH=deep-sets
INTERACTION=0
LOCAL_INTERACTION=0
DISTILL=0
SAVE_TAG=train_top_deepsets_small_ce_scratch
