# DeepSets/PFN quark-gluon student, CE-only no-teacher control: DISTILL=0
# drops --distill and every --teacher-*/--distill-* arg (train.py gates the
# whole KD path behind `if distill:`, and passes teacher_labels_dir=None), so
# plain cross-entropy, no teacher companion files touched.
#
# The no-KD counterpart of qg_deepsets_a05.sh: identical architecture and
# recipe (deep-sets, size=small, interaction on / local-interaction off,
# --use-pid, batch 128, 1000 iterations, 50 epochs, lr 5e-4, wd 0.5), so the
# only difference from that run is the teacher. Isolates "KD vs plain CE" on
# qg the way top_deepsets_ce.sh does on top.
#
# --wandb ON (inherited from _defaults.sh).
OUTDIR=/pscratch/sd/m/mbenyas/OmniLearned/checkpoints/
ARCH=deep-sets
DATASET=qg
INTERACTION=1
LOCAL_INTERACTION=0
EXTRA_FLAGS="--use-pid"
DISTILL=0
SAVE_TAG=train_qg_deepsets_small_ce_scratch
