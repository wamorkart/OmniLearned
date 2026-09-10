# Distil the large qg fine-tuned teacher (fine_tune_qg_pretrain_l) -> DeepSets/PFN
# student on quark/gluon. KD a=b=0.5, T=4, size=small, from scratch.
#
# = qg_a05.sh but with the DeepSets student arch (Phi-embed + masked pool +
# rho-MLP, no attention) instead of PET2-small, so the interaction flags are
# turned off (see top_deepsets_a05.sh for the same pattern on top tagging).
#
# Needs the teacher companion logits built first, at
# $TEACHER_ROOT/companion_fine_tune_qg_pretrain_l -- run
# scripts/save_teacher_logits_qg.sh. Batch 128 / 50 epochs / lr 5e-4 / wd 0.5
# all come from _defaults.sh unchanged.
DATASET=qg
NUM_CLASSES=2
SIZE=small
ARCH=deep-sets
INTERACTION=0
LOCAL_INTERACTION=0
TEACHER_TAG=fine_tune_qg_pretrain_l
EXTRA_FLAGS="--use-pid"
SAVE_TAG=distill_qg_deepsets_small_scratch_a05_T4
