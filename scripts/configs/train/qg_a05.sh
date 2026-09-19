# Distil the large qg fine-tuned teacher (fine_tune_qg_pretrain_l) -> PET2-small
# student on quark/gluon, KD a=b=0.5, T=4, from scratch.
# (was distill_train_qg.sh, from the distill_dev branch)
#
# Needs the teacher companion logits built first, at
# $TEACHER_ROOT/companion_fine_tune_qg_pretrain_l (see build_teacher_h5*.sh /
# save_teacher_logits_top.sh for the pattern). size=small, interaction +
# local-interaction, otherwise the shared base recipe (batch 128, 50 epochs,
# lr 5e-4, wd 0.5).
DATASET=qg
NUM_CLASSES=2
SIZE=small
INTERACTION=1
LOCAL_INTERACTION=1
TEACHER_TAG=fine_tune_qg_pretrain_l
EXTRA_FLAGS="--use-pid"
SAVE_TAG=distill_qg_small_scratch_a05_T4
