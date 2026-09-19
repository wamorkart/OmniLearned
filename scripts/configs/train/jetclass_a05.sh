# Distil pretrain_l teacher -> PET2-small student on JetClass (10-class),
# KD a=b=0.5, T=4, 100 epochs, from scratch. teacher slice 2:12 maps the
# 210-class teacher's jetclass columns to local labels 0-9.
# (was distill_train_jetclass_pretrain_l.sh)
DATASET=jetclass
NUM_CLASSES=10
NUM_FEAT=9
EPOCH=100
TEACHER_TAG=pretrain_l
TEACHER_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion
EXTRA_FLAGS="--distill-teacher-slice 2:12"
SAVE_TAG=distill_jetclass_s_pretrain_l_a05_T4_100epochs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
