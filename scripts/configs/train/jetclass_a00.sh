# Pure-KD (a=0, b=1) variant of jetclass_a05.
# (was distill_train_jetclass_a00.sh)
DATASET=jetclass
NUM_CLASSES=10
NUM_FEAT=9
EPOCH=100
ALPHA=0.0
BETA=1.0
TEACHER_TAG=pretrain_l
TEACHER_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion
EXTRA_FLAGS="--distill-teacher-slice 2:12"
SAVE_TAG=distill_jetclass_s_pretrain_l_a00_b10_T4_100epochs
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
