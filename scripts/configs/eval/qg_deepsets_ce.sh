# CE-only DeepSets-small qg student (train_qg_deepsets_small_ce_scratch), test
# split. Counterpart to eval/qg_deepsets_a05.sh -- same architecture and eval
# settings, only the checkpoint differs, so the two evals are directly
# comparable for the "KD vs plain CE" question.
# Trained by configs/train/qg_deepsets_ce.sh.
CHECKPOINT_DIR=/pscratch/sd/m/mbenyas/OmniLearned/checkpoints/
EVAL_ROOT=/pscratch/sd/m/mbenyas/OmniLearned/eval
ARCH=deep-sets
DATASET=qg
NUM_CLASSES=2
SIZE=small
INTERACTION=1
LOCAL_INTERACTION=0
EXTRA_FLAGS="--use-pid"
SAVE_TAG=train_qg_deepsets_small_ce_scratch
