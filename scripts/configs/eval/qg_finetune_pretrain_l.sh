# Large qg fine-tuned model (fine_tune_qg_pretrain_l), test split.
# (was evaluate_fine_tune_qg_pretrain_l.sh, from the distill_dev branch)
#
# The distill_dev original also had a QUANTIZE=int8|int8dq|bf16 knob; that
# depends on the torchao path in evaluate.py which is not on this branch yet,
# so it is dropped here. Add it back once that lands.
DATASET=qg
NUM_CLASSES=2
SIZE=large
INTERACTION=1
LOCAL_INTERACTION=0
EXTRA_FLAGS="--use-pid"
SAVE_TAG=fine_tune_pretrain_l_qg_int_i300_e10
