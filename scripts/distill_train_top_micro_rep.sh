#!/bin/bash
# Distill large fine-tuned top-tagging teacher -> micro student on top tagging.
# Generalized version of distill_train_top_micro.sh for repeated runs — tag set
# by DISTILL_SAVE_TAG env var so multiple independent-seed reps can share this
# script (no --seed flag exists in the CLI, so each launch gets a fresh random
# init/shuffle order from system entropy; reps are true independent replicates).
#
# Teacher: best_model_fine_tune_top_l.pt (large model fine-tuned on top tagging)
# Student: micro PET2 (4 transformer layers, base_dim=64), trained from scratch
#
# Usage (set tag before running, e.g. via distill_loop_top_micro_rep.sh):
#   DISTILL_SAVE_TAG=distill_top_micro_scratch_a05_T4_r1 bash distill_train_top_micro_rep.sh

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

TEACHER_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_l
SAVE_TAG="${DISTILL_SAVE_TAG:?DISTILL_SAVE_TAG must be set}"

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag $SAVE_TAG \
  --dataset top --mode classifier --num-classes 2 \
  --path /global/cfs/cdirs/m4567/www/ \
  --size micro \
  --interaction \
  --local-interaction \
  --batch 128 --iterations 1000 --epoch 50 \
  --lr 5e-4 --wd 0.5 \
  --num-workers 4 \
  --distill \
  --teacher-labels-dir $TEACHER_DIR \
  --teacher-tag fine_tune_top_l \
  --distill-alpha 0.5 --distill-beta 0.5 --distill-t 4 \
  --wandb --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
