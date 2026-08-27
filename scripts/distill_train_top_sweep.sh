#!/bin/bash
# Parameterized distillation train script for alpha/beta sweep on top tagging.
# Called by distill_loop_top_sweep.sh; reads ALPHA, BETA, DISTILL_T, SAVE_TAG from env.
#
# Example (manual):
#   ALPHA=0.25 BETA=0.75 DISTILL_T=4 SAVE_TAG=distill_top_small_scratch_a025_b075_T4 \
#     bash distill_train_top_sweep.sh

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

ALPHA=${ALPHA:-0.5}
BETA=${BETA:-0.5}
DISTILL_T=${DISTILL_T:-4}
SAVE_TAG=${SAVE_TAG:-distill_top_small_scratch_a05_b05_T4}
TEACHER_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_l

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag $SAVE_TAG \
  --dataset top --mode classifier --num-classes 2 \
  --path /global/cfs/cdirs/m4567/www/ \
  --size small \
  --interaction \
  --local-interaction \
  --batch 128 --iterations 1000 --epoch 50 \
  --lr 5e-4 --wd 0.5 \
  --num-workers 4 \
  --distill \
  --teacher-labels-dir $TEACHER_DIR \
  --teacher-tag fine_tune_top_l \
  --distill-alpha $ALPHA --distill-beta $BETA --distill-t $DISTILL_T \
  --wandb --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
