#!/bin/bash
# Stage 1 of the large->medium->small teacher-assistant (TAKD) experiment on
# top tagging. Distills the existing large fine-tuned teacher
# (best_model_fine_tune_top_l.pt) straight into a MEDIUM student, using the
# same teacher logits already cached for the small-student T-sweep
# (companion_fine_tune_top_l) -- no new teacher inference needed for this
# stage.
#
# alpha=0, beta=1, T=4 fixed to match the reference point of the existing
# large->small sweep (distill_top_small_scratch_a00_b10_T4[,_r1,_r2]), so the
# eventual large->medium->small result is comparable to the direct
# large->small baseline.
#
# Called by distill_loop_top_medium.sh.

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

SAVE_TAG=distill_top_medium_scratch_a00_b10_T4
TEACHER_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_l

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag $SAVE_TAG \
  --dataset top --mode classifier --num-classes 2 \
  --path /global/cfs/cdirs/m4567/www/ \
  --size medium \
  --interaction \
  --local-interaction \
  --batch 128 --iterations 1000 --epoch 50 \
  --lr 5e-4 --wd 0.5 \
  --num-workers 4 \
  --distill \
  --teacher-labels-dir $TEACHER_DIR \
  --teacher-tag fine_tune_top_l \
  --distill-alpha 0.0 --distill-beta 1.0 --distill-t 4 \
  --wandb --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
