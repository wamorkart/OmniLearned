#!/bin/bash
# Stage 3 of the TAKD experiment: distill the MEDIUM teacher-assistant
# (trained in stage 1, logits generated in stage 2) into the same small
# student architecture used in the direct large->small T-sweep, so the two
# are directly comparable.
#
# Same alpha=0, beta=1, T=4, and all other hyperparameters as
# distill_top_small_scratch_a00_b10_T4 -- only the teacher differs (medium
# TA instead of the large model directly).
#
# Only run this AFTER save_teacher_logits_top_medium.sh (stage 2) has
# produced the companion_distill_top_medium H5 files.
#
# Called by distill_loop_top_small_via_medium.sh.

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

SAVE_TAG=distill_top_small_via_medium_a00_b10_T4
TEACHER_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_distill_top_medium
TEACHER_TAG=distill_top_medium_scratch_a00_b10_T4

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
  --teacher-tag $TEACHER_TAG \
  --distill-alpha 0.0 --distill-beta 1.0 --distill-t 4 \
  --wandb --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
