#!/bin/bash
# Variant of distill_train_top_medium.sh (TAKD stage 1: large->medium) using
# alpha=0.5, beta=0.5, T=4 instead of alpha=0, beta=1.
#
# Batch size reduced from 128 to 64 vs. the a00_b10_T4 version: that run
# OOM'd on the very first forward pass (CUDA OOM in the local-interaction
# MLP, layers.py:267/444) -- medium size + --local-interaction produces
# pairwise (NxN) interaction tensors that don't fit in 40GB at batch 128.
#
# Called by distill_loop_top_medium_a05_b05_T4.sh.

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

SAVE_TAG=distill_top_medium_scratch_a05_b05_T4
TEACHER_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_l

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag $SAVE_TAG \
  --dataset top --mode classifier --num-classes 2 \
  --path /global/cfs/cdirs/m4567/www/ \
  --size medium \
  --interaction \
  --local-interaction \
  --batch 64 --iterations 1000 --epoch 50 \
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
