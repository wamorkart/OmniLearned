#!/bin/bash
# Distill large pretrained teacher -> small student on JetClass, α=0 β=1 (pure KD).
#
# Teacher: best_model_pretrain_l.pt (210-class pretrained)
# Teacher logits: companion/jetclass/{train,val}/*.h5, shape (N, 210)
# Slice: columns 2:12 (jetclass raw labels 2-11)
# Student: small PET2, 10-class classifier, trained from scratch
#
# Called by distill_loop_jetclass_a00.sh — do not run directly.

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TEACHER_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag distill_jetclass_s_pretrain_l_a00_b10_T4_100epochs \
  --dataset jetclass --mode classifier --num-classes 10 \
  --path /global/cfs/cdirs/m4567/www/ \
  --size small \
  --num-feat 9 \
  --interaction \
  --local-interaction \
  --batch 128 --iterations 1000 --epoch 100 \
  --lr 5e-4 --wd 0.5 \
  --num-workers 4 \
  --distill \
  --teacher-labels-dir $TEACHER_DIR \
  --teacher-tag pretrain_l \
  --distill-alpha 0.0 --distill-beta 1.0 --distill-t 4 \
  --distill-teacher-slice 2:12 \
  --wandb --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
