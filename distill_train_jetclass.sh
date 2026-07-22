#!/bin/bash
# Train a small student from scratch on JetClass using offline KD against
# the large pretrained model (pretrain_l) teacher logits.
#
# Teacher logit source (NPZ shards):
#   /pscratch/sd/t/twamorka/omnilearned/teacher_logits/TEST/
#     outputs_pretrain_l_jetclass_{train,val}_chunk*_rank*.npz
#
# Companion H5 files (built via build_teacher_h5.py / run_teacher_h5.sh):
#   /pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion/jetclass/{train,val}/*.h5
#
# Invoked by distill_loop_jetclass.sh inside an salloc; do not run directly.
# The srun inside launches the DDP job across all allocated nodes.

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TEACHER_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag distill_jetclass_small_scratch_a05_T4_new \
  --dataset jetclass --mode classifier --num-classes 210 \
  --path /global/cfs/cdirs/m4567/www/ \
  --size small \
  --num-feat 9 \
  --interaction \
  --batch 128 --iterations 1000 --epoch 100 \
  --lr 5e-4 --wd 0.5 \
  --num-workers 4 \
  --distill \
  --teacher-labels-dir $TEACHER_DIR \
  --teacher-tag pretrain_l \
  --distill-alpha 0.5 --distill-beta 0.5 --distill-t 4 \
  --wandb --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
