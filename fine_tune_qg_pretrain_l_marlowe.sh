#!/bin/bash
# Fine-tune the large pretrained model (pretrain_l, 460M params) on the
# quark/gluon (qg) dataset, 2-class classifier.
#
# Run inside a GPU interactive job:
# srun -N 1 -G 1 -A marlowe-m000255 -p preempt --time=00:30:00 --pty bash
# and then run bash fine_tune_qg_pretrain_l_marlowe.sh

conda activate /projects/m000255/miniconda/envs/ol_distill/

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SAVE_TAG=fine_tune_pretrain_l #distill_qg_small_scratch_a05_T4
DATASET=qg
DIR="/projects/m000255/mbenyas/output/${SAVE_TAG}_${DATASET}"
DESCRIPT_TAG=int

cmd="omnilearned train \
  -o ${DIR} \
  --save-tag ${SAVE_TAG}_${DATASET}_${DESCRIPT_TAG} \
  --pretrain-tag pretrain_l \
  --fine-tune \
  --dataset qg --mode classifier --num-classes 2 \
  --path /projects/m000255/twamorka/qg_datasets \
  --size large \
  --use-pid \
  --interaction \
  --batch 32 --epoch 15 --wd 10.0 --lr 1e-6 \
  --num-workers 4 \
  --wandb --resuming"
set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
