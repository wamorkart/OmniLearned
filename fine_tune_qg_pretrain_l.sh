#!/bin/bash
# Fine-tune the large pretrained model (pretrain_l, 460M params) on the
# quark/gluon (qg) dataset, 2-class classifier.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 1 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash fine_tune_qg_pretrain_l.sh

module load conda
conda activate ol_distill
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cmd="omnilearned train \
  -o /pscratch/sd/m/mbenyas/ \
  --save-tag fine_tune_qg_pretrain_l \
  --pretrain-tag pretrain_l \
  --fine-tune \
  --dataset qg --mode classifier --num-classes 2 \
  --path /global/cfs/cdirs/m4567/www/ \
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
