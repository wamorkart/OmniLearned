#!/bin/bash
# CE-only fine-tune of pretrain_s -> 10-class JetClass classifier.
# Baseline to compare against distill_jetclass_s_pretrain_l_a05_T4_100epochs.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 1 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash fine_tune_jetclass.sh

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag fine_tune_jetclass_s_pretrain_s \
  --pretrain-tag pretrain_s \
  --fine-tune \
  --dataset jetclass --mode classifier --num-classes 10 \
  --path /global/cfs/cdirs/m4567/www/ \
  --size small \
  --num-feat 9 \
  --interaction \
  --local-interaction \
  --batch 128 --iterations 1000 --epoch 100 \
  --lr 5e-5 --lr-factor 10 --wd 0.5 \
  --num-workers 4 \
  --wandb --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
