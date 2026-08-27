#!/bin/bash
# Fine-tune the JetClass-KD small student on top tagging (2-class).
# Starts from: best_model_distill_jetclass_s_pretrain_l_a05_T4_100epochs.pt
#   (small student distilled from pretrain_l teacher on 10-class JetClass, a=0.5, T=4, 100ep)
# Produces:    best_model_fine_tune_top_distill_jetclass_s_pretrain_l_a05_T4.pt
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 1 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash fine_tune_top_distill_jetclass_pretrain_l.sh

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag fine_tune_top_distill_jetclass_s_pretrain_l_a05_T4 \
  --pretrain-tag distill_jetclass_s_pretrain_l_a05_T4_100epochs \
  --fine-tune \
  --dataset top --mode classifier --num-classes 2 \
  --path /global/cfs/cdirs/m4567/www/ \
  --size small \
  --interaction \
  --local-interaction \
  --batch 128 --iterations 1000 --epoch 50 \
  --lr 5e-5 --lr-factor 10 --wd 0.5 \
  --num-workers 4 \
  --wandb --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
