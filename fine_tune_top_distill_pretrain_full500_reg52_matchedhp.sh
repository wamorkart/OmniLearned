#!/bin/bash
# Fine-tune the 52-GPU KD-pretrained small student on top tagging (2-class),
# using the SAME fine-tune hyperparameters as the fine_tune_top_s CE-only
# baseline (train.sh line 49: lr=5e-6, lr-factor=5.0, wd=0.1, warmup-epoch=1,
# epoch=10) instead of the more aggressive recipe in
# fine_tune_top_distill_pretrain_full500_reg52.sh (lr=5e-5, wd=0.5, no
# warmup, epoch=50).
#
# Purpose: isolate whether the KD-pretrained checkpoint is fundamentally
# weaker than pretrain_s, or whether the mismatched/aggressive fine-tune
# recipe was the cause of its underperformance vs. fine_tune_top_s
# (94.17%/0.9866 vs 94.38%/0.9875 AUC). See [[distill-lazy-teacher-progress]].
#
# Starts from: best_model_distill_pretrain_s_scratch_a05_b05_T4_full500_reg52.pt
# Produces:    best_model_fine_tune_top_distill_pretrain_s_a05_b05_T4_full500_reg52_matchedhp.pt
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 120 --nodes 1 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash fine_tune_top_distill_pretrain_full500_reg52_matchedhp.sh

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag fine_tune_top_distill_pretrain_s_a05_b05_T4_full500_reg52_matchedhp \
  --pretrain-tag distill_pretrain_s_scratch_a05_b05_T4_full500_reg52 \
  --fine-tune \
  --dataset top --mode classifier --num-classes 2 \
  --path /global/cfs/cdirs/m4567/www/ \
  --size small \
  --interaction \
  --local-interaction \
  --batch 128 --iterations 1000 --epoch 10 \
  --lr 5e-6 --lr-factor 5.0 --wd 0.1 --warmup-epoch 1 \
  --num-workers 4 \
  --wandb --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
