#!/bin/bash
# Fine-tune the PTQ'd (bf16-rounded) 52-GPU KD-pretrained small student on top
# tagging (2-class). Second half of the distill -> quantize -> fine-tune study;
# compare against fine_tune_top_distill_pretrain_s_a05_b05_T4_full500_reg52
# (same lineage, no quantization step) to see whether PTQ rounding before
# fine-tune costs anything once the fine-tune has a chance to recover it.
# Starts from: best_model_distill_pretrain_s_scratch_a05_b05_T4_full500_reg52_bf16ptq.pt
#   (produced by quantize_top_distill_pretrain_full500_reg52.sh)
# Produces:    best_model_fine_tune_top_distill_pretrain_s_a05_b05_T4_full500_reg52_bf16ptq.pt
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 1 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash fine_tune_top_distill_pretrain_full500_reg52_bf16ptq.sh

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag fine_tune_top_distill_pretrain_s_a05_b05_T4_full500_reg52_bf16ptq \
  --pretrain-tag distill_pretrain_s_scratch_a05_b05_T4_full500_reg52_bf16ptq \
  --fine-tune \
  --dataset top --mode classifier --num-classes 2 \
  --path /global/cfs/cdirs/m4567/www/ \
  --size small \
  --interaction \
  --local-interaction \
  --batch 128 --iterations 1000 --epoch 50 \
  --lr 5e-5 --lr-factor 10 --wd 0.5 \
  --num-workers 4 \
  --use-amp --amp-dtype bf16 \
  --wandb --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
