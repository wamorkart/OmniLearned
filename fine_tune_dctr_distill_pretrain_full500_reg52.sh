#!/bin/bash
# Fine-tune the 52-GPU KD-pretrained small student on the dctr reweighting/
# unfolding task (Herwig vs Pythia Z+jets, arXiv:2404.16091 Sec. VII).
# Starts from: best_model_distill_pretrain_s_scratch_a05_b05_T4_full500_reg52.pt
#   (small student, pretrain-KD, a=0.5/b=0.5, T=4, full 154-epoch pass at
#   world_size=52, job 56540549, completed 2026-08-10 -- see
#   [[distill-lazy-teacher-progress]])
# Produces:    best_model_fine_tune_dctr_distill_pretrain_s_a05_b05_T4_full500_reg52.pt
#
# --conditional --num-cond 4: dctr's h5 files carry a "global" (N,4) array
# alongside "data"/"pid" -- the paper's full-phase-space reweighting
# conditions on these jet-level kinematic features. Not used for top-tagging
# (no global array there), so this is new relative to the top fine-tune
# script this was copied from.
#
# --num-feat 5: dctr's "data" array is (N,51,5) per-particle -- one more
# kinematic feature per particle than the (N,150,4) used by top/pretrain.
# Without this override, InputBlock builds its norm/MLP for the num-feat=4
# default and crashes in DynamicTanh (layers.py) on the first forward pass:
# "RuntimeError: The size of tensor a (5) must match the size of tensor b (4)
# at non-singleton dimension 2". Fine-tune's checkpoint loader (utils.py
# filter_partial_model) already skips shape-mismatched layers rather than
# erroring, so the input embedding just gets freshly initialized for the
# 5-feature schema while the rest of the KD-pretrained backbone still loads.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 1 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash fine_tune_dctr_distill_pretrain_full500_reg52.sh

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag fine_tune_dctr_distill_pretrain_s_a05_b05_T4_full500_reg52 \
  --pretrain-tag distill_pretrain_s_scratch_a05_b05_T4_full500_reg52 \
  --fine-tune \
  --dataset dctr --mode classifier --num-classes 2 \
  --conditional --num-cond 4 \
  --num-feat 5 \
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
