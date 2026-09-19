#!/bin/bash
# ATLAS b/c-tagging (atlas_flav, --mode ftag) teacher: fine-tune pretrain_m
# into an ftag classifier. Interactive-slot version of
# train_atlas_flav_ftag_m.sbatch (same command/hyperparameters) -- see that
# file's header for the full rationale (this is the missing teacher
# checkpoint the paper never published; see atlas-flav-btagging-progress
# memory). Run via the interactive queue instead of -q regular because a
# slot freed up after moving the DeepSets spread reps off it.
#
# --fine-tune --pretrain-tag pretrain_m auto-fetches best_model_pretrain_m.pt
# from portal.nersc.gov if not already present locally -- no manual staging.
#
# CRITICAL when scoring results: label 2 = b, label 0 = light (NOT the
# standard GN2 convention) -- see atlas-flav-btagging-progress memory.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash train_atlas_flav_ftag_m.sh
#
# For a long-running loop (30 epochs on a medium model likely exceeds one
# 240-min interactive session), use train_loop_atlas_flav_ftag_m.sh instead.

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag fine_tune_atlas_flav_m \
  --dataset atlas_flav --mode ftag \
  --path /global/cfs/cdirs/m4567/www/ \
  --size medium \
  --fine-tune --pretrain-tag pretrain_m --lr-factor 1.0 \
  --epoch 30 --lr 5e-5 --wd 0.1 \
  --use-add --num-add 17 --num-classes 4 --num-gen-classes 8 \
  --batch 256 --iterations 2000 \
  --conditional --num-cond 4 --interaction \
  --num-workers 4 \
  --wandb --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
