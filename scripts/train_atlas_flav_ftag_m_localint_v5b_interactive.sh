#!/bin/bash
# Seed replicate of the atlas_flav ftag "v5" fine-tune. IDENTICAL recipe to
# v5 (upstream train.sh reference: global batch 4096 = 16 GPU x 256, --lr
# 5e-5, --lr-factor 1.0, NO warmup, --local-interaction, pscratch dataset) --
# only the save-tag differs, so this run draws a fresh random seed (no
# --seed flag is passed here either, same as v5) and independently DDP data
# order/init.
#
# WHY: v5 completed all 30 epochs clean (2026-09-10, best light-rej result
# so far, ~1.47x off paper), but this exact HP combo is also what the v1 run
# (2026-08-29) collapsed under -- jet head pinned to constant-logit ln(4).
# Before trusting v5's numbers as "this recipe works", confirm a second
# independent run also avoids the collapse. See atlas-flav-btagging-progress
# memory for the full v1/v3/v4/v5 history.
#
# Save-tag fine_tune_atlas_flav_m_localint_v5b -- fresh from pretrain_m
# (separate checkpoint namespace from v5, does not touch/resume v5's
# checkpoints). CRITICAL when scoring: label 2 = b, label 0 = light (NOT
# GN2 convention) -- see atlas-flav-btagging-progress memory.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash train_atlas_flav_ftag_m_localint_v5b_interactive.sh
# Drive with train_loop_atlas_flav_ftag_m_localint_v5b.sh (30-epoch medium
# schedule exceeds one 240-min session).

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export HDF5_USE_FILE_LOCKING=FALSE

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag fine_tune_atlas_flav_m_localint_v5b \
  --dataset atlas_flav --mode ftag \
  --path /pscratch/sd/t/twamorka/omnilearned/datasets/ \
  --size medium \
  --fine-tune --pretrain-tag pretrain_m --lr-factor 1.0 \
  --epoch 30 --lr 5e-5 --wd 0.1 \
  --use-add --num-add 17 --num-classes 4 --num-gen-classes 8 \
  --batch 256 --iterations 2000 \
  --conditional --num-cond 4 --interaction --local-interaction \
  --wandb --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
