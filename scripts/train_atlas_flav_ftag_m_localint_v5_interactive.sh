#!/bin/bash
# ATLAS b/c-tagging (atlas_flav, --mode ftag) fine-tune of pretrain_m --
# the UPSTREAM train.sh reference command, verbatim, plus --local-interaction,
# reading the dataset from the local /pscratch stage (the v4 I/O fix).
#
# This is the per-session script for the interactive resubmit loop
# (train_loop_atlas_flav_ftag_m_localint_v5.sh). Same experiment as the
# "ref16" run, differing only by: --path -> pscratch copy (not CFS),
# --num-workers left at the CLI default of 16 (ref16 forced 4).
#
# NOTE: deliberately the same HP combination as the failed v1
# (fine_tune_atlas_flav_m_localint, 2026-08-29: jet head collapsed to ln(4)):
#   global batch 4096 (16 GPU x 256), --lr 5e-5, --lr-factor 1.0, NO warmup.
# Run at explicit user request as the clean train.sh baseline. If the
# jet-class loss blows up in epoch 1, the mitigations to pull are
# --warmup-epoch 2, --lr 2e-5, --lr-factor 5-10 (see
# atlas-flav-btagging-progress memory).
#
# Save-tag fine_tune_atlas_flav_m_localint_v5 -- fresh from pretrain_m; the
# loop's later sessions --resume last_model_..._v5.pt.
#
# CRITICAL when scoring results: label 2 = b, label 0 = light (NOT the
# standard GN2 convention) -- see atlas-flav-btagging-progress memory.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash train_atlas_flav_ftag_m_localint_v5_interactive.sh
# The 30-epoch medium schedule far exceeds one 240-min session, so drive it
# with train_loop_atlas_flav_ftag_m_localint_v5.sh instead.

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
  --save-tag fine_tune_atlas_flav_m_localint_v5 \
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
