#!/bin/bash
# ATLAS b/c-tagging (atlas_flav, --mode ftag) fine-tune of pretrain_m --
# same recipe as v5/v5b (upstream train.sh reference: global batch 4096 =
# 16 GPU x 256, --lr 5e-5, --lr-factor 1.0, NO warmup, --local-interaction,
# pscratch dataset), but now with the get_loss ftag classification-loss
# weight change: loss = loss + 0.1*loss_class instead of loss + loss_class
# (src/omnilearned/utils.py get_loss, mode == "ftag" only -- all other modes
# unaffected). No CLI flag for this; it's unconditional for --mode ftag as
# of this utils.py change.
#
# WHY: down-weighting the classification term relative to whatever else is
# in the ftag loss composition, to test whether it changes convergence
# behavior / class balance (e.g. tau) versus the v5/v5b baseline. See
# atlas-flav-btagging-progress memory for the v1/v3/v4/v5/v5b history this
# is compared against.
#
# Save-tag fine_tune_atlas_flav_m_localint_v7 -- fresh from pretrain_m
# (separate checkpoint namespace, does not touch/resume v5/v5b/v6
# checkpoints). CRITICAL when scoring: label 2 = b, label 0 = light (NOT
# GN2 convention) -- see atlas-flav-btagging-progress memory.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash train_atlas_flav_ftag_m_localint_v7_interactive.sh
# Drive with train_loop_atlas_flav_ftag_m_localint_v7.sh (30-epoch medium
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
  --save-tag fine_tune_atlas_flav_m_localint_v7 \
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
