#!/bin/bash
# ATLAS b/c-tagging (atlas_flav, --mode ftag) fine-tune of pretrain_m --
# the UPSTREAM train.sh reference command, verbatim, plus --local-interaction.
#
# Rationale: pretrain_m carries local-interaction weights
# (local_physics.mlp.fc1 = (1024, 7) = input_dim(4)+3), so --local-interaction
# is REQUIRED for a clean full-body load -- an --interaction-only fine-tune
# silently drops that layer and random-inits it. This run establishes the
# clean-load reference at the paper's exact HPs and full 16-GPU throughput.
#
# NOTE: this is deliberately the same HP combination as the failed v1
# (fine_tune_atlas_flav_m_localint, 2026-08-29: jet head collapsed to ln(4)):
#   global batch 4096 (16 GPU x 256), --lr 5e-5, --lr-factor 1.0, NO warmup.
# Run at explicit user request as a control / baseline. If the jet-class loss
# blows up in epoch 1 again, the mitigations to pull are --warmup-epoch 2,
# --lr 2e-5, --lr-factor 5-10 (see atlas-flav-btagging-progress memory).
#
# Fresh save-tag fine_tune_atlas_flav_m_localint_ref16 -- does NOT touch the
# stale fine_tune_atlas_flav_m (Aug, --interaction-only, different arch) or
# the _localint_v3 chain.
#
# CRITICAL when scoring results: label 2 = b, label 0 = light (NOT the
# standard GN2 convention) -- see atlas-flav-btagging-progress memory.
#
# --fine-tune --pretrain-tag pretrain_m auto-fetches best_model_pretrain_m.pt
# from portal.nersc.gov if not already present locally.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash train_atlas_flav_ftag_m_localint_ref16.sh
#
# The 30-epoch medium schedule far exceeds one 240-min interactive session,
# so drive it with train_loop_atlas_flav_ftag_m_localint_ref16.sh instead.

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
  --save-tag fine_tune_atlas_flav_m_localint_ref16 \
  --dataset atlas_flav --mode ftag \
  --path /global/cfs/cdirs/m4567/www/ \
  --size medium \
  --fine-tune --pretrain-tag pretrain_m --lr-factor 1.0 \
  --epoch 30 --lr 5e-5 --wd 0.1 \
  --use-add --num-add 17 --num-classes 4 --num-gen-classes 8 \
  --batch 256 --iterations 2000 \
  --conditional --num-cond 4 --interaction --local-interaction \
  --num-workers 4 \
  --wandb --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
