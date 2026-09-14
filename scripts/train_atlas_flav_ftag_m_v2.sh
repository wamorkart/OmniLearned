#!/bin/bash
# ATLAS b/c-tagging (atlas_flav, --mode ftag) teacher, attempt 2. Fine-tune
# pretrain_m into an ftag classifier -- SAME reference command as
# train_atlas_flav_ftag_m.sh, i.e. WITHOUT --local-interaction.
#
# Why a fresh save-tag (fine_tune_atlas_flav_m_v2):
#   - fine_tune_atlas_flav_m_localint (adds --local-interaction) FAILED: the
#     jet-flavour head collapsed to the uniform ln(4) solution -- eval on
#     4.27M test jets gave acc 49.9% (= light prior), every OvR AUC 0.5000,
#     raw logits a literal per-row constant. See atlas-flav-btagging-progress
#     memory.
#   - fine_tune_atlas_flav_m already has 1 epoch trained (no --local-interaction,
#     val_loss_class 0.667 -- a genuinely discriminating classifier), but this
#     is a clean-slate re-run: fresh optimizer/LR state, no dependency on that
#     older partial checkpoint.
#
# --fine-tune --pretrain-tag pretrain_m auto-fetches best_model_pretrain_m.pt
# from portal.nersc.gov if not already present locally -- no manual staging.
#
# CRITICAL when scoring results: label 2 = b, label 0 = light (NOT the
# standard GN2 convention) -- see atlas-flav-btagging-progress memory.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash train_atlas_flav_ftag_m_v2.sh
#
# The 30-epoch medium schedule far exceeds one 240-min interactive session,
# so drive it with train_loop_atlas_flav_ftag_m_v2.sh instead.

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
  --save-tag fine_tune_atlas_flav_m_v2 \
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
