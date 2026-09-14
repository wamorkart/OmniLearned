#!/bin/bash
# ATLAS b/c-tagging (atlas_flav, --mode ftag) fine-tune of pretrain_m --
# same recipe as v5, but with --lr-factor 5 instead of 1.0.
#
# v5 (fine_tune_atlas_flav_m_localint_v5) ran the upstream train.sh reference
# HPs verbatim (--lr-factor 1.0) and converged cleanly (30/30 epochs, exit 0),
# but came in below the OmniLearned paper's own OmniLearned-m numbers on
# Table IV -- most strikingly on tau-jet rejection (203 vs 493 @ b-eff=70%;
# 4.3 vs 28.4 @ c-eff=30%). Root cause identified by reading the paper
# (arXiv:2510.24066, Sec. II, page 5): "The fine-tuning of OmniLearned across
# different datasets and tasks is performed by setting the learning rate of
# all network weights to be a factor 5 smaller than the output layer." That
# is exactly --lr-factor 5 -- get_param_groups (src/omnilearned/utils.py:608)
# applies lr*lr_factor to classifier.out, generator, body.local_physics
# (== --local-interaction, entirely new/randomly-init here since it wasn't in
# pretrain_m), body.add_embed, body.cond, body.embed, body.interaction, and
# plain lr to everything else. v5's lr_factor=1.0 trained all those new/
# randomly-initialized layers at the SAME rate as the pretrained body --
# 5x slower than the paper's own recipe -- which plausibly starves exactly
# the rare tau class (3.65% of the sample) that depends most on the
# classifier/generator heads adapting fast from few examples.
#
# Everything else is identical to v5: pretrain_m, --local-interaction,
# pscratch dataset stage, 30-epoch medium schedule, lr 5e-5, wd 0.1, no
# warmup, global batch 4096 (16 GPU x 256).
#
# CRITICAL when scoring results: label 2 = b, label 0 = light (NOT the
# standard GN2 convention) -- see atlas-flav-btagging-progress memory.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash train_atlas_flav_ftag_m_localint_v6_interactive.sh
# The 30-epoch medium schedule far exceeds one 240-min session, so drive it
# with train_loop_atlas_flav_ftag_m_localint_v6.sh instead.

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
  --save-tag fine_tune_atlas_flav_m_localint_v6 \
  --dataset atlas_flav --mode ftag \
  --path /pscratch/sd/t/twamorka/omnilearned/datasets/ \
  --size medium \
  --fine-tune --pretrain-tag pretrain_m --lr-factor 5 \
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
