#!/bin/bash
# ATLAS b/c-tagging (atlas_flav, --mode ftag), KEEPING --local-interaction --
# fifth attempt. This run deliberately reverts the training hyper-parameters
# back to the upstream reference (scripts/train.sh, the commented-out
# "fine_tune_atlas_flav_m" line) and changes ONLY:
#   1. + --local-interaction
#   2. --path -> the local /pscratch stage of the dataset (not CFS)
#
# So vs. the train.sh reference command the option set is identical --
#   --dataset atlas_flav --mode ftag --size medium
#   --fine-tune --pretrain-tag pretrain_m --lr-factor 1.0
#   --epoch 30 --lr 5e-5 --wd 0.1
#   --use-add --num-add 17 --num-classes 4 --num-gen-classes 8
#   --batch 256 --iterations 2000
#   --conditional --num-cond 4 --interaction
# -- with NO --warmup-epoch and lr 5e-5 (v3/v4 used --warmup-epoch 2 / lr 2e-5;
# those are dropped here). --num-workers is left unset so it takes the CLI
# default of 16 (cli.py), matching train.sh which also omits it.
#
# Operational-only additions (not "training" options):
#   - unique save-tag fine_tune_atlas_flav_m_localint_v5: fresh start from
#     pretrain_m. Does NOT resume any earlier checkpoint. NB the bare
#     "fine_tune_atlas_flav_m" tag from train.sh already exists in the
#     checkpoint dir and is a NON-local-interaction model -- reusing it with
#     --resuming would load the wrong architecture, hence the new tag.
#   - --wandb (project standing rule: always log runs to wandb)
#   - --resuming so the self-resubmit chain below picks up the last checkpoint
#     on requeue.
#
# SLURM geometry is 4 nodes x 4 = 16 GPUs (global batch 16*256 = 4096) --
# double the 8-GPU / 2048 geometry of v2/v3/v4, for wall-clock. train.sh does
# not pin a node count. Drop to 1 node / 1 task if you want the paper's
# literal global batch 256 (~16x wall-clock).
#
# CRITICAL when scoring results: label 2 = b, label 0 = light (NOT the
# standard GN2 convention of 0=b/2=light) -- see atlas-flav-btagging-progress
# memory for how this was derived.
#
# Submit with: sbatch scripts/train_atlas_flav_ftag_m_localint_v5.sbatch
#
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH -J train_atlas_flav_ftag_m_localint_v5
#SBATCH -o /pscratch/sd/t/twamorka/omnilearned/logs/atlas_flav_ftag_m_localint_v5/slurm_%j.out

set -uo pipefail

CHAIN_REMAINING=${CHAIN_REMAINING:-3}
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/atlas_flav_ftag_m_localint_v5
mkdir -p "$LOG_DIR"

export PATH=/global/homes/t/twamorka/omnilearned-clean/env/bin:$PATH

# HDF5 file locking causes slow/hanging opens over Lustre-backed CFS; already
# worked around this way elsewhere in the repo.
export HDF5_USE_FILE_LOCKING=FALSE

export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Flight-recorder auto-dump on collective timeout.
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_FR_BUFFER_SIZE=2000
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
DUMP_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/nccl_trace
mkdir -p "$DUMP_DIR"
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$DUMP_DIR/dump_${SLURM_JOB_ID}_rank_"

export DDP_TIMEOUT_MIN=90

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

echo "[$(date '+%F %T')] job ${SLURM_JOB_ID}, chain remaining (incl. this run) ${CHAIN_REMAINING}"

set +e
srun -l -u bash -c "source export_ddp.sh; $cmd"
EXIT=$?
set -e

echo "[$(date '+%F %T')] job ${SLURM_JOB_ID} exited with code ${EXIT}" | tee -a "$LOG_DIR/summary.log"

if [ "$EXIT" -eq 0 ]; then
    echo "Training completed. Not resubmitting." | tee -a "$LOG_DIR/summary.log"
elif [ "$CHAIN_REMAINING" -gt 1 ]; then
    NEXT=$((CHAIN_REMAINING - 1))
    echo "Non-zero exit, resubmitting (chain remaining after this: ${NEXT})..." \
        | tee -a "$LOG_DIR/summary.log"
    sbatch --export=ALL,CHAIN_REMAINING="$NEXT" "$0"
else
    echo "Chain exhausted (CHAIN_REMAINING reached 0) without a clean exit -- check logs." \
        | tee -a "$LOG_DIR/summary.log"
fi
