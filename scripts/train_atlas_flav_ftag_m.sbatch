#!/bin/bash
# ATLAS b/c-tagging (atlas_flav, --mode ftag) teacher: fine-tune pretrain_m
# into an ftag classifier. This is the checkpoint the paper reports b-/c-tag
# numbers for, but never published the fine-tuned weights themselves (only
# best_model_pretrain_{s,m,l}.pt are on portal.nersc.gov/cfs/m4567/checkpoints/)
# -- see atlas-flav-btagging-progress memory for the full blocker writeup.
# Any future KD-to-DeepSets work on b-tagging needs this checkpoint as the
# teacher, so it comes first.
#
# Hyperparameters are the upstream author's own reference command, copied
# verbatim from train.sh's commented-out fine_tune_atlas_flav_m line (medium
# size, lr 5e-5, wd 0.1, 30 epochs, batch 256, iterations 2000, --use-add
# --num-add 17 --num-classes 4 --num-gen-classes 8 --conditional --num-cond 4
# --interaction) -- only -o/--path/--save-tag/--wandb/--resuming changed to
# match this project's conventions.
#
# --fine-tune --pretrain-tag pretrain_m auto-fetches best_model_pretrain_m.pt
# from portal.nersc.gov if not already present locally (utils.py's
# restore_checkpoint fine_tune branch) -- no manual staging needed.
#
# CRITICAL when scoring results: label 2 = b, label 0 = light (NOT the
# standard GN2 convention of 0=b/2=light) -- see atlas-flav-btagging-progress
# memory for how this was derived.
#
# Submitted to -q regular (not interactive) because the 2-slot interactive
# cap is already consumed by the DeepSets FPGA work
# (see [[perlmutter_interactive_parallel]]).
#
# Submit with: sbatch train_atlas_flav_ftag_m.sbatch
#
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH -J train_atlas_flav_ftag_m
#SBATCH -o /pscratch/sd/t/twamorka/omnilearned/logs/atlas_flav_ftag_m/slurm_%j.out

set -uo pipefail

CHAIN_REMAINING=${CHAIN_REMAINING:-3}
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/atlas_flav_ftag_m
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
