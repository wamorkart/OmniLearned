#!/bin/bash
# ATLAS b/c-tagging (atlas_flav, --mode ftag) teacher, KEEPING
# --local-interaction -- third attempt. v2 (fine_tune_atlas_flav_m_localint_v2,
# 8 GPUs, global batch 2048) avoided the outright ln(4) collapse seen in v1
# but still showed early loss spikes (class val loss 1.3/2.7/4.2 at
# epoch 2/5/6) before recovering, and its epoch-15 timeout checkpoint scored
# ~1 order of magnitude short of the paper's light-jet rejection -- likely a
# bad basin from those early spikes, not just an undertrained model (its
# class val loss at ep15, 0.724, was already worse than the healthy 1-epoch
# fine_tune_atlas_flav_m run's 0.667 after a single epoch). See
# atlas-flav-btagging-progress memory.
#
# What changed here vs. localint_reg8 (v2):
#   - unique save-tag: fine_tune_atlas_flav_m_localint_v3 (fresh start from
#     pretrain_m, does NOT resume the v2 bad-basin checkpoint)
#   - --warmup-epoch 2 : ramp LR from 0 over the first 2 epochs instead of
#     jumping straight to 5e-5. Grad-clip (max_norm=1.0) is already applied
#     unconditionally in train.py regardless of this flag -- it did not
#     prevent v2's spikes on its own, so warmup is the actual new lever here.
#   - --lr 2e-5 (was 5e-5): lower peak LR, on top of warmup, to reduce spike
#     risk further.
#   - same 8 GPUs (2 nodes x 4) / batch 256 per-GPU (global 2048) as v2, for
#     wall-clock reasons -- not dropping to the paper's literal global batch
#     256 (would need 1 GPU, ~8x wall-clock). If warmup+lower-LR still spikes,
#     that's the next lever to pull.
#
# --fine-tune --pretrain-tag pretrain_m auto-fetches best_model_pretrain_m.pt
# from portal.nersc.gov if not already present locally -- no manual staging.
#
# CRITICAL when scoring results: label 2 = b, label 0 = light (NOT the
# standard GN2 convention of 0=b/2=light) -- see atlas-flav-btagging-progress
# memory for how this was derived.
#
# Submit with: sbatch scripts/train_atlas_flav_ftag_m_localint_v3.sbatch
#
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH -J train_atlas_flav_ftag_m_localint_v3
#SBATCH -o /pscratch/sd/t/twamorka/omnilearned/logs/atlas_flav_ftag_m_localint_v3/slurm_%j.out

set -uo pipefail

CHAIN_REMAINING=${CHAIN_REMAINING:-3}
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/atlas_flav_ftag_m_localint_v3
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
  --save-tag fine_tune_atlas_flav_m_localint_v3 \
  --dataset atlas_flav --mode ftag \
  --path /global/cfs/cdirs/m4567/www/ \
  --size medium \
  --fine-tune --pretrain-tag pretrain_m --lr-factor 1.0 \
  --epoch 30 --lr 2e-5 --warmup-epoch 2 --wd 0.1 \
  --use-add --num-add 17 --num-classes 4 --num-gen-classes 8 \
  --batch 256 --iterations 2000 \
  --conditional --num-cond 4 --interaction --local-interaction \
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
