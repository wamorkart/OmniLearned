#!/bin/bash
# ATLAS b/c-tagging (atlas_flav, --mode ftag) teacher, KEEPING
# --local-interaction -- fourth attempt. Same training config as v3
# (fine_tune_atlas_flav_m_localint_v3): --warmup-epoch 2, --lr 2e-5, 8 GPUs
# (2 nodes x 4), batch 256/GPU (global 2048). v3's CFS-hosted runs were
# heavily I/O-bound: epoch time for the identical 2000-iter workload swung
# 4829s -> 12706s (2.6x), the signature of stalling on random-access reads
# from the shared community filesystem. See atlas-flav-btagging-progress.
#
# What changed here vs. v3:
#   - unique save-tag: fine_tune_atlas_flav_m_localint_v4 (fresh start from
#     pretrain_m, clean checkpoint namespace -- does NOT resume any v3
#     checkpoint)
#   - --path now points at the local /pscratch copy of the dataset
#     (/pscratch/sd/t/twamorka/omnilearned/datasets/atlas_flav, byte-identical
#     stage of /global/cfs/cdirs/m4567/www/atlas_flav) instead of CFS. pscratch
#     Lustre is tuned for parallel I/O and far less contended.
#   - --num-workers 12 (was 4): 4 dataloader workers per rank badly
#     under-fed the pipeline; a Perlmutter GPU node has 16 cores per rank.
#
# Everything else (LR schedule, warmup, wd, batch, iterations, class/gen
# counts, interaction flags) is unchanged from v3 so the loss curve stays
# comparable.
#
# --fine-tune --pretrain-tag pretrain_m auto-fetches best_model_pretrain_m.pt
# from portal.nersc.gov if not already present locally -- no manual staging.
#
# CRITICAL when scoring results: label 2 = b, label 0 = light (NOT the
# standard GN2 convention of 0=b/2=light) -- see atlas-flav-btagging-progress
# memory for how this was derived.
#
# Submit with: sbatch scripts/train_atlas_flav_ftag_m_localint_v4.sbatch
#
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH -J train_atlas_flav_ftag_m_localint_v4
#SBATCH -o /pscratch/sd/t/twamorka/omnilearned/logs/atlas_flav_ftag_m_localint_v4/slurm_%j.out

set -uo pipefail

CHAIN_REMAINING=${CHAIN_REMAINING:-3}
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/atlas_flav_ftag_m_localint_v4
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
  --save-tag fine_tune_atlas_flav_m_localint_v4 \
  --dataset atlas_flav --mode ftag \
  --path /pscratch/sd/t/twamorka/omnilearned/datasets/ \
  --size medium \
  --fine-tune --pretrain-tag pretrain_m --lr-factor 1.0 \
  --epoch 30 --lr 2e-5 --warmup-epoch 2 --wd 0.1 \
  --use-add --num-add 17 --num-classes 4 --num-gen-classes 8 \
  --batch 256 --iterations 2000 \
  --conditional --num-cond 4 --interaction --local-interaction \
  --num-workers 12 \
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
