#!/bin/bash
# OmniFold iterative unfolding: Pythia21 (MC, reco+gen) vs Herwig (stand-in
# "data", reco only) closure benchmark. First real full-scale run -- only
# synthetic-data smoke tests had been done before (see
# omnifold-unfolding-progress memory). Both train/test_{pythia,herwig}.h5
# already exist in /pscratch/sd/t/twamorka/unfolding/ (herwig preprocessing
# job 57181770 completed 2026-08-17, real run was just never launched).
#
# sbatch (not interactive salloc, unlike train_omnifold_pythia_herwig.sh's
# own header) because both -q interactive slots are consumed by the DeepSets
# FPGA QAT work; this needs its own node so it runs fully in parallel with
# that and the atlas_flav ftag job -- see [[perlmutter_interactive_parallel]].
#
# --num-workers 0 is REQUIRED, not a default -- omnifold.py builds a fresh
# DataLoader on every Step1/Step2 call (10+ times per run) after wandb.init()
# already ran once; with num_workers>0 this would refork workers post-wandb
# on every step, reintroducing the documented wandb-fork/NCCL deadlock class.
# Do not "optimize" this away.
#
# Submit with: sbatch train_omnifold_pythia_herwig.sbatch
#
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH -J train_omnifold_pythia_herwig
#SBATCH -o /pscratch/sd/t/twamorka/omnilearned/logs/omnifold_pythia_herwig/slurm_%j.out

set -uo pipefail

CHAIN_REMAINING=${CHAIN_REMAINING:-3}
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/omnifold_pythia_herwig
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

cmd="omnilearned unfold \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag omnifold_pythia_herwig \
  --path /pscratch/sd/t/twamorka/unfolding \
  --num-feat 13 \
  --size small \
  --interaction \
  --local-interaction \
  --num-iter 5 \
  --patience 3 \
  --batch 512 --epoch 30 --warmup-epoch 1 \
  --lr 3e-5 --lr-factor 5 --wd 0.1 \
  --num-workers 0 \
  --wandb"

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
