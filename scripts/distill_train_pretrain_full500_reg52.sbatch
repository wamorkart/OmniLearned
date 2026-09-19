#!/bin/bash
# Pretrain-KD student run at 52 GPUs (13 nodes x 4, regular queue), following
# the same self-chaining sbatch pattern as
# pretrain_train_jetclass_l_2pass_localint.sbatch, not the salloc-in-a-loop
# pattern used for the -q interactive scripts (distill_loop_pretrain_merged.sh
# etc.) -- salloc is a foreground/interactive allocation whose parent process
# has to survive the whole run; sbatch submits a detached batch job the SLURM
# controller runs independently of any screen/terminal session.
#
# Standalone script + tag, deliberately separate from the live 16-GPU
# distill_pretrain_s_scratch_a05_b05_T4_full500 run
# (distill_train_pretrain_merged.sh / distill_loop_pretrain_merged.sh, job
# 56540200 as of 2026-08-09) -- see
# [[feedback-check-before-editing-live-scripts]].
#
# --epoch 154 x --iterations 1000 x --batch 128 x 52 gpus ~= 1.025e9 samples,
# matching the 16-GPU run's ~1-full-pass target over the ~1.06B-sample
# pretrain mixture (500*1000*128*16 ~= 1.024e9). --lr left at the default
# 5e-5 (same as the 16-GPU run) -- NOT scaled for the larger effective global
# batch (2048->6656), a deliberate choice, not an oversight.
#
# At ~438s/epoch-unit (measured at 16 GPUs; unconfirmed whether DDP
# all-reduce overhead changes at 52 -- first run at this world_size) the full
# 154 epochs is ~18.7h of pure compute, comfortably inside a single 48h
# regular-queue job -- CHAIN_REMAINING=3 is a safety margin for
# crashes/hangs, not an expectation of actually needing multiple 48h jobs.
#
# DDP_TIMEOUT_MIN=90 (utils.py's ddp_setup) is the real watchdog-timeout knob
# -- NCCL_TIMEOUT does NOT control it (see distill-lazy-teacher-progress
# memory). The wandb-fork-hang that used to recur ~34-38min in was
# root-caused + fixed in train.py's run() (DataLoader workers forked before
# wandb.init()) -- that fix is global (committed in the python code), so this
# script inherits it automatically. Validated clean at world_size=16; this is
# the first run at world_size=52, unproven at this scale.
#
# Submit with: sbatch distill_train_pretrain_full500_reg52.sbatch
#
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=13
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH -J distill_pretrain_full500_reg52
#SBATCH -o /pscratch/sd/t/twamorka/omnilearned/logs/distill_pretrain_full500_reg52/slurm_%j.out

set -uo pipefail

CHAIN_REMAINING=${CHAIN_REMAINING:-3}
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_pretrain_full500_reg52
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

DATA_PATH=/pscratch/sd/t/twamorka/omnilearned/datasets/

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag distill_pretrain_s_scratch_a05_b05_T4_full500_reg52 \
  --dataset pretrain --mode pretrain --num-classes 210 \
  --path $DATA_PATH \
  --size small \
  --use-pid --use-add --use-event-loss --interaction --local-interaction \
  --feature-drop 0.1 \
  --batch 128 --iterations 1000 --epoch 154 \
  --num-workers 4 \
  --distill \
  --teacher-labels-dir $DATA_PATH \
  --teacher-tag pretrain_l \
  --distill-alpha 0.5 --distill-beta 0.5 --distill-t 4 \
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
