#!/bin/bash
# Pretrain-KD student run at 52 GPUs (13 nodes x 4, regular queue), PURE KD
# variant (distill-alpha=0.0, distill-beta=1.0 -- no hard-label task loss,
# only the KD/KL term). v2 of distill_train_pretrain_full500_reg52_purekd.sbatch.
#
# Fresh save-tag/job-name/log-dir, deliberately -- the original tag's 3
# crashed sessions (jobs 56728192, 56867955, 56936741) all ran under the
# omnilearned-clean env while it was still an editable install of a stale,
# unrelated clone (distill_dev/OmniLearned) that had neither the
# wandb-forked-worker fix nor the per-epoch last_model_ checkpoint. That env
# now points at this repo's `student` branch (commit 85b8c64), which has
# both, but its checkpoint lineage was written by the buggy code, so starting
# a new tag avoids resuming training state produced under it. Per
# [[feedback-check-before-editing-live-scripts]], a new config/lineage gets
# its own script file.
#
# Same node/GPU geometry, epoch/iteration/batch math, and safety knobs as the
# original (see that file's header for the full throughput/timeout
# rationale): --epoch 154 x --iterations 1000 x --batch 128 x 52 gpus ~=
# 1.025e9 samples, ~1 full pass over the ~1.06B-sample pretrain mixture.
# --lr left at the default 5e-5, not scaled for the 52-GPU global batch.
#
# Submit with: sbatch distill_train_pretrain_full500_reg52_purekd_v2.sbatch
#
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=13
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH -J distill_pretrain_full500_reg52_purekd_v2
#SBATCH -o /pscratch/sd/t/twamorka/omnilearned/logs/distill_pretrain_full500_reg52_purekd_v2/slurm_%j.out

set -uo pipefail

CHAIN_REMAINING=${CHAIN_REMAINING:-3}
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_pretrain_full500_reg52_purekd_v2
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
  --save-tag distill_pretrain_s_scratch_a00_b10_T4_full500_reg52_v2 \
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
  --distill-alpha 0.0 --distill-beta 1.0 --distill-t 4 \
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
