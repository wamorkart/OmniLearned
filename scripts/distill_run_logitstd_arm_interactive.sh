#!/bin/bash
# Run ONE arm of the logit-standardization A/B against an already-running
# interactive salloc allocation, via `srun --jobid=<X> --overlap`. Companion
# to distill_train_pretrain_logitstd_ab.sbatch -- identical `omnilearned
# train` command and --save-tag scheme; use the .sbatch for the batch-queue
# arm so the pair stays comparable.
#
#   JOBID=57834193 STANDARDIZE=0 SEED=1234 \
#       bash scripts/distill_run_logitstd_arm_interactive.sh
#
#   STANDARDIZE=1 -> --distill-standardize   (tag suffix _std)
#   STANDARDIZE=0 -> plain KD                 (tag suffix _nostd)
#   SEED          -> --seed, identical across the pair
#
# Assumes the allocation is 4 nodes x 4 GPU (16 tasks). Launch inside screen:
#   screen -dmS logitstd_ab bash -c '... ; exec bash'
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD

JOBID=${JOBID:?set JOBID to the interactive salloc job id}
STANDARDIZE=${STANDARDIZE:?set STANDARDIZE=0 or 1}
SEED=${SEED:?set SEED to an integer}

if [ "$STANDARDIZE" -eq 1 ]; then
    STD_FLAG="--distill-standardize"; SUFFIX="std"
else
    STD_FLAG=""; SUFFIX="nostd"
fi
TAG="distill_pretrain_s_a05_T4_seed${SEED}_${SUFFIX}"

LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_pretrain_logitstd_ab
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/interactive_${TAG}_$(date +%Y%m%d_%H%M%S).log"

export PATH=/global/homes/t/twamorka/omnilearned-clean/env/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export DDP_TIMEOUT_MIN=90

DATA_PATH=/pscratch/sd/t/twamorka/omnilearned/datasets/

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag ${TAG} \
  --dataset pretrain --mode pretrain --num-classes 210 \
  --path $DATA_PATH \
  --size small \
  --use-pid --use-add --use-event-loss --interaction --local-interaction \
  --feature-drop 0.1 \
  --batch 128 --iterations 1000 --epoch 15 \
  --num-workers 4 \
  --seed ${SEED} \
  --distill \
  --teacher-labels-dir $DATA_PATH \
  --teacher-tag pretrain_l \
  --distill-alpha 0.5 --distill-beta 0.5 --distill-t 4 ${STD_FLAG} \
  --wandb"

echo "[$(date '+%F %T')] arm=${SUFFIX} seed=${SEED} alloc=${JOBID}"
echo "log: $LOG"
echo "cmd: $cmd" | tee "$LOG"

srun --jobid="$JOBID" --overlap --chdir="$REPO" -l -u \
    -N4 --ntasks-per-node=4 \
    bash -c "source export_ddp.sh; $cmd" 2>&1 | tee -a "$LOG"
rc=${PIPESTATUS[0]}
echo "[$(date '+%F %T')] arm=${SUFFIX} exited ${rc}" | tee -a "$LOG"
