#!/bin/bash
# Reduced-scale A/B for logit standardization in the pretrain-KD stage
# (large PET2 teacher -> small PET2 student, full pretrain mixture).
#
# One arm of the pair. Controlled by two env vars:
#   STANDARDIZE=1  -> adds --distill-standardize   (tag suffix _std)
#   STANDARDIZE=0  -> plain KD                     (tag suffix _nostd)
#   SEED=<int>     -> --seed, identical across the pair so init + data order
#                     match and only the KD loss differs.
#
# Short run: --epoch 15 x 1000 it x 128 batch x 16 gpus ~= 3.1e7 samples/epoch,
# ~15 epochs ~= 1.9h at the measured ~438s/epoch-unit (16 GPU, merged scratch
# data). From scratch, NOT --resuming. Cosine LR decays to ~0 by epoch 15 for
# both arms (identical schedule), so each arm is a complete mini-training.
#
# Compare on wandb: `val loss_kd_ref` (plain KL at T, standardization-independent)
# and `val loss_class`. If the curves separate cleanly the effect is real ->
# promote to a full 154-epoch run + downstream top-tagging fine-tune. If within
# run-to-run noise, add 1-2 more seeded reps per arm before concluding.
#
# Submit the pair with: bash distill_launch_pretrain_logitstd_ab.sh
#
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH -J distill_pretrain_logitstd_ab
#SBATCH -o /pscratch/sd/t/twamorka/omnilearned/logs/distill_pretrain_logitstd_ab/slurm_%j.out

set -uo pipefail

STANDARDIZE=${STANDARDIZE:?set STANDARDIZE=0 or 1}
SEED=${SEED:?set SEED to an integer}

if [ "$STANDARDIZE" -eq 1 ]; then
    STD_FLAG="--distill-standardize"
    SUFFIX="std"
else
    STD_FLAG=""
    SUFFIX="nostd"
fi
TAG="distill_pretrain_s_a05_T4_seed${SEED}_${SUFFIX}"

LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_pretrain_logitstd_ab
mkdir -p "$LOG_DIR"

export PATH=/global/homes/t/twamorka/omnilearned-clean/env/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
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

echo "[$(date '+%F %T')] job ${SLURM_JOB_ID}  tag ${TAG}"
echo "cmd: $cmd"

set +e
srun -l -u bash -c "source export_ddp.sh; $cmd"
EXIT=$?
set -e

echo "[$(date '+%F %T')] job ${SLURM_JOB_ID} (${TAG}) exited ${EXIT}" \
    | tee -a "$LOG_DIR/summary.log"
