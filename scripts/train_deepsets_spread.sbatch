#!/bin/bash
# Generic regular-queue sbatch for the DeepSets-small "spread" reps (2 extra
# independent-seed runs each for the CE-only baseline and the small-teacher
# KD run -- see fpga-deepsets-distillation-progress memory). Moved off the
# interactive-slot queue that used to drive these reps, per user request, so
# all reps run in parallel via -q regular instead of serialized through a
# single interactive slot.
#
# Parameterized by two env vars passed via --export:
#   SAVE_TAG : the checkpoint tag for this rep, e.g.
#              train_top_deepsets_small_ce_scratch_r2 (CE) or
#              distill_top_deepsets_small_scratch_teachS_a05_T4_r3 (teachS)
#   CONFIG   : "ce" or "teachs" -- selects the CE-only vs small-teacher-KD
#              omnilearned train command. Mirrors the "ce" / "teachS" configs
#              of distill_train_top_deepsets.sh, but inlines the command
#              rather than calling that script: its `module load conda`
#              env-activation doesn't reliably work under non-interactive
#              sbatch shells -- PATH-prepend is the proven sbatch pattern in
#              this repo, see train_atlas_flav_ftag_m.sbatch).
#
# `--resuming` means an already-partially-trained tag (e.g. r2, which had
# reached epoch 27/50 under the old interactive queue manager before it was
# stopped) picks up from its last_model_<tag>.pt checkpoint, not from
# scratch -- no progress lost by moving queues.
#
# Submit with:
#   sbatch --export=ALL,SAVE_TAG=train_top_deepsets_small_ce_scratch_r2,CONFIG=ce train_deepsets_spread.sbatch
#   sbatch --export=ALL,SAVE_TAG=train_top_deepsets_small_ce_scratch_r3,CONFIG=ce train_deepsets_spread.sbatch
#   sbatch --export=ALL,SAVE_TAG=distill_top_deepsets_small_scratch_teachS_a05_T4_r2,CONFIG=teachs train_deepsets_spread.sbatch
#   sbatch --export=ALL,SAVE_TAG=distill_top_deepsets_small_scratch_teachS_a05_T4_r3,CONFIG=teachs train_deepsets_spread.sbatch
#
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH -J deepsets_spread
#SBATCH -o /pscratch/sd/t/twamorka/omnilearned/logs/deepsets_spread_reg/slurm_%j_%x.out

set -uo pipefail

: "${SAVE_TAG:?SAVE_TAG must be set via --export}"
: "${CONFIG:?CONFIG must be set via --export (ce or teachs)}"
CHAIN_REMAINING=${CHAIN_REMAINING:-3}
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/deepsets_spread_reg
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

if [ "$CONFIG" = "ce" ]; then
    cmd="omnilearned train \
      -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
      --save-tag $SAVE_TAG \
      --dataset top --mode classifier --num-classes 2 \
      --path /global/cfs/cdirs/m4567/www/ \
      --arch deep-sets --size small \
      --wandb \
      --batch 128 --iterations 1000 --epoch 50 \
      --lr 5e-4 --wd 0.5 \
      --num-workers 4 \
      --resuming"
elif [ "$CONFIG" = "teachs" ]; then
    TEACHER_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_s
    cmd="omnilearned train \
      -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
      --save-tag $SAVE_TAG \
      --dataset top --mode classifier --num-classes 2 \
      --path /global/cfs/cdirs/m4567/www/ \
      --arch deep-sets --size small \
      --wandb \
      --batch 128 --iterations 1000 --epoch 50 \
      --lr 5e-4 --wd 0.5 \
      --num-workers 4 \
      --distill \
      --teacher-labels-dir $TEACHER_DIR \
      --teacher-tag fine_tune_top_s \
      --distill-alpha 0.5 --distill-beta 0.5 --distill-t 4 \
      --resuming"
else
    echo "Unknown CONFIG=$CONFIG (expected ce or teachs)" >&2
    exit 1
fi

echo "[$(date '+%F %T')] job ${SLURM_JOB_ID}, tag=${SAVE_TAG}, config=${CONFIG}, chain remaining ${CHAIN_REMAINING}"

set +e
srun -l -u bash -c "source export_ddp.sh; $cmd"
EXIT=$?
set -e

echo "[$(date '+%F %T')] job ${SLURM_JOB_ID} (${SAVE_TAG}) exited with code ${EXIT}" | tee -a "$LOG_DIR/summary.log"

if [ "$EXIT" -eq 0 ]; then
    echo "Training completed. Not resubmitting." | tee -a "$LOG_DIR/summary.log"
elif [ "$CHAIN_REMAINING" -gt 1 ]; then
    NEXT=$((CHAIN_REMAINING - 1))
    echo "Non-zero exit, resubmitting (chain remaining after this: ${NEXT})..." \
        | tee -a "$LOG_DIR/summary.log"
    sbatch --export=ALL,CHAIN_REMAINING="$NEXT",SAVE_TAG="$SAVE_TAG",CONFIG="$CONFIG" "$0"
else
    echo "Chain exhausted (CHAIN_REMAINING reached 0) without a clean exit -- check logs." \
        | tee -a "$LOG_DIR/summary.log"
fi
