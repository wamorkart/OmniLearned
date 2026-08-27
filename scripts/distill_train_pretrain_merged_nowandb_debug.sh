#!/bin/bash
# A/B test (2026-08-08): does the recurring rank-0 NCCL BROADCAST hang
# (SeqNum=7929-ish, ~35min wall-clock into every wandb-enabled session,
# see distill-lazy-teacher-progress memory) still occur with --wandb removed?
# Flight-recorder dumps showed rank 0 (the only rank running a live wandb
# session; all others use mode="disabled") never even enqueues the stalling
# collective -- it's a CPU-side Python stall, and wandb's stdout-redirect
# wrapper is the only rank-0-only code path in the whole training loop.
#
# Fresh debug tag (NOT the real distill_pretrain_s_scratch_a05_b05_T4_merged
# checkpoint) -- this run's only purpose is to observe whether the process
# survives past ~40min wall-clock without hanging; training quality is
# irrelevant, so it starts from scratch rather than resuming.
#load libs
module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export HDF5_USE_FILE_LOCKING=FALSE
export MASTER_ADDR=$(hostname)
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

export TORCH_NCCL_DUMP_ON_TIMEOUT=1
export TORCH_FR_BUFFER_SIZE=2000
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
DUMP_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/nccl_trace
mkdir -p "$DUMP_DIR"
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$DUMP_DIR/dump_${SLURM_JOB_ID}_rank_"

# Keep the short debug timeout -- if this run hangs too, we want to see it
# fail quickly, not wait 90min.
export DDP_TIMEOUT_MIN=15

DATA_PATH=/pscratch/sd/t/twamorka/omnilearned/datasets/

# --epoch 10 at ~450-1200s/epoch-unit measured range comfortably spans the
# ~35-38min mark where every --wandb run so far has hung, with margin.
# No --wandb, no --resuming (fresh debug tag, nothing to resume).
cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag distill_pretrain_nowandb_debug \
  --dataset pretrain --mode pretrain --num-classes 210 \
  --path $DATA_PATH \
  --size small \
  --use-pid --use-add --use-event-loss --interaction --local-interaction \
  --feature-drop 0.1 \
  --batch 128 --iterations 1000 --epoch 10 \
  --num-workers 4 \
  --distill \
  --teacher-labels-dir $DATA_PATH \
  --teacher-tag pretrain_l \
  --distill-alpha 0.5 --distill-beta 0.5 --distill-t 4"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
