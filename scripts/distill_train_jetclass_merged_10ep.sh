#!/bin/bash
# 10-epoch validation: pretrain-KD training on jetclass (fully merged --
# teacher_logits are now inside the source h5 files, see
# merge_teacher_logits.py) via the merged-file dataloader.py code path.
#
# Follow-up to distill_train_jetclass_merged_smoke.sh -- that run (job
# 56410376, ~11:29 on 2026-08-06) predates the merged-file fix to
# dataloader.py (saved 11:32:49 same day) by a few minutes and silently fell
# back to the old lazy-companion-file path, so it never actually validated
# the new code. This run uses the current dataloader.py (uncommitted, has the
# "teacher_logits" in f merged-file check) and a fresh tag so it can't pick
# up the old smoke checkpoint via --resuming (--resuming is not passed here
# at all -- fresh run).
#
# Confirm in the log:
#   "Teacher logits: 1000 source files (1000 merged in-file, 0 via companion .h5)"
# (vs the smoke run's "1000 companion .h5 files") before trusting the loss
# curves below as validating the merged path.
#
# Run inside a small GPU allocation, e.g.:
#   salloc -C gpu -q interactive -t 60 --nodes 1 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash distill_train_jetclass_merged_10ep.sh

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export HDF5_USE_FILE_LOCKING=FALSE
export MASTER_ADDR=$(hostname)
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

TEACHER_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/smoke \
  --save-tag distill_jetclass_merged_10ep \
  --dataset jetclass --mode pretrain --num-classes 210 \
  --path /pscratch/sd/t/twamorka/omnilearned/datasets \
  --size small \
  --use-pid --use-add --use-event-loss --interaction --local-interaction \
  --feature-drop 0.1 \
  --batch 128 --iterations 20 --epoch 10 --num-workers 4 \
  --distill \
  --teacher-labels-dir $TEACHER_DIR \
  --teacher-tag pretrain_l \
  --distill-alpha 0.5 --distill-beta 0.5 --distill-t 4 --wandb"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
