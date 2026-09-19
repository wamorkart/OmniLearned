#!/bin/bash
# Smoke test: pretrain-KD training scoped to just jetclass (the one dataset
# that's fully merged so far -- see merge_teacher_logits.py), reading
# teacher_logits straight out of the merged h5 files on $SCRATCH instead of
# via a separate companion .h5. Purpose is narrowly to confirm, before
# scaling back up to the full 7-dataset --dataset pretrain mixture:
#   - no missing-companion errors (merged files need none),
#   - the dataloader.py merged-file code path actually fires (HEPDataset
#     reads "teacher_logits" directly off the source handle),
#   - loss_kd is finite/non-zero and the printed teacher_logits sample
#     (train.py's `if batch_idx == 0: print(...)`, added for this check)
#     looks like real per-class logits, not zeros/NaN/garbage.
# Same --mode pretrain --num-classes 210 as the real target run
# (distill_train_pretrain_relaunch.sh) so this is a faithful single-dataset
# slice of it, not a different training setup.
#
# Run inside a small GPU allocation, e.g.:
#   salloc -C gpu -q interactive -t 60 --nodes 1 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash distill_train_jetclass_merged_smoke.sh

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export HDF5_USE_FILE_LOCKING=FALSE
export MASTER_ADDR=$(hostname)
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --teacher-labels-dir/--teacher-tag are still required by --distill's own
# CLI validation, but for jetclass (fully merged) load_data never actually
# looks a companion up -- every file's teacher_file_paths entry resolves to
# None since dataloader.py now finds "teacher_logits" already inside the
# source h5. Kept pointed at the real companion dir so nothing breaks if a
# jetclass file were ever missing its merge.
TEACHER_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/smoke \
  --save-tag distill_jetclass_merged_smoke \
  --dataset jetclass --mode pretrain --num-classes 210 \
  --path /pscratch/sd/t/twamorka/omnilearned/datasets \
  --size small \
  --use-pid --use-add --use-event-loss --interaction --local-interaction \
  --feature-drop 0.1 \
  --batch 128 --iterations 20 --epoch 2 --num-workers 4 \
  --distill \
  --teacher-labels-dir $TEACHER_DIR \
  --teacher-tag pretrain_l \
  --distill-alpha 0.5 --distill-beta 0.5 --distill-t 4"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
