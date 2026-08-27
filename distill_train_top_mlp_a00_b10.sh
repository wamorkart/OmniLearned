#!/bin/bash
# Pure-KD variant of distill_train_top_mlp.sh: alpha=0, beta=1 instead of
# alpha=0.5/beta=0.5 -- see EXPERIMENTS_deepsets_kd.md (a00_b10 config) for
# the rationale (matches DistillNet's pure teacher-transfer loss and this
# project's own T-sweep winner for the PET2 student).
#
# Also serves as an A/B check on the MLP student's near-random eval result
# (AUC 0.626 at a05/b05): if pure KD trains normally, the a05/b05 loss
# weighting (or a05/b05-specific optimization dynamics) was the problem, not
# an architecture-level inability to learn at all.
#
# Teacher/architecture/hyperparams otherwise identical to
# distill_train_top_mlp.sh (MLPStudent small, same teacher logits). New tag
# so --resuming can't pick up the a05_b05 checkpoint.
#
# MLPStudent ignores --interaction/--local-interaction (no attention/interaction
# matrix in this architecture), so those flags are omitted here.
#
# NOTE: --wandb intentionally OMITTED -- see distill_train_top_mlp.sh's NOTE
# (2026-08-04) for the root-caused NCCL hang this avoids.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash distill_train_top_mlp_a00_b10.sh
#
# For a long-running loop, use distill_loop_top_mlp_a00_b10.sh instead.

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

TEACHER_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_l

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag distill_top_mlp_small_scratch_a00_b10_T4 \
  --dataset top --mode classifier --num-classes 2 \
  --path /global/cfs/cdirs/m4567/www/ \
  --arch mlp --size small \
  --batch 128 --iterations 1000 --epoch 50 \
  --lr 5e-4 --wd 0.5 \
  --num-workers 4 \
  --distill \
  --teacher-labels-dir $TEACHER_DIR \
  --teacher-tag fine_tune_top_l \
  --distill-alpha 0.0 --distill-beta 1.0 --distill-t 4 \
  --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
