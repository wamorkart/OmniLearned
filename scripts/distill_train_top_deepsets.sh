#!/bin/bash
# Distill a fine-tuned top-tagging teacher -> DeepSets/PFN student on top
# tagging. Config-driven: one script for every DeepSets-KD ablation run.
#
#   CONFIG=<name> bash scripts/distill_train_top_deepsets.sh
#
# CONFIG selects a row from the table below (default: a05). Override the
# checkpoint tag with SAVE_TAG=... for independent-seed replicate runs (no
# --seed flag exists in the CLI, so a fresh tag + fresh process = a true
# replicate). Override wandb with WANDB=0/1.
#
#   CONFIG   save-tag                                              size        wd    distill  a/b       teacher  extra
#   a05      distill_top_deepsets_small_scratch_a05_T4_archfix0804 small       0.5   yes      0.5/0.5   l        -                    (reference / confirmed best)
#   a00_b10  distill_top_deepsets_small_scratch_a00_b10_T4         small       0.5   yes      0.0/1.0   l        -                    (pure KD, DistillNet-style)
#   wd005    distill_top_deepsets_small_scratch_a05_wd005_T4       small       0.05  yes      0.5/0.5   l        -                    (weight-decay ablation)
#   ewpool   distill_top_deepsets_small_scratch_a05_ewpool_T4      small       0.5   yes      0.5/0.5   l        --energy-weighted-pool
#   distillnet distill_top_deepsets_distillnet_scratch_a05_T4      distillnet  0.5   yes      0.5/0.5   l        -                    (10,981-param size scan)
#   teachS   distill_top_deepsets_small_scratch_teachS_a05_T4      small       0.5   yes      0.5/0.5   s        -                    (teacher-capacity ablation)
#   ce       train_top_deepsets_small_ce_scratch                   small       0.5   no       -         -        -                    (CE-only no-teacher baseline)
#
# See EXPERIMENTS_deepsets_kd.md for the rationale, history, and results of
# each config (including the a00_b10 mislabeled-PET2 retrain and the
# multi-node --wandb/NCCL fork-hang risk on the wandb-enabled configs).
#
# DeepSets ignores --interaction/--local-interaction (Phi-embed + pooled +
# rho-MLP, no attention), so those flags are never passed.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash scripts/distill_train_top_deepsets.sh
# For a walltime-surviving resubmit loop use distill_loop_top_deepsets.sh.

set -euo pipefail

CONFIG="${CONFIG:-${1:-a05}}"

TEACHER_ROOT=/pscratch/sd/t/twamorka/omnilearned/teacher_logits

# defaults, overridden per-config below
SIZE=small
WD=0.5
DISTILL=1
ALPHA=0.5
BETA=0.5
TEACHER_TAG=fine_tune_top_l
WANDB_DEFAULT=0
EXTRA_FLAGS=""

case "$CONFIG" in
  a05)
    TAG=distill_top_deepsets_small_scratch_a05_T4_archfix0804 ;;
  a00_b10)
    TAG=distill_top_deepsets_small_scratch_a00_b10_T4
    ALPHA=0.0; BETA=1.0; WANDB_DEFAULT=1 ;;
  wd005)
    TAG=distill_top_deepsets_small_scratch_a05_wd005_T4
    WD=0.05 ;;
  ewpool)
    TAG=distill_top_deepsets_small_scratch_a05_ewpool_T4
    WANDB_DEFAULT=1; EXTRA_FLAGS="--energy-weighted-pool" ;;
  distillnet)
    TAG=distill_top_deepsets_distillnet_scratch_a05_T4
    SIZE=distillnet ;;
  teachS)
    TAG=distill_top_deepsets_small_scratch_teachS_a05_T4
    TEACHER_TAG=fine_tune_top_s; WANDB_DEFAULT=1 ;;
  ce)
    TAG=train_top_deepsets_small_ce_scratch
    DISTILL=0; WANDB_DEFAULT=1 ;;
  *)
    echo "unknown CONFIG='$CONFIG' (see table in this script's header)" >&2
    exit 2 ;;
esac

SAVE_TAG="${SAVE_TAG:-$TAG}"
WANDB="${WANDB:-$WANDB_DEFAULT}"

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag $SAVE_TAG \
  --dataset top --mode classifier --num-classes 2 \
  --path /global/cfs/cdirs/m4567/www/ \
  --arch deep-sets --size $SIZE \
  --batch 128 --iterations 1000 --epoch 50 \
  --lr 5e-4 --wd $WD \
  --num-workers 4 \
  $EXTRA_FLAGS"

if [ "$DISTILL" = "1" ]; then
  cmd="$cmd \
  --distill \
  --teacher-labels-dir $TEACHER_ROOT/companion_$TEACHER_TAG \
  --teacher-tag $TEACHER_TAG \
  --distill-alpha $ALPHA --distill-beta $BETA --distill-t 4"
fi

[ "$WANDB" = "1" ] && cmd="$cmd --wandb"
cmd="$cmd --resuming"

echo "CONFIG=$CONFIG  SAVE_TAG=$SAVE_TAG  size=$SIZE  wd=$WD  distill=$DISTILL  wandb=$WANDB"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
