#!/bin/bash
# Smoke test for the minimal MLP student architecture (--arch mlp in
# network.py). Runs 1 epoch x 2 iterations of plain CE classification on
# top-tagging to confirm:
#   - MLPStudent forward/backward runs end-to-end under train_step/val_step,
#   - checkpoint save/restore works with MLPStudent's .body/.classifier
#     interface (generator=None),
#   - loss is finite and decreases across the 2 iterations.
# No KD, no wandb -- just validating the architecture trains.
#
# Run from inside a GPU allocation, single task (no srun needed):
#   salloc -C gpu -q interactive -t 30 --nodes 1 --ntasks-per-node 1 \
#          --gpus-per-node 1 -A m3246 bash mlp_smoke.sh

set -euo pipefail

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export HDF5_USE_FILE_LOCKING=FALSE

# Force the single-process path.
unset MASTER_ADDR RANK WORLD_SIZE WORLD_RANK LOCAL_RANK

DATA=/global/cfs/cdirs/m4567/www/

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/smoke \
  --save-tag mlp_smoke \
  --dataset top --mode classifier --num-classes 2 \
  --path $DATA \
  --arch mlp --size small \
  --batch 64 --iterations 2 --epoch 1 --num-workers 4"

set -x
$cmd
