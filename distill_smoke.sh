#!/bin/bash
# Smoke test for the from-scratch KD distillation path. Runs 1 epoch x 2
# iterations on a single small dataset (atlas) to confirm:
#   - companion teacher .h5 files load (no missing-companion error),
#   - teacher_logits reach the batch and loss_kd is finite + non-zero,
#   - forward/backward of the small student under --mode pretrain works.
# Distributed over the allocation via srun. export_ddp.sh derives MASTER_ADDR
# (first node) + a per-job MASTER_PORT and sets the SLURM rank vars, so each
# task takes the env:// rendezvous branch of ddp_setup (utils.py:566).
#
# Run from inside a GPU allocation; one srun task per GPU. For 4 GPUs:
#   salloc ... --ntasks=4 --gpus-per-task=1   (or 1 node, 4 tasks)
#   ./distill_smoke.sh 2>&1 | tee distill_smoke.log

set -euo pipefail

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

# HDF5 file locking causes slow/hanging opens over Lustre-backed CFS; already
# worked around this way elsewhere in the repo (camels.sh, quijote.sh).
export HDF5_USE_FILE_LOCKING=FALSE

# Force the single-process path (in case these leaked into the env).
unset MASTER_ADDR RANK WORLD_SIZE WORLD_RANK LOCAL_RANK

TEACHER_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion
DATA=/global/cfs/cdirs/m4567/www/

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/smoke \
  --save-tag distill_smoke \
  --dataset atlas --mode pretrain --num-classes 210 \
  --path $DATA \
  --size small \
  --use-pid --use-add --use-event-loss --interaction \
  --feature-drop 0.1 \
  --batch 64 --iterations 2 --epoch 1 --num-workers 4 \
  --distill \
  --teacher-labels-dir $TEACHER_DIR \
  --teacher-tag pretrain_l \
  --distill-alpha 0.5 --distill-beta 0.5 --distill-t 4 \
  --wandb"

set -x
# Distributed over the allocation. export_ddp.sh sets MASTER_ADDR/PORT and the
# SLURM rank vars so ddp_setup takes the env:// rendezvous branch (utils.py:566).
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "