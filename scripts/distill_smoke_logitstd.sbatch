#!/bin/bash
# 1-node smoke for the --distill-standardize (logit standardization) + --seed
# additions (commit 73a2f67). Exercises the new KD-loss path, the seed
# plumbing, and the `loss_kd_ref` log on a tiny run. NOT a quality run --
# throwaway tag, no wandb, no checkpoint kept.
#
# Uses the known-good top-tagging KD config (see scripts/configs/train/
# _defaults.sh): large fine-tuned teacher (companion_fine_tune_top_l) ->
# PET2-small student, 2 classes, top data read straight from world-readable
# CFS. That path is a SINGLE train file + single val file + single companion,
# so it stays fast even when $SCRATCH Lustre metadata is contended -- an
# earlier h1/atlas-based version of this smoke (hundreds of per-file opens on
# a loaded /pscratch) stalled in dataloader setup past the 30-min debug wall.
#
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH -J distill_smoke_logitstd
#SBATCH -o /pscratch/sd/t/twamorka/omnilearned/logs/distill_smoke_logitstd/slurm_%j.out

set -uo pipefail
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/distill_smoke_logitstd
mkdir -p "$LOG_DIR"

export PATH=/global/homes/t/twamorka/omnilearned-clean/env/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export DDP_TIMEOUT_MIN=20

DATA_PATH=/global/cfs/cdirs/m4567/www/
TEACHER_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_l

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag distill_smoke_logitstd_throwaway \
  --dataset top --mode classifier --num-classes 2 \
  --path $DATA_PATH \
  --size small \
  --interaction --local-interaction \
  --batch 128 --iterations 10 --epoch 1 \
  --num-workers 4 \
  --seed 42 \
  --distill \
  --teacher-labels-dir $TEACHER_DIR \
  --teacher-tag fine_tune_top_l \
  --distill-alpha 0.5 --distill-beta 0.5 --distill-t 4 --distill-standardize"

echo "cmd: $cmd"
set +e
srun -l -u bash -c "source export_ddp.sh; $cmd"
EXIT=$?
set -e
echo "[$(date '+%F %T')] smoke job ${SLURM_JOB_ID} exited ${EXIT}" | tee -a "$LOG_DIR/summary.log"
