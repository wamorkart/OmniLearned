#!/bin/bash
# Pretrain the large PET model on JetClass alone (mode=pretrain, num-classes=10,
# see pretrain_train_jetclass_l.sh for the full explanation of that split), with
# --local-interaction added, for 2 passes over the ~100M-jet JetClass training
# set at 52 GPUs (13 nodes x 4 GPUs, regular queue) -- 100 GPUs was judged hard
# to get allocated, so scaled down; this no longer fits in a single 48h job.
#
# --iterations 1000 sets iterations PER RANK, not a global count (train.py:587,
# dataloader.py), so samples/epoch = 1000 * batch(8) * n_gpus(52) = 416,000.
# 2 passes = 200M samples -> --epoch 481 (200e6 / 416000, rounded up).
#
# t_step measured at 0.5s/step (500s/1000 iterations, epoch 0 of the running
# 16-GPU pretrain_l_jetclass_4pass_localint interactive job, 2026-08-04) --
# confirms --local-interaction doesn't change per-step time vs. the earlier
# no-local-interaction measurement (498s/epoch). At 52 GPUs: 481 epochs *
# 500s = 240,500s ~= 66.8h (~2.8 days), over gpu_regular's 48h/job cap
# (sacctmgr show qos), so this needs the self-chain below (48h + ~18.8h in a
# second job; CHAIN_REMAINING=2 gives 96h of capacity, ~44% margin).
#
# On a non-zero exit it resubmits itself via sbatch with CHAIN_REMAINING
# decremented; on exit 0 (training finished) it stops. Submit with:
#   sbatch pretrain_train_jetclass_l_2pass_localint.sbatch
#
# CAUTION: a prior interactive-loop script (pretrain_loop_jetclass_l.sh) had
# `set -e` + an unguarded `salloc | tee` pipeline, so `set -e` fired on the
# pipeline's non-zero exit status BEFORE the resubmit logic ever ran -- the
# loop silently died on the very first timeout. This script avoids that by
# `set +e` / `set -e` around the srun call, same fix applied there.
#
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=13
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=48:00:00
#SBATCH -J pretrain_l_jetclass_2pass_localint
#SBATCH -o /pscratch/sd/t/twamorka/omnilearned/logs/pretrain_l_jetclass_2pass_localint/slurm_%j.out

set -uo pipefail

CHAIN_REMAINING=${CHAIN_REMAINING:-2}
LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/pretrain_l_jetclass_2pass_localint
mkdir -p "$LOG_DIR"

export PATH=/global/homes/t/twamorka/omnilearned-clean/env/bin:$PATH
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag pretrain_l_jetclass_2pass_localint \
  --mode pretrain --dataset jetclass --num-classes 10 \
  --path /global/cfs/cdirs/m4567/www/ \
  --size large \
  --use-pid --use-add --num-feat 4 \
  --use-event-loss --interaction --local-interaction \
  --epoch 481 --iterations 1000 --num-workers 4 \
  --lr 1e-05 --optim lion --feature-drop 0.1 --wd 0.1 --batch 8 \
  --wandb --resuming"

echo "[$(date '+%F %T')] job ${SLURM_JOB_ID}, chain remaining (incl. this run) ${CHAIN_REMAINING}"

set +e
srun -l -u bash -c "source export_ddp.sh; $cmd"
EXIT=$?
set -e

echo "[$(date '+%F %T')] job ${SLURM_JOB_ID} exited with code ${EXIT}" | tee -a "$LOG_DIR/summary.log"

if [ "$EXIT" -eq 0 ]; then
    echo "Training completed. Not resubmitting." | tee -a "$LOG_DIR/summary.log"
elif [ "$CHAIN_REMAINING" -gt 1 ]; then
    NEXT=$((CHAIN_REMAINING - 1))
    echo "Non-zero exit, resubmitting (chain remaining after this: ${NEXT})..." \
        | tee -a "$LOG_DIR/summary.log"
    sbatch --export=ALL,CHAIN_REMAINING="$NEXT" "$0"
else
    echo "Chain exhausted (CHAIN_REMAINING reached 0) without a clean exit -- check logs." \
        | tee -a "$LOG_DIR/summary.log"
fi
