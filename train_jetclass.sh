#!/bin/bash
#SBATCH -A marlowe-m000255
#SBATCH -p preempt   # NOT preempt for a multi-day run -- check `sinfo` / docs for a non-preemptible partition
#SBATCH -N 4                            # TODO: confirm with Tanvi/Vinicius -- GPU count determines dataset-pass count, see note below
#SBATCH -G 16                           # 4 nodes x 4 GPUs/node = 16 GPUs total
#SBATCH --ntasks-per-node=4              # must match GPUs-per-node
#SBATCH --time=00:30:00                  # TODO: set to a realistic wall-clock estimate for however many epochs you settle on
#SBATCH -o pretrain_l_jetclass_%j.out

# --- GPU count / epoch math ---
# Each of --iterations 1000 steps processes a synced global batch of
# (--batch * num_GPUs) samples. So:
#   samples/epoch   = iterations * batch * num_GPUs
#   total samples   = samples/epoch * epoch
#   dataset passes  = total samples / 100,000,000 (jetclass jet count)
#
# At num_GPUs=16, batch=8, iterations=1000:
#   samples/epoch = 1000 * 8 * 16 = 128,000
#   total samples = 128,000 samples/epoch * 500 epochs= 64,000,000 samples = 0.64 passes over jetclass
#
# Change--epoch and/or -N/-G to hit target amount of dataset passes?

source /projects/m000255/miniconda/etc/profile.d/conda.sh
conda activate /projects/m000255/miniconda/envs/ol_distill/

# for DDP
export MASTER_ADDR=$(hostname)

# command to pretrain large model on jetclass

cmd="omnilearned train \
    -o /projects/m000255/mbenyas/output/ \
    --save-tag pretrain_l \
    --mode pretrain \
    --dataset jetclass \
    --path /scratch/m000255/twamorka/ \
    --num-classes 10 \
    --epoch 500 \
    --iterations 1000 \
    --num-workers 32 \
    --lr 1e-05 \
    --optim lion \
    --feature-drop 0.1 \
    --wd 0.1 \
    --batch 8 \
    --size large \
    --use-pid \
    --use-add \
    --use-event-loss \
    --interaction \
    --wandb \
    --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "