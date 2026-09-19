#!/bin/bash
# OmniFold iterative unfolding: Pythia21 (MC, reco+gen) vs Herwig (stand-in
# "data", reco only) closure benchmark -- the same dataset OmniLearn's own
# omnifold.py/train_omnifold.py target. See src/omnilearned/omnifold.py for
# the algorithm port. Data preprocessed by preprocess_omnifold.py into
# /pscratch/sd/t/twamorka/unfolding/{train,test}_{pythia,herwig}.h5.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 1 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash train_omnifold_pythia_herwig.sh

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

cmd="omnilearned unfold \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag omnifold_pythia_herwig \
  --path /pscratch/sd/t/twamorka/unfolding \
  --num-feat 13 \
  --size small \
  --interaction \
  --local-interaction \
  --num-iter 5 \
  --patience 3 \
  --batch 512 --epoch 30 --warmup-epoch 1 \
  --lr 3e-5 --lr-factor 5 --wd 0.1 \
  --num-workers 0 \
  --wandb"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
