#!/bin/bash
# Fine-tune a pretrained model as the idealized-CWoLa anomaly-detection
# classifier for LHCO (lhco_ad: bkg.h5 pid=0 vs data.h5 pid=1, 2 classes).
#
# --use-add --num-add 2 is required: convert_lhco.py appends a 2-column
# one-hot (which jet each particle came from) as the last 2 columns of
# `data`, and this is how OmniLearned splices those back out before the
# kinematic features reach the model.
#
# No --interaction/--local-interaction on purpose: those compute per-particle
# nearest neighbors from delta_eta/delta_phi, which are relative to each
# particle's OWN jet axis. Since both jets' particles are merged into one
# point cloud here, that neighbor search would mix particles from unrelated
# jets that happen to land at similar relative coordinates. Revisit this if
# the interaction terms turn out to matter for performance.
#
# Run inside an salloc GPU interactive job:
#   salloc -C gpu -q interactive -t 240 --nodes 1 --ntasks-per-node 4 --gpus-per-node 4 -A m3246
# and then run bash fine_tune_lhco_ad.sh

module load conda
conda activate ol_distill
# module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ============================================================
#  EDIT THESE PER RUN -- everything else derives from them
#  NSIG must match a folder convert_lhco.py already produced
#  (LHCO/nsig_<NSIG>/lhco_ad/...) -- "all" means --nsig was left unset.
# ============================================================
SAVE_TAG_BASE=fine_tune_pretrain_s
DATASET=lhco_ad
PRETRAIN_TAG=pretrain_s
SIZE=small
NSIG=500
DESCRIPT_TAG=test2
# ============================================================

SAVE_TAG="${SAVE_TAG_BASE}_${DATASET}_nsig${NSIG}_${DESCRIPT_TAG}"
DIR="/pscratch/sd/m/mbenyas/LHCO"
LHCO_PATH="/global/cfs/cdirs/m3246/mbenyas/OmniLearned_distillation/LHCO/nsig_${NSIG}"

cmd="omnilearned train \
  -o ${DIR} \
  --save-tag ${SAVE_TAG} \
  --pretrain-tag ${PRETRAIN_TAG} \
  --fine-tune \
  --dataset ${DATASET} --mode classifier --num-classes 2 \
  --path ${LHCO_PATH} \
  --size ${SIZE} \
  --use-add --num-add 2 \
  --iterations 1000 \
  --batch 32 --epoch 10 --wd 0.01 --lr 1e-4 --lr-factor 10.0 \
  --num-workers 4 \
  --wandb --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
