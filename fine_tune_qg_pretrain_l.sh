#!/bin/bash
# Fine-tune the large pretrained model (pretrain_l, 460M params) on the
# quark/gluon (qg) dataset, 2-class classifier.
#
# Run inside an salloc GPU interactive job:
#   salloc -C gpu -q interactive -t 240 --nodes 1 --ntasks-per-node 4 --gpus-per-node 4 -A m3246
# and then run bash fine_tune_qg_pretrain_l.sh

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
# ============================================================
SAVE_TAG_BASE=fine_tune_pretrain_l
DATASET=qg
DESCRIPT_TAG=int_i300_e10 # int_localint_i300_e10 # nointterms_i300_e10 
# ============================================================

DIR="/pscratch/sd/m/mbenyas"

cmd="omnilearned train \
  -o ${DIR} \
  --save-tag ${SAVE_TAG_BASE}_${DATASET}_${DESCRIPT_TAG} \
  --pretrain-tag pretrain_l \
  --fine-tune \
  --dataset ${DATASET} --mode classifier --num-classes 2 \
  --path /global/cfs/cdirs/m4567/www/ \
  --size large \
  --use-pid \
  --interaction \
  --iterations 300 \
  --batch 32 --epoch 10 --wd 10.0 --lr 1e-6 \
  --num-workers 4 \
  --wandb --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "