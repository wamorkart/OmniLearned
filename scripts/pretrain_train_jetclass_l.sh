#!/bin/bash
# Pretrain the large PET model (mode=pretrain: joint generative + classification
# head) on the JetClass dataset alone (NOT the full multi-dataset "pretrain"
# mixture). --dataset jetclass shifts on-disk labels down to the LOCAL 10-class
# range (dataloader.py label_shift["jetclass"]=2, y = raw_label - 2), unlike
# --dataset pretrain (the 7-dataset mixture) which uses the raw un-shifted
# 210-class label space. So the classifier head here must be --num-classes 10,
# not 210 -- 210 would only be correct if training on the full mixture.
#
# Run inside an salloc GPU interactive job (16 nodes x 4 GPUs), e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 16 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash pretrain_train_jetclass_l.sh
#
# NOTE: a prior 16-node, mode=pretrain run (full mixture, num-classes=210) hit
# repeated NCCL collective timeouts (see distill-lazy-teacher-progress memory,
# Jul 2-3 attempt), which is why --num-workers is kept conservative below.
# --local-interaction is enabled below (tag pretrain_l_jetclass_4pass_localint)
# to test it as a mitigation (reduces per-step comm) -- this changes the model
# architecture vs. the earlier pretrain_l_jetclass_4pass checkpoint, so this is
# a fresh run, not a resume of that checkpoint.

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag pretrain_l_jetclass_4pass_localint \
  --mode pretrain --dataset jetclass --num-classes 10 \
  --path /global/cfs/cdirs/m4567/www/ \
  --size large \
  --use-pid --use-add --num-feat 4 \
  --use-event-loss --interaction --local-interaction \
  --epoch 3125 --iterations 1000 --num-workers 4 \
  --lr 1e-05 --optim lion --feature-drop 0.1 --wd 0.1 --batch 8 \
  --wandb --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
