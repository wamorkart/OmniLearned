#!/bin/bash
# Evaluate a plain DeepSets distillnet+ReLU atlas_flav student (KD or CE
# control, configs scripts/configs/train/atlas_flav_deepsets_distillnet_relu_ftag{,_ce}.sh)
# on the atlas_flav TEST split. DeepSets counterpart of
# eval_distill_atlas_flav_small_a05_T4_interactive.sh; arch flags must match
# training. Trained in --mode ftag, evaluated in --mode classifier (the
# per-track origin head is training-only).
#
#   EVAL_TAG=<save_tag> [OUTPUT_DIR=...] [NUM_CHUNKS=1] [CHUNK_IDX=0] \
#   [EXTRA_EVAL_FLAGS="--num-interaction-layers 1 --interaction-k 64"] \
#     salloc ... bash scripts/eval_atlas_flav_deepsets_interactive.sh
#
# Score with (label 2 = b, 0 = light -- NOT GN2 convention):
#   python tools/metrics/compute_metrics_flav.py --indir <OUTPUT_DIR> --tag <EVAL_TAG>

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHECKPOINT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints/
EVAL_TAG=${EVAL_TAG:?EVAL_TAG required}
OUTPUT_DIR=${OUTPUT_DIR:-/pscratch/sd/t/twamorka/omnilearned/eval/${EVAL_TAG}_full/}
mkdir -p "$OUTPUT_DIR"

NUM_CHUNKS=${NUM_CHUNKS:-1}
CHUNK_IDX=${CHUNK_IDX:-0}
EXTRA_EVAL_FLAGS=${EXTRA_EVAL_FLAGS:-}

cmd="omnilearned evaluate \
  -i $CHECKPOINT_DIR \
  -o $OUTPUT_DIR \
  --save-tag $EVAL_TAG \
  --dataset atlas_flav --mode classifier \
  --path /pscratch/sd/t/twamorka/omnilearned/datasets/ \
  --arch deep-sets --size distillnet --act-layer relu \
  --num-feat 4 \
  --use-add --num-add 17 --num-classes 4 \
  --conditional --num-cond 4 \
  --batch 512 --num-workers 8 \
  --dataset-type test \
  --num-chunks $NUM_CHUNKS --chunk-idx $CHUNK_IDX $EXTRA_EVAL_FLAGS"

echo "[$(date '+%F %T')] chunk ${CHUNK_IDX}/${NUM_CHUNKS}"
echo "$cmd"

set -x
srun -l -u bash -c "
  source export_ddp.sh
  EVAL_AMP=bf16 $cmd
  "
