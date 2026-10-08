#!/bin/bash
# Evaluate the PET2-small KD student distilled from the v7 atlas_flav ftag
# teacher (config scripts/configs/train/atlas_flav_small_a05_T4.sh, save-tag
# distill_atlas_flav_small_a05_T4) on the atlas_flav TEST split. Mirrors
# eval_atlas_flav_ftag_m_localint_v7_interactive.sh (same test chunk 0/6, so
# student and teacher are scored on identical jets) with --size small.
# best_model = val loss 0.4305 at epoch 15 (loss rose afterwards).
#
# Score with (label 2 = b, 0 = light -- NOT GN2 convention):
#   python tools/metrics/compute_metrics_flav.py \
#       --indir /pscratch/sd/t/twamorka/omnilearned/eval/distill_atlas_flav_small_a05_T4/ \
#       --tag distill_atlas_flav_small_a05_T4
#
# Drive with scripts/eval_loop_distill_atlas_flav_small_a05_T4.sh (1 node x 4 GPUs).

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHECKPOINT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints/
EVAL_TAG=${EVAL_TAG:-distill_atlas_flav_small_a05_T4}
OUTPUT_DIR=${OUTPUT_DIR:-/pscratch/sd/t/twamorka/omnilearned/eval/${EVAL_TAG}/}
mkdir -p "$OUTPUT_DIR"

NUM_CHUNKS=${NUM_CHUNKS:-6}
CHUNK_IDX=${CHUNK_IDX:-0}

cmd="omnilearned evaluate \
  -i $CHECKPOINT_DIR \
  -o $OUTPUT_DIR \
  --save-tag $EVAL_TAG \
  --dataset atlas_flav --mode classifier \
  --path /pscratch/sd/t/twamorka/omnilearned/datasets/ \
  --size small \
  --num-feat 4 \
  --use-add --num-add 17 --num-classes 4 \
  --conditional --num-cond 4 --interaction --local-interaction \
  --batch 512 --num-workers 8 \
  --dataset-type test \
  --num-chunks $NUM_CHUNKS --chunk-idx $CHUNK_IDX"

echo "[$(date '+%F %T')] chunk ${CHUNK_IDX}/${NUM_CHUNKS}"
echo "$cmd"

set -x
srun -l -u bash -c "
  source export_ddp.sh
  EVAL_AMP=bf16 $cmd
  "
