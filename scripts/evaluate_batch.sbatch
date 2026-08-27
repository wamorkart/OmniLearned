#!/bin/bash
# Batch-queue runner for ONE evaluation chunk of ONE dataset/split.
# Submitted by submit_datasets.sh, which sets --nodes/--time/--dependency and
# exports the parameters below. Self-contained: puts the project conda env on
# PATH (no `module load`), so it works regardless of the submitting shell.
#
# Required env (passed via sbatch --export): DATASET DATASET_TYPE NUM_CHUNKS
#   CHUNK_IDX OUTPUT_DIR SAVE_TAG CHECKPOINT_DIR DATA_PATH SIZE EVAL_AMP
#   BATCH NUM_WORKERS
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
set -euo pipefail

: "${DATASET:?}"; : "${DATASET_TYPE:?}"; : "${NUM_CHUNKS:?}"; : "${CHUNK_IDX:?}"
: "${OUTPUT_DIR:?}"; : "${SAVE_TAG:?}"; : "${CHECKPOINT_DIR:?}"; : "${DATA_PATH:?}"
SIZE=${SIZE:-large}
EVAL_AMP=${EVAL_AMP:-fp32}
BATCH=${BATCH:-64}
NUM_WORKERS=${NUM_WORKERS:-8}

export PATH=/global/homes/t/twamorka/omnilearned-clean/env/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Compute nodes can't reach the login hostname; use the first allocation node.
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)

mkdir -p "$OUTPUT_DIR"

echo "=== $(date '+%F %T')  $DATASET/$DATASET_TYPE chunk $CHUNK_IDX/$NUM_CHUNKS "\
     "| nodes=$SLURM_NNODES amp=$EVAL_AMP batch=$BATCH ==="

cmd="omnilearned evaluate \
    -i $CHECKPOINT_DIR \
    -o $OUTPUT_DIR \
    --save-tag $SAVE_TAG \
    --dataset $DATASET \
    --path $DATA_PATH \
    --size $SIZE \
    --interaction --local-interaction --use-pid --use-add --use-event-loss \
    --num-classes 210 \
    --batch $BATCH \
    --num-workers $NUM_WORKERS \
    --dataset-type $DATASET_TYPE \
    --num-chunks $NUM_CHUNKS \
    --chunk-idx $CHUNK_IDX"

srun -l -u bash -c "source export_ddp.sh; EVAL_AMP=$EVAL_AMP $cmd"
