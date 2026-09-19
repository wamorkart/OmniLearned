#!/bin/bash
# Evaluate the large model on the top dataset and dump raw logits for
# test + train + val splits. Designed to fit within a 4-hour interactive job:
#   - Sharding splits the dataset across NUM_SHARDS independent jobs.
#   - Chunking flushes every FLUSH_EVERY batches, so a timed-out job can
#     resume from the last fully-written chunk on the next launch.
#
# Usage:
#   NUM_SHARDS=4 SHARD_ID=0 bash evaluate_top_large.sh
#   NUM_SHARDS=4 SHARD_ID=1 bash evaluate_top_large.sh   # etc.
# Or via SLURM array: sbatch --array=0-3 ...    (SHARD_ID defaults to $SLURM_ARRAY_TASK_ID)

module load pytorch

export MASTER_ADDR=$(hostname)

CHECKPOINT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints/
OUTPUT_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits_top/
SAVE_TAG=pretrain_l
DATASET=atlas
SIZE=large

NUM_SHARDS=${NUM_SHARDS:-2}
SHARD_ID=${SHARD_ID:-${SLURM_ARRAY_TASK_ID:-0}}
FLUSH_EVERY=${FLUSH_EVERY:-200}

mkdir -p "$OUTPUT_DIR"

cmd="omnilearned evaluate \
    -i $CHECKPOINT_DIR \
    -o $OUTPUT_DIR \
    --save-tag $SAVE_TAG \
    --dataset $DATASET \
    --path /global/cfs/cdirs/m4567/www/ \
    --size $SIZE \
    --mode classifier \
    --num-classes 210 \
    --use-event-loss
    --interaction \
    --local-interaction \
    --batch 64 \
    --num-workers 4 \
    --num-shards $NUM_SHARDS \
    --shard-id $SHARD_ID \
    --flush-every $FLUSH_EVERY \
    --eval-train-val \
    "

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
