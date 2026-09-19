#!/bin/bash
# Run one chunk of the full evaluation. Each chunk processes 1/NUM_CHUNKS of
# the dataset across all DDP ranks, writing per-rank .npz files tagged with
# the chunk index. Run one chunk per interactive Slurm session, then merge
# with concat_logits.py once all chunks are done.
#
# Usage:
#   NUM_CHUNKS=6 bash evaluate_chunk.sh 0    # session 1
#   NUM_CHUNKS=6 bash evaluate_chunk.sh 1    # session 2
#   ...
#   NUM_CHUNKS=6 bash evaluate_chunk.sh 5    # session 6
#
# To see which chunks are done in $OUTPUT_DIR:
#   ls $OUTPUT_DIR/outputs_${SAVE_TAG}_${DATASET}_${DATASET_TYPE}_chunk*_rank*.npz \
#     | sed -E 's/.*_chunk([0-9]+)of[0-9]+_rank.*/\1/' | sort -un

set -euo pipefail

CHUNK_IDX=${1:?usage: evaluate_chunk.sh <chunk_idx>   (env: NUM_CHUNKS, default 6)}
NUM_CHUNKS=${NUM_CHUNKS:-6}

if (( CHUNK_IDX < 0 || CHUNK_IDX >= NUM_CHUNKS )); then
    echo "ERROR: CHUNK_IDX=$CHUNK_IDX out of range for NUM_CHUNKS=$NUM_CHUNKS" >&2
    exit 1
fi

module load pytorch
# When salloc runs us on the login node, $(hostname) gives login40 — the
# compute nodes can't reach that on the rendezvous port, so DDP hangs.
# Use the first node in the allocation instead.
if [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
else
    export MASTER_ADDR=$(hostname)
fi

CHECKPOINT_DIR=${CHECKPOINT_DIR:-/pscratch/sd/t/twamorka/omnilearned/checkpoints/}
OUTPUT_DIR=${OUTPUT_DIR:-/pscratch/sd/t/twamorka/omnilearned/teacher_logits/TEST/}
SAVE_TAG=${SAVE_TAG:-pretrain_l}
DATASET=${DATASET:-atlas}
SIZE=${SIZE:-large}
DATASET_TYPE=${DATASET_TYPE:-test}
NUM_CLASSES=${NUM_CLASSES:-210}
EXTRA_FLAGS=${EXTRA_FLAGS:---local-interaction --use-pid --use-add --use-event-loss}

mkdir -p "$OUTPUT_DIR"

cmd="omnilearned evaluate \
    -i $CHECKPOINT_DIR \
    -o $OUTPUT_DIR \
    --save-tag $SAVE_TAG \
    --dataset $DATASET \
    --path /global/cfs/cdirs/m4567/www/ \
    --size $SIZE \
    --interaction \
    --num-classes $NUM_CLASSES \
    --batch ${BATCH:-64} \
    --num-workers ${NUM_WORKERS:-8} \
    --dataset-type $DATASET_TYPE \
    --num-chunks $NUM_CHUNKS \
    --chunk-idx $CHUNK_IDX \
    $EXTRA_FLAGS"

LOG="$OUTPUT_DIR/chunk${CHUNK_IDX}of${NUM_CHUNKS}.log"
echo "Streaming srun output to $LOG (tail -f to follow)"
set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    " 2>&1 | tee "$LOG"
