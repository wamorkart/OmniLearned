#!/bin/bash
# Drive the full chunked evaluation from a login node: salloc + run + release
# allocation, one chunk at a time. Each chunk gets a fresh 4h interactive
# allocation, so a preemption or wall-time hit only burns the current chunk.
#
# Already-finished chunks (16/16 rank files present) are skipped, so
# resuming after a failure is just `bash run_all_chunks.sh` again.
#
# Run from the project root on a NERSC login node:
#   bash run_all_chunks.sh
# or to rerun a specific subset:
#   RUN_CHUNKS="2 5" bash run_all_chunks.sh
#
# Caveat: each salloc may sit in the interactive queue for several minutes
# before launching. Total wall time = sum(queue+run) for all 6 chunks.

set -euo pipefail
trap 'ec=$?; echo "ERROR: $BASH_SOURCE line $LINENO exited with $ec: $BASH_COMMAND" >&2' ERR

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT"

# Count files matching a glob, robust to no-match (returns "0").
# Plain ls of a non-matching glob with `set -o pipefail` causes the pipeline
# to exit non-zero, which trips `set -e`.
count_glob() {
    local pat=$1
    local n
    n=$(ls $pat 2>/dev/null | wc -l) || n=0
    echo "$n"
}

# -- Tunables -----------------------------------------------------------------
NUM_CHUNKS=${NUM_CHUNKS:-6}
ACCOUNT=${ACCOUNT:-m3246}
TIME=${TIME:-240}                  # minutes per allocation (interactive cap = 240)
NODES=${NODES:-4}
TASKS_PER_NODE=${TASKS_PER_NODE:-4}
GPUS_PER_NODE=${GPUS_PER_NODE:-4}

# Must match evaluate_chunk.sh — used here only to check which chunks are done.
SAVE_TAG=${SAVE_TAG:-pretrain_l}
DATASET=${DATASET:-atlas}
DATASET_TYPE=${DATASET_TYPE:-train}
OUTPUT_DIR=${OUTPUT_DIR:-/pscratch/sd/t/twamorka/omnilearned/teacher_logits/TEST/}
EXPECTED_RANKS=$(( NODES * TASKS_PER_NODE ))

# -- One-shot self-heal -------------------------------------------------------
STALE_INDEX="$SCRATCH/omnilearned_cache/${DATASET}/${DATASET_TYPE}/file_index.npy"
if [ -f "$STALE_INDEX" ]; then
    sz=$(stat -c%s "$STALE_INDEX")
    if [ "$sz" -lt 1000 ]; then
        echo "Removing tiny ($sz-byte) stale scratch index: $STALE_INDEX"
        rm -f "$STALE_INDEX"
    fi
fi

mkdir -p "$OUTPUT_DIR"

# -- Helpers ------------------------------------------------------------------
chunk_done() {
    local K=$1
    local pat="$OUTPUT_DIR/outputs_${SAVE_TAG}_${DATASET}_${DATASET_TYPE}_chunk${K}of${NUM_CHUNKS}_rank*.npz"
    local n
    n=$(count_glob "$pat")
    [ "$n" -eq "$EXPECTED_RANKS" ]
}

list_remaining() {
    local out=()
    if [ -n "${RUN_CHUNKS:-}" ]; then
        out=( $RUN_CHUNKS )
    else
        for ((K=0; K<NUM_CHUNKS; K++)); do
            if chunk_done "$K"; then
                continue
            fi
            out+=("$K")
        done
    fi
    echo "${out[@]}"
}

# -- Status before running ----------------------------------------------------
# When RUN_CHUNKS is set, only show those (so a smoke run with NUM_CHUNKS=1000
# doesn't print 1000 status lines).
if [ -n "${RUN_CHUNKS:-}" ]; then
    STATUS_CHUNKS=( $RUN_CHUNKS )
else
    STATUS_CHUNKS=()
    for ((K=0; K<NUM_CHUNKS; K++)); do STATUS_CHUNKS+=("$K"); done
fi
echo "=== Pre-run status ==="
for K in "${STATUS_CHUNKS[@]}"; do
    if chunk_done "$K"; then
        echo "  chunk $K: DONE  (skipped)"
    else
        n=$(count_glob "$OUTPUT_DIR/outputs_${SAVE_TAG}_${DATASET}_${DATASET_TYPE}_chunk${K}of${NUM_CHUNKS}_rank*.npz")
        echo "  chunk $K: pending  ($n/$EXPECTED_RANKS rank files)"
    fi
done

REMAINING=( $(list_remaining) )
if [ "${#REMAINING[@]}" -eq 0 ]; then
    echo "Nothing to do — all chunks complete."
else
    echo "Will launch chunks: ${REMAINING[*]}"
fi

# -- Main loop ----------------------------------------------------------------
for K in "${REMAINING[@]}"; do
    echo ""
    echo "============================================================"
    echo "=== Chunk $K/$NUM_CHUNKS   start $(date '+%F %T')"
    echo "============================================================"

    # salloc <opts> <command> allocates, runs the command on the submit
    # host, and releases the allocation when the command exits. The command
    # in turn calls srun via evaluate_chunk.sh to launch on compute nodes.
    salloc \
        -C gpu \
        -q interactive \
        -t "$TIME" \
        --nodes "$NODES" \
        --ntasks-per-node="$TASKS_PER_NODE" \
        --gpus-per-node="$GPUS_PER_NODE" \
        -A "$ACCOUNT" \
        bash -c "cd $PROJECT && NUM_CHUNKS=$NUM_CHUNKS DATASET_TYPE=$DATASET_TYPE OUTPUT_DIR=$OUTPUT_DIR SAVE_TAG=$SAVE_TAG DATASET=$DATASET bash evaluate_chunk.sh $K" \
        || echo "WARNING: chunk $K exited non-zero (salloc rc=$?). Continuing to next chunk."

    echo ""
    echo "=== Chunk $K   finished $(date '+%F %T')"

    if chunk_done "$K"; then
        echo "  -> all $EXPECTED_RANKS rank files written"
    else
        n=$(count_glob "$OUTPUT_DIR/outputs_${SAVE_TAG}_${DATASET}_${DATASET_TYPE}_chunk${K}of${NUM_CHUNKS}_rank*.npz")
        echo "  WARNING: only $n/$EXPECTED_RANKS rank files. Rerun this chunk."
    fi
done

# -- Concat once everything is in place ---------------------------------------
ALL_DONE=true
for ((K=0; K<NUM_CHUNKS; K++)); do
    if ! chunk_done "$K"; then
        ALL_DONE=false
        break
    fi
done

if $ALL_DONE; then
    echo ""
    echo "=== All $NUM_CHUNKS chunks done. Shards are ready in $OUTPUT_DIR ==="
    echo "    (Not merging: the per-chunk/rank .npz shards are consumed directly"
    echo "     downstream. A full merge would need ~2x the split size in RAM.)"
else
    echo ""
    echo "=== Some chunks incomplete. Rerun: bash run_all_chunks.sh ==="
fi
