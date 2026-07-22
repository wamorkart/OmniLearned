#!/bin/bash
# Smoke test for run_all_chunks.sh orchestration. Verifies end-to-end:
#   - salloc + interactive queue allocation grabs the requested resources
#   - evaluate_chunk.sh runs inside the allocation, writes rank files
#   - allocation is released cleanly
#   - the outer loop moves to the next chunk
#   - a chunk that was already done in a prior smoke run is skipped
#
# By construction:
#   - NUM_CHUNKS=1000 so each chunk is tiny (~20k events, ~20 batches/rank).
#   - RUN_CHUNKS="0 1" so only two chunks are launched.
#   - SMOKE outputs go to a separate dir to avoid touching real results.
#   - TIME=30 (the salloc minute budget) is way over the per-chunk wall time,
#     so the test won't be killed by walltime even if it queues a few minutes.
#   - concat_logits.py won't run automatically: the run_all_chunks.sh ALL_DONE
#     check requires ALL 1000 chunks present, which they aren't.
#
# Run from a NERSC login node:
#   bash run_all_chunks_smoke.sh
#
# Expected outcome:
#   - Two allocations are taken and released in sequence.
#   - $SMOKE_OUTPUT_DIR contains 32 .npz files: 16 ranks x 2 chunks.
#   - Script prints "Some chunks incomplete. Rerun:" at the end (expected;
#     this is the smoke not running 1000 chunks).
#
# After the smoke passes:
#   rm -rf $SMOKE_OUTPUT_DIR     # optional cleanup
#   bash run_all_chunks.sh       # the real 6-chunk run

set -euo pipefail
trap 'ec=$?; echo "ERROR: $BASH_SOURCE line $LINENO exited with $ec: $BASH_COMMAND" >&2' ERR

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT"

count_glob() { local n; n=$(ls $1 2>/dev/null | wc -l) || n=0; echo "$n"; }

SMOKE_OUTPUT_DIR=${SMOKE_OUTPUT_DIR:-/pscratch/sd/t/twamorka/omnilearned/teacher_logits/orchestrate_smoke/}
mkdir -p "$SMOKE_OUTPUT_DIR"

echo "=== Smoke: launching run_all_chunks.sh in smoke mode ==="
echo "    NUM_CHUNKS=1000  RUN_CHUNKS='0 1'  TIME=30 min"
echo "    OUTPUT_DIR=$SMOKE_OUTPUT_DIR"
echo ""

NUM_CHUNKS=1000 \
RUN_CHUNKS="0 1" \
TIME=30 \
OUTPUT_DIR="$SMOKE_OUTPUT_DIR" \
bash run_all_chunks.sh

echo ""
echo "=== Smoke: post-run check ==="
SAVE_TAG=pretrain_l
DATASET=atlas
SPLIT=${DATASET_TYPE:-train}
NPATTERN="$SMOKE_OUTPUT_DIR/outputs_${SAVE_TAG}_${DATASET}_${SPLIT}_chunk*of1000_rank*.npz"
n_total=$(count_glob "$NPATTERN")
n_c0=$(count_glob "$SMOKE_OUTPUT_DIR/outputs_${SAVE_TAG}_${DATASET}_${SPLIT}_chunk0of1000_rank*.npz")
n_c1=$(count_glob "$SMOKE_OUTPUT_DIR/outputs_${SAVE_TAG}_${DATASET}_${SPLIT}_chunk1of1000_rank*.npz")

echo "  chunk 0 rank files: $n_c0 / 16"
echo "  chunk 1 rank files: $n_c1 / 16"
echo "  total smoke files : $n_total"

if [ "$n_c0" -eq 16 ] && [ "$n_c1" -eq 16 ]; then
    echo ""
    echo "=== SMOKE PASSED ==="
    echo "Orchestration works. Clean up with: rm -rf $SMOKE_OUTPUT_DIR"
    echo "Then run the real evaluation: bash run_all_chunks.sh"
    exit 0
else
    echo ""
    echo "=== SMOKE FAILED ==="
    echo "At least one chunk didn't produce all 16 rank files."
    echo "Inspect $SMOKE_OUTPUT_DIR and the script output above."
    exit 1
fi
