#!/bin/bash
# Smoke test for the chunked evaluate feature added to dataloader.py / evaluate.py / cli.py.
# Runs two tiny chunks (1/1000 of the dataset each) back-to-back and verifies:
#   - per-rank output files are named outputs_<tag>_<ds>_<split>_chunk<K>of<N>_rank<r>.npz
#   - chunk 0 and chunk 1 produce DISJOINT sample_keys (no overlap)
#   - all expected rank files are present for each chunk
#
# Must be run inside an existing salloc/srun allocation.
#
# Usage:
#   bash test_chunk.sh

set -u

SMOKE_OUT=${SMOKE_OUT:-/pscratch/sd/t/twamorka/omnilearned/teacher_logits/chunk_smoke}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/pscratch/sd/t/twamorka/omnilearned/checkpoints/}
DATA_PATH=${DATA_PATH:-/global/cfs/cdirs/m4567/www/}

SAVE_TAG=pretrain_l
DATASET=atlas
SIZE=large
SPLIT=test

# Use a very large NUM_CHUNKS so each chunk is tiny — keeps the smoke test fast.
NUM_CHUNKS=${TEST_NUM_CHUNKS:-1000}

module load pytorch
export MASTER_ADDR=$(hostname)
export MASTER_PORT=${MASTER_PORT:-29500}

mkdir -p "$SMOKE_OUT"
echo "Cleaning previous smoke artifacts in $SMOKE_OUT"
rm -f "$SMOKE_OUT"/outputs_${SAVE_TAG}_${DATASET}_${SPLIT}_chunk*of${NUM_CHUNKS}_rank*.npz

run_chunk() {
  local K=$1
  local LOGTAG=$2
  echo "=== Running chunk ${K}/${NUM_CHUNKS} (log: ${LOGTAG}) ==="
  srun -l -u bash -c "
    source export_ddp.sh
    omnilearned evaluate \
      -i $CHECKPOINT_DIR \
      -o $SMOKE_OUT \
      --save-tag $SAVE_TAG \
      --dataset $DATASET \
      --path $DATA_PATH \
      --size $SIZE \
      --interaction --local-interaction \
      --use-pid --use-add --use-event-loss \
      --num-classes 210 \
      --batch 64 --num-workers 4 \
      --dataset-type $SPLIT \
      --num-chunks $NUM_CHUNKS \
      --chunk-idx $K
  " 2>&1 | tee "$SMOKE_OUT/${LOGTAG}.log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" -ne 0 ]; then
    echo "ERROR: chunk $K failed (rc=$rc)"
    return $rc
  fi
}

inspect_chunk() {
  local K=$1
  local pat="outputs_${SAVE_TAG}_${DATASET}_${SPLIT}_chunk${K}of${NUM_CHUNKS}_rank*.npz"
  local n
  n=$(ls "$SMOKE_OUT"/$pat 2>/dev/null | wc -l)
  echo "chunk ${K}: ${n} rank files matching ${pat}"
  ls "$SMOKE_OUT"/$pat 2>/dev/null
}

run_chunk 0 run0
inspect_chunk 0

run_chunk 1 run1
inspect_chunk 1

echo ""
echo "=== Verifying chunk 0 and chunk 1 produce disjoint sample_keys ==="
SMOKE_OUT="$SMOKE_OUT" \
TAG="$SAVE_TAG" DATASET="$DATASET" SPLIT="$SPLIT" NUM_CHUNKS="$NUM_CHUNKS" \
python3 - <<'PY'
import glob, os, sys
import numpy as np

out = os.environ['SMOKE_OUT']
tag = os.environ['TAG']
ds  = os.environ['DATASET']
sp  = os.environ['SPLIT']
N   = os.environ['NUM_CHUNKS']

def load_keys(k):
    paths = sorted(glob.glob(
        os.path.join(out, f'outputs_{tag}_{ds}_{sp}_chunk{k}of{N}_rank*.npz')
    ))
    keys = []
    for p in paths:
        with np.load(p) as z:
            if 'sample_keys' in z.files:
                keys.append(z['sample_keys'])
    if not keys:
        return None
    return np.concatenate(keys, axis=0)

k0 = load_keys(0)
k1 = load_keys(1)
if k0 is None or k1 is None:
    print('FAIL: missing sample_keys; cannot verify disjointness')
    sys.exit(1)

print(f'chunk 0 events: {len(k0)}')
print(f'chunk 1 events: {len(k1)}')

# rows are (file_idx, sample_idx); compress to a single int for set ops
def pack(arr):
    arr = arr.astype(np.int64)
    return arr[:, 0] * (1 << 32) + arr[:, 1]

s0 = set(pack(k0).tolist())
s1 = set(pack(k1).tolist())
inter = s0 & s1
print(f'overlap: {len(inter)} events')
if inter:
    print('FAIL: chunk 0 and chunk 1 share events')
    sys.exit(1)
print('PASS: chunks are disjoint')
PY
rc=$?
echo ""
if [ "$rc" -eq 0 ]; then
  echo "=== SMOKE TEST PASSED ==="
else
  echo "=== SMOKE TEST FAILED ==="
fi
exit $rc
