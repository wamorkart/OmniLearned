#!/bin/bash
# Smoke test for sharded + chunked eval. Runs inside an existing SLURM
# allocation (use salloc to obtain one). Drives two short eval passes
# against shard 0 of the small `top` dataset, verifies chunks landed,
# fp16 dtype is correct, and that the second pass resumes from the first.

set -u

SMOKE_OUT=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/smoke
CHECKPOINT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints/
DATA_PATH=.
PROJECT=/global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned

# Smoke uses `top` (small + local) — keep DATASET in sync with --dataset below.
DATASET=top
SAVE_TAG=pretrain_l
NUM_SHARDS=64
SHARD_ID=0
SPLIT=test
PATTERN_BASE="outputs_${SAVE_TAG}_${DATASET}_${SPLIT}_shard${SHARD_ID}of${NUM_SHARDS}"

CONDA_ENV=/global/homes/t/twamorka/omnilearned-clean/env

cd "$PROJECT"
module load conda
conda activate "$CONDA_ENV"
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

mkdir -p "$SMOKE_OUT"
rm -f "$SMOKE_OUT"/${PATTERN_BASE}_rank*_chunk*.npz \
      "$SMOKE_OUT"/${PATTERN_BASE}_rank*_chunk*.npz.tmp \
      "$SMOKE_OUT"/run1.log "$SMOKE_OUT"/run2.log

run_eval() {
  local TLIMIT=$1
  local LOGTAG=$2
  echo "=== Running eval with ${TLIMIT}s timeout (log: ${LOGTAG}) ==="
  timeout "$TLIMIT" srun -l -u bash -c "
    module load conda
    conda activate $CONDA_ENV
    source export_ddp.sh
    omnilearned evaluate \
      -i $CHECKPOINT_DIR \
      -o $SMOKE_OUT \
      --save-tag $SAVE_TAG \
      --dataset $DATASET \
      --path $DATA_PATH \
      --size large \
      --mode classifier --num-classes 210 \
      --use-event-loss \
      --interaction --local-interaction \
      --batch 64 --num-workers 4 \
      --num-shards $NUM_SHARDS --shard-id $SHARD_ID --flush-every 10
  " 2>&1 | tee "$SMOKE_OUT/${LOGTAG}.log" || true
}

inspect_chunks() {
  local TAG=$1
  echo "=== Chunk inventory (${TAG}) ==="
  ls "$SMOKE_OUT"/${PATTERN_BASE}_rank*_chunk*.npz 2>/dev/null | wc -l
  echo "Per-rank chunk counts:"
  for r in $(seq 0 15); do
    cnt=$(ls "$SMOKE_OUT"/${PATTERN_BASE}_rank${r}_chunk*.npz 2>/dev/null | wc -l)
    printf "  rank %2d: %s chunks\n" "$r" "$cnt"
  done
}

# Run 1 ------------------------------------------------------------------
echo "===================== RUN 1 ====================="
run_eval 600 run1
inspect_chunks "after run 1"

# Sanity-check one chunk
echo "=== Inspecting rank 0, chunk 0 ==="
SMOKE_OUT="$SMOKE_OUT" PATTERN_BASE="$PATTERN_BASE" python3 - <<'PY'
import numpy as np, glob, os
out = os.environ['SMOKE_OUT']
base = os.environ['PATTERN_BASE']
fs = sorted(glob.glob(os.path.join(out, f'{base}_rank0_chunk*.npz')))
print(f'rank0 chunks: {len(fs)}')
if fs:
    d = np.load(fs[0])
    print('keys:', list(d.keys()))
    print('logits:', d['logits'].dtype, d['logits'].shape)
    if 'event_logits' in d.files:
        print('event_logits:', d['event_logits'].dtype, d['event_logits'].shape)
    print('pid:', d['pid'].dtype, d['pid'].shape)
PY

PRE=$(ls "$SMOKE_OUT"/${PATTERN_BASE}_rank0_chunk*.npz 2>/dev/null | wc -l)

# Run 2 - resume --------------------------------------------------------
echo "===================== RUN 2 (resume) ====================="
run_eval 300 run2
inspect_chunks "after run 2"

POST=$(ls "$SMOKE_OUT"/${PATTERN_BASE}_rank0_chunk*.npz 2>/dev/null | wc -l)
echo "rank0 chunks: PRE=$PRE  POST=$POST"
if [ "$POST" -gt "$PRE" ]; then
  echo "RESUME OK (POST > PRE)"
else
  echo "RESUME FAILED (POST not greater than PRE — check stdout for [resume] log line)"
fi
