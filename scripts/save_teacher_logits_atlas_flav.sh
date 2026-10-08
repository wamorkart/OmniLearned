#!/bin/bash
# Save atlas_flav teacher logits (train + val) for KD, then build the
# per-source-file companion H5s that `--teacher-labels-dir` reads.
#
# Teacher: fine_tune_atlas_flav_m_localint_v7 (best ftag fine-tune; scores in
# docs/ATLAS_FLAV_RESULTS.md). Flags mirror
# scripts/eval_atlas_flav_ftag_m_localint_v7_interactive.sh.
#
# The train split is ~168M jets, so each split is evaluated in NUM_CHUNKS
# pieces (one srun per chunk) to bound per-rank memory; build_teacher_h5.py
# globs every chunk's npz. A chunk whose 16 rank files already exist is
# skipped, so the script can be rerun after a timeout.
#
# Run inside an allocation, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash scripts/save_teacher_logits_atlas_flav.sh

set -euo pipefail

module load conda
conda activate "${OMNILEARNED_ENV:-/global/homes/t/twamorka/omnilearned-clean/env}"
module load pytorch

export MASTER_ADDR=$(hostname)
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO="${OMNILEARNED_REPO:-/global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned}"
SCRATCH="${OMNILEARNED_SCRATCH:-/pscratch/sd/t/twamorka/omnilearned}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$SCRATCH/checkpoints/}"
TAG="${TAG:-fine_tune_atlas_flav_m_localint_v7}"
NPZ_DIR="${NPZ_DIR:-$SCRATCH/teacher_logits/atlas_flav_$TAG}"
COMPANION_DIR="${COMPANION_DIR:-$SCRATCH/teacher_logits/companion_$TAG}"
DATA_PATH="${DATA_PATH:-$SCRATCH/datasets/}"
NUM_CHUNKS="${NUM_CHUNKS:-4}"
DATASET=atlas_flav

cd "$REPO"
mkdir -p "$NPZ_DIR"
NRANKS="${SLURM_NTASKS:?run inside an salloc/sbatch allocation}"

evaluate_chunk() {
    local SPLIT=$1 CHUNK=$2 NCH=$3
    local suffix="" have
    [ "$NCH" -gt 1 ] && suffix="_chunk${CHUNK}of${NCH}"
    have=$(ls "$NPZ_DIR"/outputs_${TAG}_${DATASET}_${SPLIT}${suffix}_rank*.npz 2>/dev/null | wc -l || true)
    if [ "$have" -eq "$NRANKS" ]; then
        echo "=== $(date '+%F %T')  $SPLIT chunk $CHUNK/$NCH already done, skipping ==="
        return
    fi
    echo "=== $(date '+%F %T')  evaluating $DATASET/$SPLIT chunk $CHUNK/$NCH ==="
    srun -l -u bash -c "
        source $REPO/export_ddp.sh
        EVAL_AMP=bf16 omnilearned evaluate \
          -i $CHECKPOINT_DIR \
          -o $NPZ_DIR \
          --save-tag $TAG \
          --dataset $DATASET --mode classifier \
          --path $DATA_PATH \
          --size medium \
          --num-feat 4 \
          --use-add --num-add 17 --num-classes 4 \
          --conditional --num-cond 4 --interaction --local-interaction \
          --batch 512 --num-workers 8 \
          --dataset-type $SPLIT \
          --num-chunks $NCH --chunk-idx $CHUNK
    "
    echo "=== $(date '+%F %T')  $SPLIT chunk $CHUNK/$NCH done ==="
}

for ((c = 0; c < NUM_CHUNKS; c++)); do
    evaluate_chunk train "$c" "$NUM_CHUNKS"
done
evaluate_chunk val 0 1

echo "=== $(date '+%F %T')  building companion H5 files ==="
python3 "$REPO/tools/preprocess/build_teacher_h5.py" \
    --npz-dir "$NPZ_DIR" \
    --tag "$TAG" \
    --data-path "$DATA_PATH" \
    --out-dir "$COMPANION_DIR" \
    --dataset "$DATASET" \
    --split "train,val" \
    --skip-existing

echo "=== $(date '+%F %T')  all done -- companions in $COMPANION_DIR/$DATASET/ ==="
