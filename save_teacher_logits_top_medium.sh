#!/bin/bash
# Stage 2 of the TAKD experiment: generate teacher logits from the finished
# medium student (best_model_distill_top_medium_scratch_a00_b10_T4.pt) so it
# can act as the teacher for the final medium->small stage.
#
# Mirrors save_teacher_logits_top.sh (which did this for the large model)
# exactly, just pointed at the medium checkpoint/size and writing to its own
# npz/companion dirs so it never touches the large teacher's cached logits.
#
# Only run this AFTER distill_loop_top_medium.sh has produced
# best_model_distill_top_medium_scratch_a00_b10_T4.pt.
#
# Run inside an salloc GPU interactive job:
#   salloc -C gpu -q interactive -t 240 --nodes 1 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash save_teacher_logits_top_medium.sh

set -euo pipefail

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO=/global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned
CHECKPOINT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints/
NPZ_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/top_medium
COMPANION_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_distill_top_medium
DATA_PATH=/global/cfs/cdirs/m4567/www/
TAG=distill_top_medium_scratch_a00_b10_T4
DATASET=top

CKPT="${CHECKPOINT_DIR}/best_model_${TAG}.pt"
if [ ! -f "$CKPT" ]; then
    echo "ERROR: $CKPT not found -- run distill_loop_top_medium.sh (stage 1) first." >&2
    exit 1
fi

mkdir -p "$NPZ_DIR"

evaluate_split() {
    local SPLIT=$1
    echo "=== $(date '+%F %T')  evaluating $DATASET/$SPLIT ==="
    srun -l -u bash -c "
        source $REPO/export_ddp.sh
        omnilearned evaluate \
          -i $CHECKPOINT_DIR \
          -o $NPZ_DIR \
          --save-tag $TAG \
          --dataset $DATASET \
          --path $DATA_PATH \
          --size medium \
          --interaction \
          --local-interaction \
          --mode classifier \
          --num-classes 2 \
          --batch 32 \
          --num-workers 4 \
          --dataset-type $SPLIT
    "
    echo "=== $(date '+%F %T')  $DATASET/$SPLIT done ==="
}

# --- Phase 1: generate NPZ shards for train and val ---
evaluate_split train
evaluate_split val

# --- Phase 2: convert NPZ shards -> companion H5 files ---
echo "=== $(date '+%F %T')  building companion H5 files ==="
python3 "$REPO/build_teacher_h5.py" \
    --npz-dir "$NPZ_DIR" \
    --tag "$TAG" \
    --data-path "$DATA_PATH" \
    --out-dir "$COMPANION_DIR" \
    --dataset "$DATASET" \
    --split "train,val" \
    --skip-existing

echo "=== $(date '+%F %T')  all done — companions in $COMPANION_DIR/$DATASET/ ==="
