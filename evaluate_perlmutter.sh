#!/bin/bash
# Evaluate a checkpoint (e.g. the distilled small student trained on top) on a split.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 60 --nodes 1 --ntasks-per-node 4 --gpus-per-node 4 -A m3246
# then run bash evaluate_perlmutter.sh

module load conda
conda activate ol_distill

export MASTER_ADDR=$(hostname)

# ============================================================
#  EDIT THESE TWO LINES PER RUN -- everything else derives from them
# ============================================================
SAVE_TAG=distill_top_small_scratch_a05_T4   # checkpoint name, under $CHECKPOINT_DIR
QUANTIZATION=int8dq                           # "none", "int8", "int8dq", or "bf16"
# ============================================================

# case "$QUANTIZATION" in
#     none|int8|int8dq|bf16) ;;
#     *)
#         echo "ERROR: QUANTIZATION must be 'none', 'int8', 'int8dq', or 'bf16', got '$QUANTIZATION'" >&2
#         exit 1
#         ;;
# esac

CHECKPOINT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints/
OUTPUT_DIR=/pscratch/sd/m/mbenyas/${SAVE_TAG}_${QUANTIZATION}   # my own scratch -- where eval outputs land
DATASET_TYPE=${DATASET_TYPE:-test}

# make sure these match how the checkpoint was trained
DATASET=top
MODE=classifier
PATH_TO_DATA=/global/cfs/cdirs/m4567/www/
SIZE=small
NUM_CLASSES=2                                # top-tagging is binary
BATCH_SIZE=128
USE_PID_FLAG=""                              
INTERACTION_FLAG=--interaction
LOCAL_INTERACTION_FLAG=--local-interaction

mkdir -p "$OUTPUT_DIR"
export QUANTIZE=${QUANTIZATION}
export CUDA_LAUNCH_BLOCKING=1  # forces synchronous CUDA calls so crash tracebacks point at the real failing kernel, not a later op

cmd="omnilearned evaluate \
    -i ${CHECKPOINT_DIR} \
    -o ${OUTPUT_DIR} \
    --save-tag ${SAVE_TAG} \
    --dataset ${DATASET} \
    --mode ${MODE} \
    --path ${PATH_TO_DATA} \
    --size ${SIZE} \
    ${USE_PID_FLAG} \
    ${INTERACTION_FLAG} \
    ${LOCAL_INTERACTION_FLAG} \
    --num-classes ${NUM_CLASSES} \
    --batch ${BATCH_SIZE} \
    --num-workers 4 \
    --dataset-type $DATASET_TYPE"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "