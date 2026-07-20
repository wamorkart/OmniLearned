#!/bin/bash
# Evaluate the large model (fine-tuned on the quark/gluon dataset and distilled) on the test split.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 60 --nodes 1 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash evaluate_fine_tuned_qg_l.sh

module load conda
conda activate ol_distill
module load pytorch

export MASTER_ADDR=$(hostname)

CHECKPOINT_DIR=/pscratch/sd/m/mbenyas
OUTPUT_DIR=/pscratch/sd/m/mbenyas
SAVE_TAG=distill_qg_small_scratch_a05_T4
DATASET_TYPE=${DATASET_TYPE:-test}

mkdir -p "$OUTPUT_DIR"

cmd="omnilearned evaluate \
    -i $CHECKPOINT_DIR \
    -o $OUTPUT_DIR \
    --save-tag $SAVE_TAG \
    --dataset qg --mode classifier \
    --path /global/cfs/cdirs/m4567/www/ \
    --size small \
    --use-pid \
    --interaction \
    --local-interaction \
    --num-classes 2 \
    --batch 128 \
    --num-workers 4 \
    --dataset-type $DATASET_TYPE"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
