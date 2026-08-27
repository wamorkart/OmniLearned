#!/bin/bash
# Evaluate fine_tune_top_l on the top dataset to generate teacher logits for KD.
# Run once for train and once for val (controlled by DATASET_TYPE).
#
# Usage (two separate interactive jobs, or sequentially in one):
#   DATASET_TYPE=train bash evaluate_top_for_distill.sh
#   DATASET_TYPE=val   bash evaluate_top_for_distill.sh

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)

CHECKPOINT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints/
OUTPUT_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/TEST/
SAVE_TAG=fine_tune_top_l
DATASET_TYPE=${DATASET_TYPE:-train}

mkdir -p "$OUTPUT_DIR"

cmd="omnilearned evaluate \
    -i $CHECKPOINT_DIR \
    -o $OUTPUT_DIR \
    --save-tag $SAVE_TAG \
    --dataset top \
    --path /global/cfs/cdirs/m4567/www/ \
    --size large \
    --interaction \
    --local-interaction \
    --num-classes 2 \
    --batch 64 \
    --num-workers 4 \
    --dataset-type $DATASET_TYPE"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
