#!/bin/bash
# Evaluate the large model (fine-tuned on the quark/gluon dataset) on a split.

# In terminal:
# Activate environment before: conda activate /projects/m000255/miniconda/envs/ol_distill/
# Then run bash evaluate_fine_tune_qg_pretrain_l_marlowe.sh

export MASTER_ADDR=$(hostname)

SAVE_TAG_BASE=fine_tune_pretrain_l
DATASET=qg
DESCRIPT_TAG=int_i300_e10
SAVE_TAG=${SAVE_TAG_BASE}_${DATASET}_${DESCRIPT_TAG}

CHECKPOINT_DIR="/projects/m000255/mbenyas/output/${SAVE_TAG_BASE}_${DATASET}"
OUTPUT_DIR="${CHECKPOINT_DIR}"
DATASET_TYPE=${DATASET_TYPE:-test}

mkdir -p "$OUTPUT_DIR"

cmd="omnilearned evaluate \
    -i ${CHECKPOINT_DIR} \
    -o ${OUTPUT_DIR} \
    --save-tag ${SAVE_TAG} \
    --dataset qg \
    --path /projects/m000255/twamorka/ \
    --size large \
    --use-pid \
    --interaction \
    --num-classes 2 \
    --batch 128 \
    --num-workers 4 \
    --dataset-type $DATASET_TYPE"

set -x
srun -N 1 -G 1 --cpus-per-task=4 -A marlowe-m000255 -p preempt --time=00:30:00 -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "