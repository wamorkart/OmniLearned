#!/bin/bash
# Evaluate the CE-only fine-tuned JetClass student on the test split.
# Writes per-rank logits to OUTPUT_DIR as outputs_<tag>_jetclass_test_<rank>.npz.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 60 --nodes 1 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash evaluate_jetclass_finetune.sh

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)

CHECKPOINT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints/
OUTPUT_DIR=/pscratch/sd/t/twamorka/omnilearned/eval/jetclass_finetune/
SAVE_TAG=fine_tune_jetclass_s_pretrain_s
DATASET_TYPE=${DATASET_TYPE:-test}

mkdir -p "$OUTPUT_DIR"

cmd="omnilearned evaluate \
    -i $CHECKPOINT_DIR \
    -o $OUTPUT_DIR \
    --save-tag $SAVE_TAG \
    --dataset jetclass \
    --path /global/cfs/cdirs/m4567/www/ \
    --size small \
    --num-feat 9 \
    --interaction \
    --local-interaction \
    --num-classes 10 \
    --batch 128 \
    --num-workers 4 \
    --dataset-type $DATASET_TYPE"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
