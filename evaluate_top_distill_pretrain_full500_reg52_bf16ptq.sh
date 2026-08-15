#!/bin/bash
# Evaluate fine_tune_top_distill_pretrain_s_a05_b05_T4_full500_reg52_bf16ptq on
# the top test split (distill -> quantize -> fine-tune lineage). Compare
# against evaluate_top_distill_pretrain_full500_reg52.sh's output (same
# lineage, no PTQ step) via compute_metrics_top.py.
# Writes per-rank outputs to OUTPUT_DIR for use with compute_metrics_top.py.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 60 --nodes 1 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash evaluate_top_distill_pretrain_full500_reg52_bf16ptq.sh

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)

CHECKPOINT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints/
OUTPUT_DIR=/pscratch/sd/t/twamorka/omnilearned/eval/top_distill_pretrain_full500_reg52_bf16ptq/
SAVE_TAG=fine_tune_top_distill_pretrain_s_a05_b05_T4_full500_reg52_bf16ptq
DATASET_TYPE=${DATASET_TYPE:-test}

mkdir -p "$OUTPUT_DIR"

cmd="omnilearned evaluate \
    -i $CHECKPOINT_DIR \
    -o $OUTPUT_DIR \
    --save-tag $SAVE_TAG \
    --dataset top \
    --path /global/cfs/cdirs/m4567/www/ \
    --size small \
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
