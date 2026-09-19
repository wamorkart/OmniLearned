#!/bin/bash
# Evaluate the atlas_flav ftag localint "v5" fine-tune (fine_tune_atlas_flav_m_localint_v5,
# scripts/train_atlas_flav_ftag_m_localint_v5.sbatch / train_loop_atlas_flav_ftag_m_localint_v5.sh)
# on the atlas_flav TEST split. Copy of eval_atlas_flav_ftag_m_localint_v3_interactive.sh
# with the save-tag / output dir pointed at the v5 checkpoint.
#
# v5 = clean upstream train.sh HPs (lr 5e-5, no warmup, global batch 4096) +
# --local-interaction, reading the dataset from the pscratch stage. Unlike
# v3, it did NOT collapse: ran the full 30/30 epochs to completion (exit 0,
# 2026-09-10 22:45), loss/val-loss plateaued smoothly over the last 5 epochs
# (train 0.9410->0.9393, val 0.9459->0.9444), best checkpoint at epoch 30.
# best_model_fine_tune_atlas_flav_m_localint_v5.pt is the one to score.
#
# --path points at the pscratch dataset copy (same as the v5 training run)
# rather than CFS, since the pscratch stage is already local and faster.
#
# CRITICAL when scoring: label 2 = b, label 0 = light (NOT GN2 convention) --
# see atlas-flav-btagging-progress memory. Score with
#   python tools/metrics/compute_metrics_flav.py \
#       --indir /pscratch/sd/t/twamorka/omnilearned/eval/atlas_flav_ftag_m_localint_v5/ \
#       --tag fine_tune_atlas_flav_m_localint_v5
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash scripts/eval_atlas_flav_ftag_m_localint_v5_interactive.sh
# or via the resubmit loop: scripts/eval_loop_atlas_flav_ftag_m_localint_v5.sh

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHECKPOINT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints/
OUTPUT_DIR=/pscratch/sd/t/twamorka/omnilearned/eval/atlas_flav_ftag_m_localint_v5/
mkdir -p "$OUTPUT_DIR"

# One 1/6 chunk of the test split (~4M jets) is ample for AUC / rejection at
# the working points. Set NUM_CHUNKS=1 to score the full ~25M-jet split.
NUM_CHUNKS=${NUM_CHUNKS:-6}
CHUNK_IDX=${CHUNK_IDX:-0}

cmd="omnilearned evaluate \
  -i $CHECKPOINT_DIR \
  -o $OUTPUT_DIR \
  --save-tag fine_tune_atlas_flav_m_localint_v5 \
  --dataset atlas_flav --mode classifier \
  --path /pscratch/sd/t/twamorka/omnilearned/datasets/ \
  --size medium \
  --num-feat 4 \
  --use-add --num-add 17 --num-classes 4 \
  --conditional --num-cond 4 --interaction --local-interaction \
  --batch 512 --num-workers 8 \
  --dataset-type test \
  --num-chunks $NUM_CHUNKS --chunk-idx $CHUNK_IDX"

echo "[$(date '+%F %T')] chunk ${CHUNK_IDX}/${NUM_CHUNKS}"
echo "$cmd"

set -x
srun -l -u bash -c "
  source export_ddp.sh
  EVAL_AMP=bf16 $cmd
  "
