#!/bin/bash
# Evaluate the atlas_flav ftag localint re-run (fine_tune_atlas_flav_m_localint_v3,
# script train_atlas_flav_ftag_m_localint_reg8.sbatch, job 57925112) on the
# atlas_flav TEST split. Copy of eval_atlas_flav_ftag_m_interactive.sh with the
# save-tag / output dir pointed at the _v2 checkpoint.
#
# localint_v3 is the warmup-2 + lr-2e-5 stability-fix run. It was killed by its
# 24h wall limit mid-epoch-8; the auto-resubmit chain did NOT fire (whole job
# cancelled, not just the step). best_model is epoch 5 (best by total val loss;
# class val loss 0.6140) -- the clean pre-spike checkpoint, i.e. what "the
# epoch-4 checkpoint" refers to (epoch 4/5 are equivalent, class val 0.613 vs
# 0.614; epoch 6 blew up, train class loss 3.75). This scores it against the
# paper and the localint_v2 epoch-15 numbers. Re-run if training resumes.
#
# evaluate.py loads best_model_<save-tag>.pt (utils.get_checkpoint_name). Its
# test_step only materialises predictions for mode in {classifier, regression,
# segmentation}; "ftag" writes nothing. The ftag network computes the
# jet-flavour head as y_pred = classifier(x_body) for BOTH mode="ftag" and
# mode="classifier" (network.py forward) and the checkpoint carries a full
# classifier_head, so --mode classifier reproduces exactly the ftag jet-flavour
# logits. The per-track origin head is not scored.
#
# restore_checkpoint runs fine_tune=False: body strict=False, classifier_head
# strict=True, generator is None in classifier mode -> skipped.
#
# CRITICAL when scoring: label 2 = b, label 0 = light (NOT GN2 convention) --
# see atlas-flav-btagging-progress memory. Score with
#   python tools/metrics/compute_metrics_flav.py \
#       --indir /pscratch/sd/t/twamorka/omnilearned/eval/atlas_flav_ftag_m_localint_v3/ \
#       --tag fine_tune_atlas_flav_m_localint_v3
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash scripts/eval_atlas_flav_ftag_m_localint_v3_interactive.sh
# or via the resubmit loop: scripts/eval_loop_atlas_flav_ftag_m_localint_v3.sh

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CHECKPOINT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints/
OUTPUT_DIR=/pscratch/sd/t/twamorka/omnilearned/eval/atlas_flav_ftag_m_localint_v3/
mkdir -p "$OUTPUT_DIR"

# One 1/6 chunk of the test split (~4M jets) is ample for AUC / rejection at
# the working points. Set NUM_CHUNKS=1 to score the full ~25M-jet split.
NUM_CHUNKS=${NUM_CHUNKS:-6}
CHUNK_IDX=${CHUNK_IDX:-0}

cmd="omnilearned evaluate \
  -i $CHECKPOINT_DIR \
  -o $OUTPUT_DIR \
  --save-tag fine_tune_atlas_flav_m_localint_v3 \
  --dataset atlas_flav --mode classifier \
  --path /global/cfs/cdirs/m4567/www/ \
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
