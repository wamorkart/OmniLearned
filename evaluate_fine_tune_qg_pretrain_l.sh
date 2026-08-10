#!/bin/bash
# Evaluate the large model (fine-tuned on the quark/gluon dataset) on a split.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 60 --nodes 1 --ntasks-per-node 4 --gpus-per-node 4 -A m3246 
# then run bash evaluate_fine_tune_qg_pretrain_l.sh

module load conda
conda activate ol_distill
# module load pytorch

export MASTER_ADDR=$(hostname)

# ============================================================
#  EDIT THESE PER RUN -- everything else derives from them
#  also don't forget to hardcode interaction / local-interaction terms
# ============================================================
SAVE_TAG_BASE=fine_tune_pretrain_l
DATASET=qg
DESCRIPT_TAG=int_i300_e10 # int_localint_i300_e10 #nointterms_i300_e10
DATASET_TYPE=${DATASET_TYPE:-test}
QUANTIZATION=none                          # "none", "int8", "int8dq", or "bf16"
# ============================================================

case "$QUANTIZATION" in
    none|int8|int8dq|bf16) ;;
    *)
        echo "ERROR: QUANTIZATION must be 'none', 'int8', 'int8dq', or 'bf16', got '$QUANTIZATION'" >&2
        exit 1
        ;;
esac

SAVE_TAG="${SAVE_TAG_BASE}_${DATASET}_${DESCRIPT_TAG}"
CHECKPOINT_DIR=/pscratch/sd/m/mbenyas
OUTPUT_DIR=/pscratch/sd/m/mbenyas/${SAVE_TAG_BASE}_${DATASET}_${DESCRIPT_TAG}_${QUANTIZATION}

mkdir -p "$OUTPUT_DIR"
export QUANTIZE=${QUANTIZATION}

cmd="omnilearned evaluate \
    -i $CHECKPOINT_DIR \
    -o $OUTPUT_DIR \
    --save-tag $SAVE_TAG \
    --dataset ${DATASET} \
    --path /global/cfs/cdirs/m4567/www/ \
    --size large \
    --use-pid \
    --interaction \
    --num-classes 2 \
    --batch 128 \
    --num-workers 4 \
    --dataset-type $DATASET_TYPE"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    which omnilearned
    python -c 'import omnilearned; print(omnilearned.__file__)'
    $cmd
    "