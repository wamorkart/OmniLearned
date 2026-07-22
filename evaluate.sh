#load libs
module load pytorch

# for DDP
export MASTER_ADDR=$(hostname)

# Evaluate the large model and write per-rank logits to $OUTPUT_DIR as
# outputs_<tag>_<dataset>_<split>_<rank>.npz.
#
# Override DATASET_TYPE to pick the split: test (default), train, or val.

CHECKPOINT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints/
OUTPUT_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/TEST/
SAVE_TAG=fine_tune_top_s
DATASET=top
SIZE=small
DATASET_TYPE=${DATASET_TYPE:-test}

mkdir -p "$OUTPUT_DIR"

cmd="omnilearned evaluate \
    -i $CHECKPOINT_DIR \
    -o $OUTPUT_DIR \
    --save-tag $SAVE_TAG \
    --dataset $DATASET \
    --path /global/cfs/cdirs/m4567/www/ \
    --size $SIZE \
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