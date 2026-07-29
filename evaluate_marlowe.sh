#!/bin/bash
#SBATCH -A marlowe-m000255
#SBATCH -p preempt
#SBATCH -N 1
#SBATCH -G 1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH -o evaluate_distill_top_small_%j.out

# Evaluation script.
# Once sbatch evaluate_marlowe.sh is submitted,
# track it with squeue -u $USER and
# tail -f evaluate_distill_top_small_400173.out

# export MASTER_ADDR=$(hostname)

# Evaluate a distilled small student model (trained from scratch, KD) on the
# top-tagging dataset.
SAVE_TAG=distill_top_small_scratch_a05_T4
CHECKPOINT_DIR=/projects/m000255/twamorka/checkpoints
OUTPUT_DIR=/projects/m000255/mbenyas/output/${SAVE_TAG}   # where my eval outputs land
DATASET_TYPE=${DATASET_TYPE:-test}

# make sure these match how the checkpoint was trained
DATASET=top
PATH_TO_DATA=/projects/m000255/twamorka/
SIZE=small
NUM_CLASSES=2                                # top-tagging is binary
USE_PID_FLAG=""                              # set to "" or --use-pid
INTERACTION_FLAG=--interaction               # set to "" or --interaction
LOCAL_INTERACTION_FLAG=--local-interaction   # set to "" or --local-interaction

mkdir -p "$OUTPUT_DIR"

omnilearned evaluate \
    -i ${CHECKPOINT_DIR} \
    -o ${OUTPUT_DIR} \
    --save-tag ${SAVE_TAG} \
    --dataset ${DATASET} \
    --path ${PATH_TO_DATA} \
    --size ${SIZE} \
    ${USE_PID_FLAG} \
    ${INTERACTION_FLAG} \
    ${LOCAL_INTERACTION_FLAG} \
    --num-classes ${NUM_CLASSES} \
    --batch 128 \
    --num-workers 4 \
    --dataset-type $DATASET_TYPE \

# cmd="omnilearned evaluate \
#     -i ${CHECKPOINT_DIR} \
#     -o ${OUTPUT_DIR} \
#     --save-tag ${SAVE_TAG} \
#     --dataset ${DATASET} \
#     --path ${PATH_TO_DATA} \
#     --size ${SIZE} \
#     ${USE_PID_FLAG} \
#     ${INTERACTION_FLAG} \
#     --num-classes ${NUM_CLASSES} \
#     --batch 128 \
#     --num-workers 4 \
#     --dataset-type $DATASET_TYPE"

# set -x
# srun -N 1 -G 1 --cpus-per-task=4 -A marlowe-m000255 -p preempt --time=00:30:00 -l -u \
#     bash -c "
#     source export_ddp.sh
#     $cmd
#     "