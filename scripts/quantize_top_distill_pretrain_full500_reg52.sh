#!/bin/bash
# PTQ step of the distill -> quantize -> fine-tune study on top tagging.
# Round-trips best_model_distill_pretrain_s_scratch_a05_b05_T4_full500_reg52.pt
# (small student, pretrain-KD, a=0.5/b=0.5, T=4, full 154-epoch pass at
# world_size=52, job 56540549 -- see [[distill-lazy-teacher-progress]]) through
# bfloat16 and back, saving a new checkpoint with the quantization-rounded
# weights. No GPU/DDP needed -- runs on a login node in seconds.
# Produces: best_model_distill_pretrain_s_scratch_a05_b05_T4_full500_reg52_bf16ptq.pt

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env

python quantize_checkpoint.py \
    --checkpoint-dir /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
    --tag distill_pretrain_s_scratch_a05_b05_T4_full500_reg52 \
    --dtype bfloat16
