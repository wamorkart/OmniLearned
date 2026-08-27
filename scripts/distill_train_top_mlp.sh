#!/bin/bash
# Distill large fine-tuned top-tagging teacher -> minimal MLP student on top
# tagging (--arch mlp in network.py: masked mean-pool over raw features ->
# 1 hidden layer -> linear classifier, no per-particle embedding). Intended
# as a lower-bound baseline against the DeepSets/PET2-small/micro KD results.
#
# NOTE (2026-08-04): --wandb is intentionally OMITTED here. The DeepSets KD
# run (distill_train_top_deepsets.sh) was found to hang indefinitely with
# --wandb enabled -- root-caused via a live py-spy stack dump to a hang
# inside loss.backward()'s NCCL all-reduce, not inside wandb.init() itself.
# wandb forks a subprocess (wandb-core) for its internal service AFTER
# DDP() has already established the NCCL communicator (model wrap happens
# before wandb.init() in train.py's run()); that fork appears to corrupt
# NCCL's internal state such that the *next* collective op (the first
# gradient all-reduce) hangs. WANDB_START_METHOD=thread does NOT fix this --
# it is deprecated/non-functional in the installed wandb version (0.27.2).
# Root cause not yet fixed in code; --wandb is dropped here to unblock
# training. Progress must be read from the log files / training_*.json
# instead of the wandb dashboard until this is resolved.
#
# Teacher: best_model_fine_tune_top_l.pt (large PET2 model fine-tuned on top tagging)
# Student: MLPStudent, size=small (hidden_dim=64 -- see get_mlp_parameters)
# Same KD config (a=0.5/b=0.5/T=4) as the existing PET2/DeepSets-student
# results, for direct comparability:
#   distill_top_small_scratch_a05_T4  (PET2 small KD): AUC 0.9875, 1/FPR@30%=2804
#   distill_top_micro_scratch_a05_T4  (PET2 micro KD): AUC 0.9876, 1/FPR@30%=2844
# Reuses companion H5 logits already generated from fine_tune_top_l:
#   /pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_l/top/{train,val}
#
# MLPStudent ignores --interaction/--local-interaction (no attention/interaction
# matrix in this architecture), so those flags are omitted here.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash distill_train_top_mlp.sh
#
# For a long-running loop that survives walltime limits/preemption, use
# distill_loop_top_mlp.sh instead (wraps this script in an salloc
# resubmit loop, run inside a screen session).

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

TEACHER_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_l

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag distill_top_mlp_small_scratch_a05_T4 \
  --dataset top --mode classifier --num-classes 2 \
  --path /global/cfs/cdirs/m4567/www/ \
  --arch mlp --size small \
  --batch 128 --iterations 1000 --epoch 50 \
  --lr 5e-4 --wd 0.5 \
  --num-workers 4 \
  --distill \
  --teacher-labels-dir $TEACHER_DIR \
  --teacher-tag fine_tune_top_l \
  --distill-alpha 0.5 --distill-beta 0.5 --distill-t 4 \
  --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
