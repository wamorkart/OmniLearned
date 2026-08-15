#!/bin/bash
# Distill large fine-tuned top-tagging teacher -> DeepSets/PFN student on top
# tagging. First real (non-smoke) run of --arch deep-sets.
#
# NOTE (2026-08-04): cli.py's `train` command declared --arch but never
# forwarded it into run_training(...), so every prior --arch deep-sets run
# (including the original distill_top_deepsets_small_scratch_a05_T4 run)
# silently trained a plain PET2-small model instead. Fixed in cli.py
# (arch=arch now passed through) and confirmed via deepsets_smoke.sh
# (Total params: 0.30M, matches DeepSets small config). The old checkpoint
# files under the distill_top_deepsets_small_scratch_a05_T4 tag are
# mislabeled PET2 checkpoints -- left on disk for reference but not resumed
# from. This run uses a new tag so --resuming can't pick them up.
#
# Teacher: best_model_fine_tune_top_l.pt (large PET2 model fine-tuned on top tagging)
# Student: DeepSets, size=small (base_dim=128, 3 phi layers, 2 rho layers)
# Same KD config (a=0.5/b=0.5/T=4) as the existing small/micro PET2-student
# results, for direct comparability:
#   distill_top_small_scratch_a05_T4  (PET2 small KD): AUC 0.9875, 1/FPR@30%=2804
#   distill_top_micro_scratch_a05_T4  (PET2 micro KD): AUC 0.9876, 1/FPR@30%=2844
# Reuses companion H5 logits already generated from fine_tune_top_l:
#   /pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion_fine_tune_top_l/top/{train,val}
#
# DeepSets ignores --interaction/--local-interaction (no attention/interaction
# matrix in this architecture -- Phi-embed + masked-mean-pool + rho-MLP), so
# those flags are omitted here (train.py's arch==deep-sets branch never reads
# them).
#
# NOTE (2026-08-05): --wandb is intentionally OMITTED. This run was found to
# hang indefinitely with --wandb enabled -- root-caused via a live py-spy
# stack dump to a hang inside loss.backward()'s NCCL all-reduce, not inside
# wandb.init() itself. wandb forks a subprocess (wandb-core) for its
# internal service AFTER DDP() has already established the NCCL
# communicator (model wrap happens before wandb.init() in train.py's run());
# that fork appears to corrupt NCCL's internal state such that the *next*
# collective op (the first gradient all-reduce) hangs. WANDB_START_METHOD=
# thread does NOT fix this -- deprecated/non-functional in wandb 0.27.2.
# Root cause not fixed in code; --wandb dropped to unblock training (same
# workaround used for the MLP-student KD run, distill_train_top_mlp.sh).
# train.py now always writes epoch_log_<save_tag>.csv in the checkpoint dir
# every epoch (independent of wandb), so per-epoch progress is not lost.
#
# Run inside an salloc GPU interactive job, e.g.:
#   salloc -C gpu -q interactive -t 240 --nodes 4 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash distill_train_top_deepsets.sh
#
# For a long-running loop that survives walltime limits/preemption, use
# distill_loop_top_deepsets.sh instead (wraps this script in an salloc
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
  --save-tag distill_top_deepsets_small_scratch_a05_T4_archfix0804 \
  --dataset top --mode classifier --num-classes 2 \
  --path /global/cfs/cdirs/m4567/www/ \
  --arch deep-sets --size small \
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
