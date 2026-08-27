#!/bin/bash
#load libs
module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

# HDF5 file locking causes slow/hanging opens over Lustre-backed CFS; already
# worked around this way elsewhere in the repo (camels.sh, quijote.sh) but
# missing here, where lazy per-sample companion .h5 reads make it most likely
# to matter.
export HDF5_USE_FILE_LOCKING=FALSE

# for DDP
export MASTER_ADDR=$(hostname)

# NCCL_TIMEOUT is not the knob that controls PyTorch's ProcessGroupNCCL watchdog
# -- the watchdog timeout comes from the `timeout=` kwarg to init_process_group(),
# set via DDP_TIMEOUT (90 min) in utils.py:ddp_setup(). That widened timeout plus
# HDF5_USE_FILE_LOCKING=FALSE above are the two fixes for the Jun 15 / Jul 2-3
# NCCL collective-timeout crashes -- neither fix had ever actually been run
# against this workload before this relaunch. NCCL_DEBUG=WARN prints one-line
# summaries on collective failures without the full INFO flood.
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# VALIDATION RELAUNCH (2026-08-05): short --epoch 10 run under a fresh tag to
# confirm the HDF5-locking + DDP_TIMEOUT fixes actually survive the epoch-0
# cold-start file-opening burst on the full 7-dataset pretrain mixture, before
# committing to the full --epoch 500 (~1 pass over the 1.06B-sample mixture,
# ~191h/~8 days at 16 GPUs) run. New tag (not the old distill_pretrain_s_*
# tags) so --resuming can't pick up the old dead epoch-2/epoch-30 checkpoints.
# Once this survives past epoch 2 (further than either prior crash), bump
# --epoch back up to 500 and let --resuming carry the checkpoint forward.
TEACHER_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag distill_pretrain_s_scratch_a05_b05_T4_relaunch0805 \
  --dataset pretrain --mode pretrain --num-classes 210 \
  --path /global/cfs/cdirs/m4567/www/ \
  --size small \
  --use-pid --use-add --use-event-loss --interaction --local-interaction \
  --feature-drop 0.1 \
  --batch 128 --iterations 1000 --epoch 10 \
  --num-workers 4 \
  --distill \
  --teacher-labels-dir $TEACHER_DIR \
  --teacher-tag pretrain_l \
  --distill-alpha 0.5 --distill-beta 0.5 --distill-t 4 \
  --wandb --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
