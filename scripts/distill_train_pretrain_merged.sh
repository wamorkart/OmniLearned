#!/bin/bash
#load libs
module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

# HDF5 file locking causes slow/hanging opens over Lustre-backed CFS; already
# worked around this way elsewhere in the repo (camels.sh, quijote.sh).
export HDF5_USE_FILE_LOCKING=FALSE

# for DDP
export MASTER_ADDR=$(hostname)

# NCCL_TIMEOUT is not the knob that controls PyTorch's ProcessGroupNCCL watchdog
# -- the watchdog timeout comes from the `timeout=` kwarg to init_process_group(),
# set via DDP_TIMEOUT (90 min) in utils.py:ddp_setup(). NCCL_DEBUG=WARN prints
# one-line summaries on collective failures without the full INFO flood.
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Flight-recorder auto-dump on collective timeout: writes a per-rank trace of
# recent NCCL ops to disk so a hang can be root-caused from the dump instead
# of guessed at from log timestamps. None of the 4 hangs on 2026-08-07 left a
# dump file despite the log claiming an attempt -- these vars weren't set.
export TORCH_NCCL_DUMP_ON_TIMEOUT=1
# TORCH_NCCL_TRACE_BUFFER_SIZE is deprecated in this torch version (renamed
# TORCH_FR_BUFFER_SIZE) -- set both so the buffer size is honored regardless
# of which name this torch build actually reads.
export TORCH_FR_BUFFER_SIZE=2000
export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
DUMP_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/nccl_trace
mkdir -p "$DUMP_DIR"
export TORCH_NCCL_DEBUG_INFO_TEMP_FILE="$DUMP_DIR/dump_${SLURM_JOB_ID}_rank_"

# Real DDP watchdog timeout (utils.py default is already 90 min; this line is
# just being explicit). The 2026-08-07/08 debugging window temporarily dropped
# this to 15 min to make a repeat of the wandb-fork hang cheap to observe --
# that hang is now root-caused and fixed (pre-wandb.init() DataLoader fork),
# confirmed by a clean 2.5h run with zero watchdog hits. Reverted to 90 for
# the full-scale run so a legitimate transient stall doesn't get killed early.
export DDP_TIMEOUT_MIN=90

# MERGED-DATA RUN (2026-08-07): all 7 pretrain-mixture datasets (atlas, aspen,
# jetclass, jetclass2, h1, cms_qcd, cms_bsm) are now migrated to $SCRATCH with
# teacher_logits merged directly into each source .h5 (merge_teacher_logits.py),
# so --path points at the scratch merged datasets instead of the CFS source
# (/global/cfs/cdirs/m4567/www/) that the two prior attempts stalled/crashed
# against (Jun 15 epoch 30/500, Jul 2-3 epoch 2/500, and this same tag's
# relaunch0805 attempt also stalled at epoch 2/10 against CFS on Aug 5). This
# removes the CFS/Lustre file-locking-stall hypothesis for the ~7,000-file
# epoch-0 cold-start burst.
#
# --teacher-labels-dir/--teacher-tag are still required by train.py's
# `--distill requires both` check, but every source file under DATA_PATH now
# carries its own `teacher_logits` dataset (see merge_teacher_logits.py), so
# HEPDataset reads it straight from the source file handle and never opens a
# separate companion file -- teacher-labels-dir is set to the SAME merged dir
# so nothing points at the old (now-unused) companion tree.
DATA_PATH=/pscratch/sd/t/twamorka/omnilearned/datasets/

# New tag (2026-08-09): distill_pretrain_s_scratch_a05_b05_T4_full500 -- a
# deliberate FRESH START at epoch 0, not a continuation of the
# _merged tag's epoch-40 checkpoint. User's choice: keep the epoch-40
# checkpoint under the old tag untouched/unused, start this real 500-epoch
# run clean under its own name. --resuming stays on (harmless no-op until
# this tag has its own checkpoint to resume from on future loop resubmits --
# train.py only resumes if a checkpoint file matching --save-tag exists).
#
# --epoch 500: the real target -- iterations(1000) x batch(128) x 16 GPUs =
# 2,048,000 samples/epoch-unit, and the full 7-dataset pretrain mixture is
# ~1.06B samples, so --epoch 500 ~= 0.97 of one true pass over the full
# mixture. (The 40-epoch validation run under the old _merged tag already
# confirmed the wandb-fork-hang fix holds clean through a full session --
# best val loss 8.08622 @ epoch 39, wandb run 7lld3ae0.)
#
# NOTE: a 48-GPU/-q regular variant (--epoch 167, 12 nodes) was drafted and
# reverted 2026-08-09 -- it briefly overwrote this file while job 56540200
# (this exact 16-GPU/interactive config, launched 11:03) was already live, an
# accident, not intentional. If revisiting 48-GPU scaling, use a NEW script
# filename, never edit this one while its own loop might still be running.
cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag distill_pretrain_s_scratch_a05_b05_T4_full500 \
  --dataset pretrain --mode pretrain --num-classes 210 \
  --path $DATA_PATH \
  --size small \
  --use-pid --use-add --use-event-loss --interaction --local-interaction \
  --feature-drop 0.1 \
  --batch 128 --iterations 1000 --epoch 500 \
  --num-workers 4 \
  --distill \
  --teacher-labels-dir $DATA_PATH \
  --teacher-tag pretrain_l \
  --distill-alpha 0.5 --distill-beta 0.5 --distill-t 4 \
  --wandb --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
