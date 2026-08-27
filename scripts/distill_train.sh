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
# (confirmed: the Jul 3 crash still hit the 1,800,000ms/30min default despite this
# being set to 600000) -- the watchdog timeout comes from the `timeout=` kwarg to
# init_process_group(), set via DDP_TIMEOUT (90 min) in utils.py:ddp_setup(). That
# widened timeout is the actual fix for rank stalls from lazy CFS h5 companion reads.
# NCCL_DEBUG=WARN prints one-line summaries on collective failures without
# the full INFO flood; bump to INFO if you need the per-op sequence numbers.
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Offline KD: small student distilled FROM SCRATCH against the large (pretrain_l)
# teacher logits stored as lazily-read companion .h5 files.
#   - student init: from scratch (NO --fine-tune / --pretrain-tag) for a clean
#     A/B vs the CE-only baseline best_model_pretrain_s.pt
#   - teacher logits: companion dir, keyed by source-file/sample (--teacher-tag
#     is just a label now; the path is <dir>/<dataset>/<split>/<stem>.h5)
#   - wandb run name == --save-tag, so keep the tag unique per run

TEACHER_DIR=/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion

cmd="omnilearned train \
  -o /pscratch/sd/t/twamorka/omnilearned/checkpoints/ \
  --save-tag distill_pretrain_s_scratch_a05_b05_T4 \
  --dataset pretrain --mode pretrain --num-classes 210 \
  --path /global/cfs/cdirs/m4567/www/ \
  --size small \
  --use-pid --use-add --use-event-loss --interaction --local-interaction \
  --feature-drop 0.1 \
  --batch 128 --iterations 1000 --epoch 500 \
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
