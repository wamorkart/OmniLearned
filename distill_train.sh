#load libs
module load pytorch

# for DDP
export MASTER_ADDR=$(hostname)

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
  --save-tag distill_pretrain_s_scratch_a05_T4 \
  --dataset pretrain --mode pretrain --num-classes 210 \
  --path /global/cfs/cdirs/m4567/www/ \
  --size small \
  --use-pid --use-add --use-event-loss --interaction \
  --feature-drop 0.1 \
  --batch 128 --iterations 1000 --epoch 500 \
  --num-workers 32 \
  --distill \
  --teacher-labels-dir $TEACHER_DIR \
  --teacher-tag pretrain_l \
  --distill-alpha 0.5 --distill-beta 0.5 --distill-t 4 \
  --wandb"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
