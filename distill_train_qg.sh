#load libs
module load conda
conda activate ol_distill
module load pytorch

# for DDP
export MASTER_ADDR=$(hostname)

export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Teacher: best_model_fine_tune_qg_l.pt (large model fine-tuned on quark gluon)
# Student: small model trained from scratch on quark gluon
# Requires companion H5 logits pre-generated from fine_tune_qg_l via evaluate + run_teacher_h5.sh
TEACHER_DIR=/pscratch/sd/m/mbenyas/

cmd="omnilearned train \
  -o /pscratch/sd/m/mbenyas/ \
  --save-tag distill_qg_small_scratch_a05_T4 \
  --dataset qg --mode classifier --num-classes 2 \
  --path /global/cfs/cdirs/m4567/www/ \
  --size small \
  --use-pid \
  --interaction \
  --local-interaction \
  --batch 128 --iterations 1000 --epoch 50 \
  --lr 5e-4 --wd 0.5 \
  --num-workers 4 \
  --distill \
  --teacher-labels-dir $TEACHER_DIR \
  --teacher-tag fine_tune_qg_pretrain_l \
  --distill-alpha 0.5 --distill-beta 0.5 --distill-t 4 \
  --wandb --resuming"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
