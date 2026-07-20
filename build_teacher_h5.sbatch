#!/bin/bash
# Convert sharded teacher-logit .npz -> per-source-h5 companions for ONE
# dataset/split. Pure CPU/IO (streams one npz at a time), so it runs on a
# shared CPU node like prep_index.sbatch. Train splits read hundreds of GB of
# npz, hence batch rather than a 4h interactive slot.
#
# Required env (via sbatch --export): DATASET SPLIT
# Optional: NPZ_DIR OUT_DIR DATA_PATH TAG
#SBATCH -A m3246
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
set -euo pipefail
: "${DATASET:=qg}"; : "${SPLIT:=train,val}"

NPZ_DIR=${NPZ_DIR:-/pscratch/sd/m/mbenyas}
OUT_DIR=${OUT_DIR:-/pscratch/sd/m/mbenyas}
DATA_PATH=${DATA_PATH:-/global/cfs/cdirs/m4567/www/}
TAG=${TAG:-fine_tune_qg_pretrain_l}

# export PATH=/global/homes/t/twamorka/omnilearned-clean/env/bin:$PATH
module load conda
conda activate ol_distill

REPO=/global/cfs/cdirs/m3246/mbenyas/OmniLearned_distillation
echo "=== $(date '+%F %T')  build_teacher_h5 $DATASET/$SPLIT (tag=$TAG) ==="
python3 "$REPO/build_teacher_h5.py" \
    --npz-dir "$NPZ_DIR" \
    --tag "$TAG" \
    --data-path "$DATA_PATH" \
    --out-dir "$OUT_DIR" \
    --dataset "$DATASET" \
    --split "$SPLIT"
echo "=== done $(date '+%F %T') ==="
