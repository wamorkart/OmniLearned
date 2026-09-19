#!/bin/bash
# Submit the logit-standardization A/B pair (see
# distill_train_pretrain_logitstd_ab.sbatch header). Same SEED both arms so the
# only difference is --distill-standardize.
#
# Add reps by re-running with a different SEED, e.g.:
#   SEED=777 bash distill_launch_pretrain_logitstd_ab.sh
set -euo pipefail
cd "$(dirname "$0")"

SEED=${SEED:-1234}

echo "submitting A/B pair, seed ${SEED}"
sbatch --export=ALL,STANDARDIZE=0,SEED="${SEED}" distill_train_pretrain_logitstd_ab.sbatch
sbatch --export=ALL,STANDARDIZE=1,SEED="${SEED}" distill_train_pretrain_logitstd_ab.sbatch
echo "check: squeue --me ; wandb project OmniBoone runs distill_pretrain_s_a05_T4_seed${SEED}_{nostd,std}"
