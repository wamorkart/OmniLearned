#!/bin/bash
# DeepSets width-scan "tiny" point (base_dim=64, phi=3, rho=2, ~75,719 params)
# on the REGULAR queue -- runs in parallel with the interactive
# size_scan_deepsets_queue.sh driver (which is restricted to "nano micro" via
# SCAN_WIDTHS while this job owns "tiny"). Reuses the regular-queue budget the
# dropped atlas_flav localint_v2 chain was using.
#
# Same a05_T4 KD recipe as every other width -- assembled by run_train.sh from
# configs/train/top_deepsets_tiny.sh (+ _defaults.sh): KD alpha=beta=0.5 / T=4,
# teacher fine_tune_top_l, wd 0.5, batch 128, 1000 it x 50 ep, lr 5e-4, --wandb.
# --resuming, so a resubmitted chain link picks up from the last checkpoint.
#
# On a clean exit it evaluates itself on the test split and scores it, appending
# to the same results log the interactive driver writes to.
#
# Submit:  sbatch scripts/train_deepsets_tiny_reg.sbatch
#
#SBATCH -A m3246
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH -J train_deepsets_tiny_reg
#SBATCH -o /pscratch/sd/t/twamorka/omnilearned/logs/size_scan_deepsets/tiny_reg_slurm_%j.out

set -uo pipefail

SCRIPT_DIR=/global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned/scripts
CONFIG=top_deepsets_tiny
TAG=distill_top_deepsets_tiny_scratch_a05_T4
SIZE=tiny
CHAIN_REMAINING=${CHAIN_REMAINING:-3}

LOG_DIR=/pscratch/sd/t/twamorka/omnilearned/logs/size_scan_deepsets
EVAL_DIR=/pscratch/sd/t/twamorka/omnilearned/eval/top_distill_deepsets
RESULTS="$LOG_DIR/results.log"
PY=/global/homes/t/twamorka/omnilearned-clean/env/bin/python
mkdir -p "$LOG_DIR"

export PATH=/global/homes/t/twamorka/omnilearned-clean/env/bin:$PATH
export HDF5_USE_FILE_LOCKING=FALSE
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

echo "[$(date '+%F %T')] tiny_reg job ${SLURM_JOB_ID}, chain remaining (incl this) ${CHAIN_REMAINING}" \
    | tee -a "$LOG_DIR/tiny_reg_summary.log"

cd "$(dirname "$SCRIPT_DIR")"
set +e
bash "$SCRIPT_DIR/run_train.sh" "$CONFIG"
EXIT=$?
set -e

echo "[$(date '+%F %T')] tiny_reg job ${SLURM_JOB_ID} run_train exited ${EXIT}" \
    | tee -a "$LOG_DIR/tiny_reg_summary.log"

if [ "$EXIT" -ne 0 ]; then
    if [ "$CHAIN_REMAINING" -gt 1 ]; then
        NEXT=$((CHAIN_REMAINING - 1))
        echo "Non-zero exit, resubmitting (chain remaining after this: ${NEXT})..." \
            | tee -a "$LOG_DIR/tiny_reg_summary.log"
        sbatch --export=ALL,CHAIN_REMAINING="$NEXT" "$0"
    else
        echo "Chain exhausted without a clean exit -- check logs." \
            | tee -a "$LOG_DIR/tiny_reg_summary.log"
    fi
    exit "$EXIT"
fi

echo "Training completed. Evaluating tiny on the test split." \
    | tee -a "$LOG_DIR/tiny_reg_summary.log"

SAVE_TAG="$TAG" SIZE="$SIZE" DATASET_TYPE=test \
    bash "$SCRIPT_DIR/evaluate_top_distill_deepsets.sh" \
    2>&1 | tee -a "$LOG_DIR/tiny_reg_eval.out"

npz=$(ls "$EVAL_DIR"/outputs_"${TAG}"_top_test_rank*.npz 2>/dev/null | wc -l)
echo "[$(date '+%F %T')] found $npz per-rank npz for $TAG" \
    | tee -a "$LOG_DIR/tiny_reg_summary.log"
if [ "$npz" -gt 0 ]; then
    {
        echo
        echo "==== width=tiny  ($TAG)  regular-queue job ${SLURM_JOB_ID}  $(date '+%F %T') ===="
    } | tee -a "$RESULTS"
    "$PY" tools/metrics/compute_metrics_top.py --indir "$EVAL_DIR" --tag "$TAG" \
        2>&1 | tee -a "$RESULTS"
fi
