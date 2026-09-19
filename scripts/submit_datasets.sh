#!/bin/bash
# Submit teacher-logit evaluation to the REGULAR batch queue for several
# datasets/splits, one sbatch job per chunk. Unlike the interactive
# run_all_chunks.sh (capped at 2 concurrent jobs), the batch queue lets all
# these jobs queue and backfill in parallel.
#
# For each dataset/split it:
#   - finds the event count (cached index if present, else scans h5 metadata),
#   - picks NUM_CHUNKS so each job is ~TARGET_EVENTS (~1h on NODES nodes),
#   - estimates --time from the measured atlas throughput + safety + setup,
#   - submits an index-build prep job first (dependency) when no index exists,
#   - skips chunks whose rank files are already complete.
#
# Usage:
#   bash submit_datasets.sh                  # submit with defaults
#   DRY_RUN=1 bash submit_datasets.sh        # print the plan, submit nothing
#   DATASETS="jetclass h1" SPLITS="train test" bash submit_datasets.sh
#   EVAL_AMP=bf16 bash submit_datasets.sh    # after the bf16 A/B passes
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"
PYBIN=/global/homes/t/twamorka/omnilearned-clean/env/bin/python

# -- Tunables -----------------------------------------------------------------
DATASETS=${DATASETS:-"jetclass jetclass2 h1 cms_qcd cms_bsm aspen"}  # atlas handled interactively
SPLITS=${SPLITS:-"train"}
NODES=${NODES:-4}
GPUS_PER_NODE=4
QOS=${QOS:-regular}
ACCOUNT=${ACCOUNT:-m3246}
SAVE_TAG=${SAVE_TAG:-pretrain_l}
SIZE=${SIZE:-large}
BATCH=${BATCH:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
# fp32 is the safe default; switch to bf16 only after checks/ab_precision.sh
# confirms the logits are equivalent.
EVAL_AMP=${EVAL_AMP:-fp32}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/pscratch/sd/t/twamorka/omnilearned/checkpoints/}
OUTPUT_DIR=${OUTPUT_DIR:-/pscratch/sd/t/twamorka/omnilearned/teacher_logits/TEST/}
DATA_PATH=${DATA_PATH:-/global/cfs/cdirs/m4567/www/}
DRY_RUN=${DRY_RUN:-0}

# Estimation constants (derived from atlas chunk20: 7.42M ev / 16 GPUs in
# ~2599s => ~178 ev/s/GPU at fp32,batch64; bf16 is faster, so fp32 time is a
# safe upper bound either way).
RATE_EV_S_GPU=178
TARGET_EVENTS=10000000          # ~1h/chunk on 4 nodes
SAFETY=1.6                      # margin over the point estimate
SETUP_SEC=1200                  # checkpoint load + cached-index load + DDP init
MAX_MIN=720                     # regular-qos wall cap (12h)
MIN_MIN=30

EXPECTED_RANKS=$(( NODES * GPUS_PER_NODE ))
GPUS=$(( NODES * GPUS_PER_NODE ))
mkdir -p "$OUTPUT_DIR"

count_glob() { local n; n=$(ls $1 2>/dev/null | wc -l) || n=0; echo "$n"; }

# Event count for dataset/split: cached index first, else scan h5 metadata.
count_events() {
    local ds=$1 sp=$2
    "$PYBIN" - "$ds" "$sp" "$DATA_PATH" <<'PY'
import os, sys, glob
import numpy as np
ds, sp, data = sys.argv[1], sys.argv[2], sys.argv[3]
scratch = os.environ.get("SCRATCH", "")
cands = []
if scratch:
    cands.append(os.path.join(scratch, "omnilearned_cache", ds, sp, "file_index.npy"))
cands.append(os.path.join(data, ds, sp, "file_index.npy"))
for c in cands:
    if os.path.isfile(c):
        try:
            print(int(np.load(c, mmap_mode="r").shape[0])); sys.exit(0)
        except Exception:
            pass
# fall back to summing dataset lengths from h5 metadata (fast: shape only)
import h5py
total = 0
files = sorted(glob.glob(os.path.join(data, ds, sp, "*.h5")) +
               glob.glob(os.path.join(data, ds, sp, "*.hdf5")))
for f in files:
    try:
        with h5py.File(f, "r") as h:
            total += len(h["data"])
    except Exception:
        pass
print(total if files else -1)
PY
}

has_cached_index() {
    local ds=$1 sp=$2
    [ -f "$SCRATCH/omnilearned_cache/$ds/$sp/file_index.npy" ] && return 0
    [ -f "$DATA_PATH/$ds/$sp/file_index.npy" ] && return 0
    return 1
}

fmt_time() {  # seconds -> HH:MM:SS, clamped to [MIN_MIN, MAX_MIN]
    local sec=$1 min
    min=$(( (sec + 59) / 60 ))
    (( min < MIN_MIN )) && min=$MIN_MIN
    (( min > MAX_MIN )) && min=$MAX_MIN
    printf "%02d:%02d:00" $(( min / 60 )) $(( min % 60 ))
}

declare -A PREP_JOB   # dataset -> prep jobid (one index build per dataset)

submit_prep() {  # $1=dataset -> echoes jobid (cached in PREP_JOB)
    local ds=$1
    if [ -n "${PREP_JOB[$ds]:-}" ]; then echo "${PREP_JOB[$ds]}"; return; fi
    local jid
    if [ "$DRY_RUN" = 1 ]; then
        jid="DRYPREP_$ds"
    else
        jid=$(sbatch --parsable -J "idx_$ds" -t 00:40:00 \
                -o "$OUTPUT_DIR/prep_${ds}.log" \
                --export=ALL,DATASET="$ds",DATA_PATH="$DATA_PATH" \
                scripts/prep_index.sbatch)
    fi
    PREP_JOB[$ds]=$jid
    echo "$jid"
}

echo "=== Plan (NODES=$NODES, GPUs=$GPUS, qos=$QOS, amp=$EVAL_AMP, batch=$BATCH) ==="
printf "%-10s %-6s %14s %7s %6s %10s  %s\n" DATASET SPLIT EVENTS CHUNKS WALL PENDING NOTE

for ds in $DATASETS; do
    for sp in $SPLITS; do
        total=$(count_events "$ds" "$sp")
        if [ "$total" -le 0 ] 2>/dev/null; then
            printf "%-10s %-6s %14s %7s %6s %10s  %s\n" "$ds" "$sp" "n/a" "-" "-" "-" "no data found, skipped"
            continue
        fi
        nc=$(( (total + TARGET_EVENTS - 1) / TARGET_EVENTS ))
        (( nc < 1 )) && nc=1
        per_chunk=$(( total / nc ))
        est_sec=$(( per_chunk / (GPUS * RATE_EV_S_GPU) ))
        wall_sec=$(awk -v e="$est_sec" -v s="$SAFETY" -v o="$SETUP_SEC" 'BEGIN{printf "%d", e*s + o}')
        wall=$(fmt_time "$wall_sec")

        # which chunks still need running?
        pending=()
        for ((K=0; K<nc; K++)); do
            pat="$OUTPUT_DIR/outputs_${SAVE_TAG}_${ds}_${sp}_chunk${K}of${nc}_rank*.npz"
            [ "$(count_glob "$pat")" -eq "$EXPECTED_RANKS" ] || pending+=("$K")
        done

        note=""
        dep=""
        if ! has_cached_index "$ds" "$sp" && [ "${#pending[@]}" -gt 0 ]; then
            pj=$(submit_prep "$ds")
            dep="--dependency=afterok:$pj"
            note="index prep job $pj"
        fi

        printf "%-10s %-6s %14d %7d %6s %10s  %s\n" \
            "$ds" "$sp" "$total" "$nc" "$wall" "${#pending[@]}/$nc" "$note"

        for K in "${pending[@]}"; do
            export_vars="ALL,DATASET=$ds,DATASET_TYPE=$sp,NUM_CHUNKS=$nc,CHUNK_IDX=$K"
            export_vars+=",OUTPUT_DIR=$OUTPUT_DIR,SAVE_TAG=$SAVE_TAG,CHECKPOINT_DIR=$CHECKPOINT_DIR"
            export_vars+=",DATA_PATH=$DATA_PATH,SIZE=$SIZE,EVAL_AMP=$EVAL_AMP,BATCH=$BATCH,NUM_WORKERS=$NUM_WORKERS"
            if [ "$DRY_RUN" = 1 ]; then
                echo "    DRY: sbatch -J ev_${ds}_${sp}_c${K} --nodes=$NODES -t $wall $dep ..."
            else
                sbatch --parsable -J "ev_${ds}_${sp}_c${K}" \
                    -A "$ACCOUNT" -q "$QOS" --nodes="$NODES" -t "$wall" $dep \
                    -o "$OUTPUT_DIR/${ds}_${sp}_chunk${K}of${nc}.log" \
                    --export="$export_vars" \
                    scripts/evaluate_batch.sbatch >/dev/null \
                    && echo "    submitted chunk $K  (-t $wall ${dep:+$dep})"
            fi
        done
    done
done

echo ""
if [ "$DRY_RUN" = 1 ]; then
    echo "DRY_RUN: nothing submitted. Re-run without DRY_RUN=1 to submit."
else
    echo "Submitted. Watch with:  squeue -u \$USER"
fi
