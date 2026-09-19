#!/bin/bash
# Build per-source teacher-logit companions for ALL pretrain datasets and splits,
# ONE interactive job per (dataset, split): allocate a node, convert that single
# pair, release the node, then the next pair allocates a fresh job.
#
# Run TWO instances in parallel (uses your 2-interactive-job allowance) by giving
# each a disjoint share of the work list. Instance SHARD of NSHARDS owns every
# pair whose global index satisfies  index % NSHARDS == SHARD .
#
# Usage (from a login node -- this is the driver, it submits the salloc jobs):
#   terminal A:  ./run_teacher_h5.sh 0      # owns even-indexed pairs
#   terminal B:  ./run_teacher_h5.sh 1      # owns odd-indexed pairs
# (NSHARDS defaults to 2; pass it as $2, e.g. ./run_teacher_h5.sh 0 1 for a single serial run.)
#
# Optional env overrides: NPZ_DIR OUT_DIR DATA_PATH TAG WALLTIME
set -uo pipefail

SHARD=${1:?usage: $0 <shard-index> [nshards]}
NSHARDS=${2:-2}

REPO=/global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned
NPZ_DIR=${NPZ_DIR:-/pscratch/sd/t/twamorka/omnilearned/teacher_logits/TEST}
OUT_DIR=${OUT_DIR:-/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion}
DATA_PATH=${DATA_PATH:-/global/cfs/cdirs/m4567/www/}
TAG=${TAG:-pretrain_l}
WALLTIME=${WALLTIME:-04:00:00}

SALLOC_OPTS=(-A m3246 -C cpu -q interactive -t "$WALLTIME" --ntasks=1 --cpus-per-task=16)

DATASETS=(atlas aspen jetclass jetclass2 h1 cms_qcd cms_bsm)
SPLITS=(train val test)

# Global work list, split-major so the heavy train splits run first.
WORK=()
for sp in "${SPLITS[@]}"; do
    for ds in "${DATASETS[@]}"; do
        WORK+=("$ds:$sp")
    done
done

echo "=== $(date '+%F %T')  shard $SHARD of $NSHARDS  (tag=$TAG) ==="
echo "    one interactive job per pair; this shard owns:"
for i in "${!WORK[@]}"; do
    (( i % NSHARDS == SHARD )) && echo "        ${WORK[$i]}"
done

n_ok=0; n_skip=0; n_fail=0
for i in "${!WORK[@]}"; do
    (( i % NSHARDS == SHARD )) || continue
    ds=${WORK[$i]%%:*}
    sp=${WORK[$i]##*:}

    # Don't allocate a node for a split with no teacher npz (e.g. test never run).
    if ! compgen -G "$NPZ_DIR/outputs_${TAG}_${ds}_${sp}_*.npz" > /dev/null; then
        echo ">>> $(date '+%F %T')  $ds/$sp: no npz shards, skipping"
        (( n_skip++ ))
        continue
    fi

    echo ">>> $(date '+%F %T')  $ds/$sp: requesting interactive job"
    # salloc holds one interactive job for just this pair, then releases it.
    if salloc "${SALLOC_OPTS[@]}" srun --ntasks=1 --cpus-per-task=16 \
            bash -c "export PATH=/global/homes/t/twamorka/omnilearned-clean/env/bin:\$PATH; \
                python3 '$REPO/tools/preprocess/build_teacher_h5.py' \
                    --npz-dir '$NPZ_DIR' \
                    --tag '$TAG' \
                    --data-path '$DATA_PATH' \
                    --out-dir '$OUT_DIR' \
                    --dataset '$ds' \
                    --split '$sp' \
                    --skip-existing"; then
        echo "<<< $(date '+%F %T')  $ds/$sp: done (job released)"
        (( n_ok++ ))
    else
        echo "!!! $(date '+%F %T')  $ds/$sp: FAILED (continuing)"
        (( n_fail++ ))
    fi
done

echo "=== $(date '+%F %T')  shard $SHARD done: ok=$n_ok skipped=$n_skip failed=$n_fail ==="
(( n_fail == 0 ))
