#!/bin/bash
# A/B test: run the real evaluate pipeline on a small slice of atlas/train
# twice -- once fp32, once bf16 autocast -- on a single GPU, then compare the
# logits. Validates that bf16 produces equivalent distillation targets and
# shows the forward-time speedup, before committing to all datasets.
#
# Run from project root on a login node:
#   bash checks/ab_precision.sh
set -euo pipefail

PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT"

CHECKPOINT_DIR=${CHECKPOINT_DIR:-/pscratch/sd/t/twamorka/omnilearned/checkpoints/}
# Big NUM_CHUNKS -> tiny slice (atlas/train ~156M events / 8000 ~= 19.5k events).
NUM_CHUNKS_AB=${NUM_CHUNKS_AB:-8000}
# atlas events have high particle multiplicity; the interaction matmul is the
# memory limiter, so batch must stay modest (256 OOMs a 40GB A100). 64 matches
# the validated production batch.
BATCH=${BATCH:-64}

# Inner script run inside the allocation on one GPU. No srun / no export_ddp.sh
# and MASTER_ADDR unset => evaluate's ddp_setup takes the single-process path.
read -r -d '' INNER <<'INNER_EOF' || true
set -euo pipefail
# The project's conda env is self-contained (own python3.11 + torch cu130 +
# omnilearned). Putting its bin on PATH is enough; the omnilearned console
# script's shebang pins the right interpreter. No module load / conda activate.
export PATH=/global/homes/t/twamorka/omnilearned-clean/env/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset MASTER_ADDR || true

run_eval() {  # $1=EVAL_AMP  $2=outdir
    local amp=$1 out=$2
    rm -rf "$out"; mkdir -p "$out"
    echo "############ EVAL_AMP=$amp -> $out ############"
    EVAL_AMP="$amp" omnilearned evaluate \
        -i "$CHECKPOINT_DIR" -o "$out" \
        --save-tag pretrain_l --dataset atlas \
        --path /global/cfs/cdirs/m4567/www/ \
        --size large \
        --interaction --local-interaction --use-pid --use-add --use-event-loss \
        --num-classes 210 \
        --batch "$BATCH" --num-workers 8 \
        --dataset-type train \
        --num-chunks "$NUM_CHUNKS_AB" --chunk-idx 0
}

run_eval fp32 checks/ab_fp32
run_eval bf16 checks/ab_bf16
echo "############ COMPARE ############"
python3 checks/ab_compare.py checks/ab_fp32 checks/ab_bf16
INNER_EOF

salloc -C gpu -q interactive -t 30 --nodes 1 --ntasks-per-node=1 --gpus-per-node=1 -A m3246 \
    bash -c "cd '$PROJECT' && export CHECKPOINT_DIR='$CHECKPOINT_DIR' NUM_CHUNKS_AB='$NUM_CHUNKS_AB' BATCH='$BATCH' && $INNER"
