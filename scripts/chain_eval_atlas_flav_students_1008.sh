#!/bin/bash
# Full test-split eval + scoring, one after another in one interactive slot, of:
#   - the PET2-small KD rerun (distill_atlas_flav_small_a05_T4_r2)
#   - the DeepSets+GNN(K=64) ftag students (KD, then its no-KD control)
set -uo pipefail
cd /global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned
unset OMNILEARNED_ENV
L=/pscratch/sd/t/twamorka/omnilearned/logs
E=/pscratch/sd/t/twamorka/omnilearned/eval
PY=/global/homes/t/twamorka/omnilearned-clean/env/bin/python

run_one() {  # <tag> <eval script> [extra eval flags]
  local TAG=$1 EV=$2 OUT=$E/${1}_full/
  EVAL_TAG=$TAG OUTPUT_DIR=$OUT NUM_CHUNKS=1 EXTRA_EVAL_FLAGS="${3:-}" \
    LOOP_LOG_DIR=$L/eval_full_$TAG NODES=1 MAX_LOOPS=3 \
    scripts/lib/resubmit_loop.sh bash $EV || { echo "CHAIN: eval $TAG failed"; return; }
  $PY tools/metrics/compute_metrics_flav.py --indir $OUT --tag $TAG | tee $OUT/metrics.txt
}

run_one distill_atlas_flav_small_a05_T4_r2 scripts/eval_distill_atlas_flav_small_a05_T4_interactive.sh
GNN="--num-interaction-layers 1 --interaction-k 64"
run_one distill_atlas_flav_deepsets_distillnet_relu_ftag_a05_T4_gnn1_k64 scripts/eval_atlas_flav_deepsets_interactive.sh "$GNN"
run_one train_atlas_flav_deepsets_distillnet_relu_ftag_gnn1_k64_ce_scratch scripts/eval_atlas_flav_deepsets_interactive.sh "$GNN"
echo "CHAIN: done"
