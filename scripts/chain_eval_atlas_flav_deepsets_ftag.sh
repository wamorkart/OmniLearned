#!/bin/bash
# Full test-split eval + scoring of the plain DeepSets ftag students (KD, then
# its no-KD control), one after the other in one interactive slot.
set -uo pipefail
cd /global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned
unset OMNILEARNED_ENV
L=/pscratch/sd/t/twamorka/omnilearned/logs
E=/pscratch/sd/t/twamorka/omnilearned/eval
PY=/global/homes/t/twamorka/omnilearned-clean/env/bin/python
EV=scripts/eval_atlas_flav_deepsets_interactive.sh

for TAG in distill_atlas_flav_deepsets_distillnet_relu_ftag_a05_T4 \
           train_atlas_flav_deepsets_distillnet_relu_ftag_ce_scratch; do
  OUT=$E/${TAG}_full/
  EVAL_TAG=$TAG OUTPUT_DIR=$OUT NUM_CHUNKS=1 \
    LOOP_LOG_DIR=$L/eval_full_$TAG NODES=1 MAX_LOOPS=3 \
    scripts/lib/resubmit_loop.sh bash $EV || { echo "CHAIN: eval $TAG failed"; continue; }
  $PY tools/metrics/compute_metrics_flav.py --indir $OUT --tag $TAG | tee $OUT/metrics.txt
done
echo "CHAIN: done"
