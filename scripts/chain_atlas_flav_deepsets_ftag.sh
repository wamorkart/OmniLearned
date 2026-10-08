#!/bin/bash
# Plain DeepSets ftag students on atlas_flav: KD run, then its no-KD control,
# one after the other (one interactive slot).
set -uo pipefail
cd /global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned
unset OMNILEARNED_ENV
L=/pscratch/sd/t/twamorka/omnilearned/logs

LOOP_LOG_DIR=$L/distill_loop_atlas_flav_deepsets_distillnet_relu_ftag MAX_LOOPS=8 \
  scripts/lib/resubmit_loop.sh bash scripts/run_train.sh atlas_flav_deepsets_distillnet_relu_ftag \
  || echo "CHAIN: KD training failed"

LOOP_LOG_DIR=$L/train_loop_atlas_flav_deepsets_distillnet_relu_ftag_ce MAX_LOOPS=8 \
  scripts/lib/resubmit_loop.sh bash scripts/run_train.sh atlas_flav_deepsets_distillnet_relu_ftag_ce \
  || echo "CHAIN: CE training failed"
echo "CHAIN: done"
