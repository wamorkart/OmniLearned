#!/bin/bash
# KD-student full eval -> no-KD baseline training -> baseline full eval.
set -uo pipefail
cd /global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned
unset OMNILEARNED_ENV
L=/pscratch/sd/t/twamorka/omnilearned/logs
E=/pscratch/sd/t/twamorka/omnilearned/eval
EV=scripts/eval_distill_atlas_flav_small_a05_T4_interactive.sh

NUM_CHUNKS=1 EVAL_TAG=distill_atlas_flav_small_a05_T4 OUTPUT_DIR=$E/distill_atlas_flav_small_a05_T4_full/ \
  LOOP_LOG_DIR=$L/eval_full_distill_atlas_flav_small_a05_T4 NODES=1 MAX_LOOPS=3 \
  scripts/lib/resubmit_loop.sh bash $EV || echo "CHAIN: KD eval failed"

LOOP_LOG_DIR=$L/train_loop_atlas_flav_small_ce MAX_LOOPS=8 \
  scripts/lib/resubmit_loop.sh bash scripts/run_train.sh atlas_flav_small_ce || { echo "CHAIN: training failed"; exit 1; }

NUM_CHUNKS=1 EVAL_TAG=train_atlas_flav_small_ce_scratch OUTPUT_DIR=$E/train_atlas_flav_small_ce_scratch_full/ \
  LOOP_LOG_DIR=$L/eval_full_train_atlas_flav_small_ce_scratch NODES=1 MAX_LOOPS=3 \
  scripts/lib/resubmit_loop.sh bash $EV || echo "CHAIN: baseline eval failed"
echo "CHAIN: done"
