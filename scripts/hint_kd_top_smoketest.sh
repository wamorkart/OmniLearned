#!/bin/bash
# One-shot interactive launcher for hint_kd_top_smoketest.py -- see that
# file's docstring for what/why. Single-process (no srun/DDP needed for a
# 100-iteration smoke test), run inside an salloc allocation so it's not on
# the shared login node.
module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch
export MASTER_ADDR=$(hostname)
cd /global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned
python hint_kd_top_smoketest.py
