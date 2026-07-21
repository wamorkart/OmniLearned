#!/bin/bash
# Prints metrics after omnilearned evaluate has been run. First step is concatenating
# all 16 npz files which were created after running a salloc session.

module load conda
conda activate ol_distill
module load pytorch

# Edit these for whichever run you want to evaluate
DIR=/pscratch/sd/m/mbenyas
SAVE_TAG=fine_tune_qg_pretrain_l #distill_qg_small_scratch_a05_T4
DATASET=qg
DATASET_TYPE=test
 
python - <<PY
import glob
import numpy as np
from omnilearned.utils import print_metrics
 
pattern = f"outputs_${SAVE_TAG}_${DATASET}_${DATASET_TYPE}_*.npz"
files = sorted(glob.glob(f"${DIR}/{pattern}"))
print(f"Found {len(files)} rank file(s) matching {pattern}")
 
predictions = np.concatenate([np.load(f)["prediction"] for f in files])
labels = np.concatenate([np.load(f)["pid"] for f in files])
print(f"Total ${DATASET_TYPE} examples: {len(labels)}\n")
 
print_metrics(predictions, labels)
PY