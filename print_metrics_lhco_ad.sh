#!/bin/bash
# Prints metrics after evaluate_lhco_ad.sh has been run. First step is
# concatenating all rank npz files produced by evaluate.
#
# background_class=0 (the default): in lhco_ad, pid=0 is the pure background
# stand-in and pid=1 is "data" (potentially signal-contaminated), matching
# classify_lhco.py's convention (combine() labels appended data 1).

module load conda
conda activate ol_distill

# ============================================================
#  EDIT THESE PER RUN -- must match the evaluate_lhco_ad.sh run
# ============================================================
SAVE_TAG_BASE=fine_tune_pretrain_s
DATASET=lhco_ad
NSIG=all
DESCRIPT_TAG=i300_e10
QUANTIZATION=none
DATASET_TYPE=test
# ============================================================

SAVE_TAG="${SAVE_TAG_BASE}_${DATASET}_nsig${NSIG}_${DESCRIPT_TAG}"
DIR="/pscratch/sd/m/mbenyas/${SAVE_TAG}_${QUANTIZATION}"

OUT_FILE="$DIR/metrics_${SAVE_TAG}_${DATASET_TYPE}_$(date +%Y%m%d_%H%M%S).txt"

python - <<PY | tee "$OUT_FILE"
import glob
import numpy as np
from omnilearned.utils import print_metrics

pattern = f"outputs_${SAVE_TAG}_${DATASET}_${DATASET_TYPE}_*.npz"
files = sorted(glob.glob(f"${DIR}/{pattern}"))
print(f"Found {len(files)} rank file(s) matching {pattern}")

predictions = np.concatenate([np.load(f)["prediction"] for f in files])
labels = np.concatenate([np.load(f)["pid"] for f in files])
print(f"Total ${DATASET_TYPE} examples: {len(labels)}\n")

print_metrics(predictions, labels, background_class=0)
PY

echo "Saved metrics to $OUT_FILE"
