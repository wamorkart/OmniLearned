#!/bin/bash
# Prints metrics after omnilearned evaluate has been run. First step is concatenating
# all rank npz files produced by evaluate.

module load conda
conda activate ol_distill

# ============================================================
#  EDIT THESE TWO LINES PER RUN -- everything else derives from them
# ============================================================
SAVE_TAG=distill_top_small_scratch_a05_T4   # full save-tag as passed to `evaluate --save-tag`
QUANTIZATION=int8                           # "none", "int8", "int8dq", or "bf16" -- must match the evaluate_perlmutter.sh run
# ============================================================

case "$QUANTIZATION" in
    none|int8|int8dq|bf16) ;;
    *)
        echo "ERROR: QUANTIZATION must be 'none', 'int8', 'int8dq', or 'bf16', got '$QUANTIZATION'" >&2
        exit 1
        ;;
esac

DATASET=top
DIR="/pscratch/sd/m/mbenyas/${SAVE_TAG}_${QUANTIZATION}"
DATASET_TYPE=test

# Where to save results -- one file per run, named after the config above,
# so re-running for a different SAVE_TAG doesn't overwrite previous results.
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

print_metrics(predictions, labels)
PY

echo "Saved metrics to $OUT_FILE"