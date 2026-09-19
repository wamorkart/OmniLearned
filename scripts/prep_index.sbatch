#!/bin/bash
# Build (and cache to $SCRATCH) the file_index for a dataset's splits, so the
# parallel chunk jobs don't race to build it. Pure CPU/IO -- runs on a shared
# CPU node. Chunk jobs depend on this via --dependency=afterok.
#
# Required env: DATASET DATA_PATH
#SBATCH -A m3246
#SBATCH -C cpu
#SBATCH -q shared
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
set -euo pipefail
: "${DATASET:?}"; : "${DATA_PATH:?}"

export PATH=/global/homes/t/twamorka/omnilearned-clean/env/bin:$PATH
echo "=== $(date '+%F %T')  building index for $DATASET (all splits) ==="
# The dataloader CLI builds + saves the index for train/test/val.
omnilearned dataloader -d "$DATASET" -f "$DATA_PATH"
echo "=== done ==="
