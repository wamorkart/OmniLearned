#!/bin/bash
# Evaluate top_l_scratch (large model trained from scratch on top tagging, no
# JetClass pretraining) on the top/test split, then print accuracy/AUC/rejection
# metrics — comparison arm for fine_tune_top_l (94.44% acc / 0.9880 AUC).
#
# Run inside an salloc GPU interactive job:
#   salloc -C gpu -q interactive -t 60 --nodes 1 --ntasks-per-node 4 \
#          --gpus-per-node 4 -A m3246 bash scripts/evaluate_top_l_scratch.sh

set -euo pipefail

module load conda
conda activate "${OMNILEARNED_ENV:-/global/homes/t/twamorka/omnilearned-clean/env}"
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints
EVAL_DIR=/pscratch/sd/t/twamorka/omnilearned/eval/top_l_scratch
TAG=top_l_scratch
mkdir -p "$EVAL_DIR"

CKPT="$CHECKPOINT_DIR/best_model_${TAG}.pt"
SENTINEL="$EVAL_DIR/outputs_${TAG}_top_test_rank0.npz"

if [ ! -f "$CKPT" ]; then
    echo "evaluate_top_l_scratch.sh: no checkpoint at $CKPT" >&2
    exit 1
fi

if [ -f "$SENTINEL" ]; then
    echo "[SKIP] $TAG — eval outputs already exist"
else
    echo ""
    echo "=== Evaluating $TAG ==="
    srun -l -u \
        bash -c "
        source '$SCRIPT_DIR/export_ddp.sh'
        omnilearned evaluate \
            -i '$CHECKPOINT_DIR' \
            -o '$EVAL_DIR' \
            --save-tag '$TAG' \
            --dataset top \
            --path /global/cfs/cdirs/m4567/www/ \
            --size large \
            --interaction \
            --local-interaction \
            --num-classes 2 \
            --batch 64 \
            --num-workers 4 \
            --dataset-type test
        "
fi

echo ""
echo "Computing metrics..."
echo ""

python3 - "$EVAL_DIR" "$TAG" <<'PYEOF'
import sys, glob, os
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

EVAL_DIR, TAG = sys.argv[1], sys.argv[2]
SIGNAL_CLASS = 1

def rej_at_eff(fpr, tpr, eff):
    idx = np.searchsorted(tpr, eff)
    if idx >= len(fpr) or fpr[idx] == 0:
        return float("inf")
    return 1.0 / fpr[idx]

def load_and_compute(tag):
    pattern = os.path.join(EVAL_DIR, f"outputs_{tag}_top_test_rank*.npz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        return None
    preds, labels = [], []
    for p in paths:
        z = np.load(p)
        preds.append(z["prediction"].astype(np.float32))
        labels.append(z["pid"])
    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    scores = preds[:, SIGNAL_CLASS]
    acc = (preds.argmax(1) == labels).mean()
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    return acc, auc, rej_at_eff(fpr, tpr, 0.50), rej_at_eff(fpr, tpr, 0.30)

W = 34
print(f"{'Run':<{W}} {'Acc':>7}  {'AUC':>7}  {'1/FPR@50%':>10}  {'1/FPR@30%':>10}")
print("-" * (W + 44))

res = load_and_compute(TAG)
if res is None:
    print(f"  {TAG:<{W-2}} {'(no outputs)':>7}")
else:
    acc, auc, r50, r30 = res
    print(f"  {TAG:<{W-2}} {acc*100:>6.2f}%  {auc:>7.4f}  {r50:>10.0f}  {r30:>10.0f}")

print()
print("Reference (from prior eval):")
print(f"  {'fine_tune_top_s  (CE-only small)':<{W-2}} {'94.38%':>7}  {'0.9875':>7}  {'577':>10}  {'2556':>10}")
print(f"  {'fine_tune_top_l  (CE-only large, JetClass-pretrained)':<{W-2}} {'94.44%':>7}  {'0.9880':>7}  {'645':>10}  {'3365':>10}")
PYEOF
