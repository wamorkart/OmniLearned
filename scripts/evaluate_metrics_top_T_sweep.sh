#!/bin/bash
# Evaluate all T-sweep distillation runs on top tagging, then print a metrics table.
#
# Skips runs whose eval outputs already exist (idempotent).
# Runs that have no checkpoint (still training) are also skipped.
#
# Usage — run inside a 1-node interactive salloc:
#   salloc -C gpu -q interactive -t 120 \
#          --nodes 1 --ntasks-per-node 4 --gpus-per-node 4 -A m3246 \
#          bash evaluate_metrics_top_T_sweep.sh

set -euo pipefail

module load conda
conda activate /global/homes/t/twamorka/omnilearned-clean/env
module load pytorch

export MASTER_ADDR=$(hostname)
export NCCL_TIMEOUT=600000
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CHECKPOINT_DIR=/pscratch/sd/t/twamorka/omnilearned/checkpoints
EVAL_DIR=/pscratch/sd/t/twamorka/omnilearned/eval/top_distill
mkdir -p "$EVAL_DIR"

TAGS=(
    distill_top_small_scratch_a00_b10_T1
    distill_top_small_scratch_a00_b10_T2
    distill_top_small_scratch_a00_b10_T4
    distill_top_small_scratch_a00_b10_T8
    distill_top_small_scratch_a00_b10_T16
    distill_top_small_scratch_a00_b10_T4_r1
    distill_top_small_scratch_a00_b10_T4_r2
    distill_top_small_scratch_a00_b10_T0.5
    distill_top_small_scratch_a00_b10_T0.25
    distill_top_small_scratch_a00_b10_T0.125
)

# ── Evaluate ─────────────────────────────────────────────────────────────────

for TAG in "${TAGS[@]}"; do
    CKPT="$CHECKPOINT_DIR/best_model_${TAG}.pt"
    SENTINEL="$EVAL_DIR/outputs_${TAG}_top_test_rank0.npz"

    if [ ! -f "$CKPT" ]; then
        echo "[SKIP] $TAG — checkpoint not found (still training?)"
        continue
    fi
    if [ -f "$SENTINEL" ]; then
        echo "[SKIP] $TAG — eval outputs already exist"
        continue
    fi

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
            --size small \
            --interaction \
            --local-interaction \
            --num-classes 2 \
            --batch 128 \
            --num-workers 4 \
            --dataset-type test
        "
done

# ── Metrics ──────────────────────────────────────────────────────────────────

echo ""
echo "Computing metrics..."
echo ""

python3 - <<'PYEOF'
import glob, os
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

EVAL_DIR = "/pscratch/sd/t/twamorka/omnilearned/eval/top_distill"
SIGNAL_CLASS = 1

T_SWEEP = [
    ("distill_top_small_scratch_a00_b10_T1",   "α=0 β=1  T=1"),
    ("distill_top_small_scratch_a00_b10_T2",   "α=0 β=1  T=2"),
    ("distill_top_small_scratch_a00_b10_T4",   "α=0 β=1  T=4  [α-sweep base]"),
    ("distill_top_small_scratch_a00_b10_T8",   "α=0 β=1  T=8"),
    ("distill_top_small_scratch_a00_b10_T16",  "α=0 β=1  T=16"),
    ("distill_top_small_scratch_a00_b10_T4_r1","α=0 β=1  T=4  r1"),
    ("distill_top_small_scratch_a00_b10_T4_r2","α=0 β=1  T=4  r2"),
    ("distill_top_small_scratch_a00_b10_T0.5",  "α=0 β=1  T=0.5"),
    ("distill_top_small_scratch_a00_b10_T0.25", "α=0 β=1  T=0.25"),
    ("distill_top_small_scratch_a00_b10_T0.125","α=0 β=1  T=0.125"),
]

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

rows = [(label, load_and_compute(tag)) for tag, label in T_SWEEP]

W = 32
print(f"{'Run':<{W}} {'Acc':>7}  {'AUC':>7}  {'1/FPR@50%':>10}  {'1/FPR@30%':>10}")
print("-" * (W + 44))

for label, res in rows:
    if res is None:
        print(f"  {label:<{W-2}} {'(no outputs)':>7}")
    else:
        acc, auc, r50, r30 = res
        print(f"  {label:<{W-2}} {acc*100:>6.2f}%  {auc:>7.4f}  {r50:>10.0f}  {r30:>10.0f}")

# T=4 rep variance (base + r1 + r2)
t4_tags = {
    "distill_top_small_scratch_a00_b10_T4",
    "distill_top_small_scratch_a00_b10_T4_r1",
    "distill_top_small_scratch_a00_b10_T4_r2",
}
t4_rows = [res for tag, (_, res) in zip([x[0] for x in T_SWEEP], rows)
           if tag in t4_tags and res is not None]
if len(t4_rows) > 1:
    aucs = [r[1] for r in t4_rows]
    r30s = [r[3] for r in t4_rows]
    print()
    print(f"  T=4 variance (n={len(t4_rows)}): "
          f"AUC {np.mean(aucs):.4f} ± {np.std(aucs):.4f}   "
          f"1/FPR@30% {np.mean(r30s):.0f} ± {np.std(r30s):.0f}")

print()
print("Reference (from α-sweep, T=4):")
print(f"  {'α=0.25 β=0.75':<{W-2}} {'94.38%':>7}  {'0.9877':>7}  {'603':>10}  {'3205':>10}")
print(f"  {'α=0.50 β=0.50 (n=5 mean)':<{W-2}} {'94.31%':>7}  {'0.9875':>7}  {'596':>10}  {'2953':>10}")
print(f"  {'fine_tune_top_s  (CE-only)':<{W-2}} {'94.38%':>7}  {'0.9875':>7}  {'577':>10}  {'2556':>10}")
print(f"  {'fine_tune_top_l  (CE large)':<{W-2}} {'94.44%':>7}  {'0.9880':>7}  {'645':>10}  {'3365':>10}")
PYEOF
