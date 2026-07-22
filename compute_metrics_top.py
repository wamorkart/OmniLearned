"""Compute AUC, accuracy, and background rejection at fixed signal efficiencies for top tagging.

Binary classification: label 0 = QCD (background), label 1 = top (signal).

Metrics:
  - AUC
  - Accuracy
  - 1/FPR at signal efficiency (TPR) = 50% and 30%

Usage:
    python compute_metrics_top.py --indir /path/to/eval/dir --tag <save_tag>
"""

import argparse
import glob
import os

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

SIGNAL_CLASS = 1
SIGNAL_EFFS = [0.50, 0.30]


def load_rank_files(indir, tag):
    pattern = os.path.join(indir, f"outputs_{tag}_top_test_rank*.npz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching: {pattern}")
    print(f"Found {len(paths)} rank files")
    preds, labels = [], []
    for p in paths:
        z = np.load(p)
        preds.append(z["prediction"].astype(np.float32))
        labels.append(z["pid"])
    return np.concatenate(preds), np.concatenate(labels)


def rej_at_eff(fpr, tpr, eff):
    """Return 1/FPR at the first point where TPR >= eff."""
    idx = np.searchsorted(tpr, eff)
    if idx >= len(fpr) or fpr[idx] == 0:
        return float("inf")
    return 1.0 / fpr[idx]


def compute_and_print(preds, labels, tag=""):
    scores = preds[:, SIGNAL_CLASS]
    pred_classes = preds.argmax(axis=1)

    acc = (pred_classes == labels).mean()
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)

    header = f"=== {tag} ===" if tag else "=== Results ==="
    print(f"\n{header}")
    print(f"  N events  : {len(labels):,}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  AUC       : {auc:.4f}")
    for eff in SIGNAL_EFFS:
        rej = rej_at_eff(fpr, tpr, eff)
        print(f"  1/FPR @ {int(eff*100)}% sig eff : {rej:.1f}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Directory with per-rank npz files")
    ap.add_argument("--tag", required=True, help="save_tag used during evaluate")
    args = ap.parse_args()

    preds, labels = load_rank_files(args.indir, args.tag)
    compute_and_print(preds, labels, args.tag)


if __name__ == "__main__":
    main()
