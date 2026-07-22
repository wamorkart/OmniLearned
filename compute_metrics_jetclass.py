"""Compute accuracy, AUC, and 1/FPR at fixed signal efficiencies for JetClass (10-class).

Metrics are computed one-vs-rest (OvR) per class:
  - AUC
  - 1/FPR at signal efficiency (TPR) = 50% and 30%

Usage:
    # from raw rank files (no concat step needed)
    python compute_metrics_jetclass.py --indir /path/to/eval/dir --tag <save_tag>

    # from a pre-concatenated file produced by concat_logits.py
    python compute_metrics_jetclass.py --file /path/to/<tag>_jetclass_test.npz
"""

import argparse
import glob
import os

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

CLASS_NAMES = [
    "QCD", "Hbb", "Hcc", "Hgg", "H4q",
    "Hqql", "Zqq", "Wqq", "Tbqq", "Tbl",
]

SIGNAL_EFFS = [0.50, 0.30]


def load_rank_files(indir, tag):
    pattern = os.path.join(indir, f"outputs_{tag}_jetclass_test_rank*.npz")
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


def load_concat_file(path):
    z = np.load(path)
    return z["prediction"].astype(np.float32), z["pid"]


def rej_at_eff(fpr, tpr, eff):
    """Return 1/FPR at the first point where TPR >= eff."""
    idx = np.searchsorted(tpr, eff)
    if idx >= len(fpr) or fpr[idx] == 0:
        return float("inf")
    return 1.0 / fpr[idx]


def compute_and_print(preds, labels, tag=""):
    n = len(labels)
    pred_classes = preds.argmax(axis=1)
    acc = (pred_classes == labels).mean()

    auc_macro = roc_auc_score(labels, preds, multi_class="ovr", average="macro")
    auc_per_class = roc_auc_score(labels, preds, multi_class="ovr", average=None)

    # per-class OvR ROC -> 1/FPR at each signal efficiency
    rej = {eff: [] for eff in SIGNAL_EFFS}
    for i in range(len(CLASS_NAMES)):
        binary = (labels == i).astype(int)
        fpr, tpr, _ = roc_curve(binary, preds[:, i])
        for eff in SIGNAL_EFFS:
            rej[eff].append(rej_at_eff(fpr, tpr, eff))

    header = f"=== {tag} ===" if tag else "=== Results ==="
    print(f"\n{header}")
    print(f"  N events  : {n:,}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  AUC macro : {auc_macro:.4f}")
    print()

    eff_headers = "".join(f"  {'1/FPR@'+str(int(e*100))+'%':>12}" for e in SIGNAL_EFFS)
    print(f"  {'Class':<8}  {'AUC':>6}{eff_headers}")
    print(f"  {'-'*8}  {'-'*6}" + "".join(f"  {'-'*12}" for _ in SIGNAL_EFFS))
    for i, (name, a) in enumerate(zip(CLASS_NAMES, auc_per_class)):
        rej_cols = "".join(f"  {rej[e][i]:>12.1f}" for e in SIGNAL_EFFS)
        print(f"  {name:<8}  {a:.4f}{rej_cols}")

    print()
    macro_rej = "".join(
        f"  {np.mean([r for r in rej[e] if np.isfinite(r)]):>12.1f}" for e in SIGNAL_EFFS
    )
    print(f"  {'macro':<8}  {auc_macro:.4f}{macro_rej}")


def main():
    ap = argparse.ArgumentParser()
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--indir", help="Directory with per-rank npz files")
    group.add_argument("--file", help="Pre-concatenated npz file")
    ap.add_argument("--tag", default="", help="save_tag (required with --indir)")
    args = ap.parse_args()

    if args.indir:
        if not args.tag:
            ap.error("--tag is required when using --indir")
        preds, labels = load_rank_files(args.indir, args.tag)
        tag = args.tag
    else:
        preds, labels = load_concat_file(args.file)
        tag = os.path.basename(args.file)

    compute_and_print(preds, labels, tag)


if __name__ == "__main__":
    main()
