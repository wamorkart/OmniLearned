"""Compute metrics for the resumed no-interaction micro-KD reps (r2, r4) and
build a full comparison table against established benchmarks from this study.

Usage:
    python compare_noint_r2_r4.py
"""

import glob
import os

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

EVAL_DIR = "/pscratch/sd/t/twamorka/omnilearned/eval/top_distill_micro_noint/"
SIGNAL_CLASS = 1
SIGNAL_EFFS = [0.50, 0.30]

NOINT_TAGS = {
    "r2": "distill_top_micro_noint_scratch_a05_T4_r2",
    "r4": "distill_top_micro_noint_scratch_a05_T4_r4",
}

# Established reference numbers already computed earlier in this study.
PRIOR = {
    "fine_tune_top_s (small, CE-only)": dict(acc=94.38, auc=0.9875, r50=577, r30=2556),
    "fine_tune_top_l (large teacher)": dict(acc=94.44, auc=0.9880, r50=645, r30=3365),
    "distill_top_small_scratch_a05_T4 (small KD)": dict(acc=94.32, auc=0.9875, r50=580, r30=2804),
    "distill_top_micro_scratch_a05_T4 (micro KD, with interaction, n=5 mean)": dict(
        acc=94.34, auc=0.9876, r50=605, r30=3021
    ),
    "train_top_micro_ce_scratch (CE-only micro, n=5 mean)": dict(
        acc=94.12, auc=0.9863, r50=442, r30=1639
    ),
}


def load_rank_files(indir, tag):
    pattern = os.path.join(indir, f"outputs_{tag}_top_test_rank*.npz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching: {pattern}")
    preds, labels = [], []
    for p in paths:
        z = np.load(p)
        preds.append(z["prediction"].astype(np.float32))
        labels.append(z["pid"])
    return np.concatenate(preds), np.concatenate(labels)


def rej_at_eff(fpr, tpr, eff):
    idx = np.searchsorted(tpr, eff)
    if idx >= len(fpr) or fpr[idx] == 0:
        return float("inf")
    return 1.0 / fpr[idx]


def compute(tag):
    preds, labels = load_rank_files(EVAL_DIR, tag)
    scores = preds[:, SIGNAL_CLASS]
    pred_classes = preds.argmax(axis=1)
    acc = (pred_classes == labels).mean() * 100
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    r50 = rej_at_eff(fpr, tpr, 0.50)
    r30 = rej_at_eff(fpr, tpr, 0.30)
    return dict(acc=acc, auc=auc, r50=r50, r30=r30, n=len(labels))


def main():
    noint_results = {}
    for label, tag in NOINT_TAGS.items():
        try:
            noint_results[label] = compute(tag)
        except FileNotFoundError as e:
            print(f"WARNING: {e}")

    lines = []
    lines.append("=== No-interaction micro-KD resumed reps (this run) ===")
    lines.append(f"{'Rep':<8}{'N events':>12}{'Acc':>10}{'AUC':>10}{'1/FPR@50%':>12}{'1/FPR@30%':>12}")
    for label, m in noint_results.items():
        lines.append(
            f"{label:<8}{m['n']:>12,}{m['acc']:>9.2f}%{m['auc']:>10.4f}{m['r50']:>12.1f}{m['r30']:>12.1f}"
        )
    if len(noint_results) > 1:
        arr = lambda k: np.array([m[k] for m in noint_results.values()])
        lines.append(
            f"{'mean':<8}{'':>12}{arr('acc').mean():>9.2f}%{arr('auc').mean():>10.4f}"
            f"{arr('r50').mean():>12.1f}{arr('r30').mean():>12.1f}"
        )
        lines.append(
            f"{'std':<8}{'':>12}{arr('acc').std(ddof=1):>9.2f}%{arr('auc').std(ddof=1):>10.4f}"
            f"{arr('r50').std(ddof=1):>12.1f}{arr('r30').std(ddof=1):>12.1f}"
        )

    lines.append("")
    lines.append("=== Full comparison table ===")
    lines.append(f"{'Model':<62}{'Acc':>10}{'AUC':>12}{'1/FPR@50%':>14}{'1/FPR@30%':>14}")
    for name, m in PRIOR.items():
        lines.append(f"{name:<62}{m['acc']:>9.2f}%{m['auc']:>12.4f}{m['r50']:>14.1f}{m['r30']:>14.1f}")
    for label, m in noint_results.items():
        name = f"distill_top_micro_noint_scratch_a05_T4_{label} (no-interaction KD)"
        lines.append(f"{name:<62}{m['acc']:>9.2f}%{m['auc']:>12.4f}{m['r50']:>14.1f}{m['r30']:>14.1f}")

    out = "\n".join(lines)
    print(out)

    results_path = "/pscratch/sd/t/twamorka/omnilearned/results/noint_r2_r4_comparison.txt"
    with open(results_path, "w") as f:
        f.write(out + "\n")
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
