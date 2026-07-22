"""Compare CE-only micro-student reps against the KD micro-student reps and
prior small/teacher benchmarks on top tagging. Reuses the same metric
definitions as compute_metrics_top.py (AUC, accuracy, 1/FPR@50%/30%).

Usage:
    python compare_micro_ce_vs_kd.py
"""

import glob
import os

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

EVAL_DIR_CE = "/pscratch/sd/t/twamorka/omnilearned/eval/top_micro_ce/"
SIGNAL_CLASS = 1
SIGNAL_EFFS = [0.50, 0.30]

CE_TAGS = [f"train_top_micro_ce_scratch_r{i}" for i in range(1, 6)]

# Fixed reference numbers already computed earlier in this study.
KD_MICRO_REPS = {
    "orig": dict(acc=94.35, auc=0.9876, r50=601, r30=2844),
    "r1": dict(acc=94.35, auc=0.9876, r50=601, r30=3365),
    "r2": dict(acc=94.32, auc=0.9875, r50=601, r30=2729),
    "r3": dict(acc=94.34, auc=0.9875, r50=618, r30=3014),
    "r4": dict(acc=94.35, auc=0.9876, r50=603, r30=3155),
}
PRIOR = {
    "fine_tune_top_s (small, CE-only)": dict(acc=94.38, auc=0.9875, r50=577, r30=2556),
    "fine_tune_top_l (large teacher)": dict(acc=94.44, auc=0.9880, r50=645, r30=3365),
    "distill_top_small_scratch_a05_T4 (small KD)": dict(acc=94.32, auc=0.9875, r50=580, r30=2804),
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
    preds, labels = load_rank_files(EVAL_DIR_CE, tag)
    scores = preds[:, SIGNAL_CLASS]
    pred_classes = preds.argmax(axis=1)
    acc = (pred_classes == labels).mean() * 100
    auc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    r50 = rej_at_eff(fpr, tpr, 0.50)
    r30 = rej_at_eff(fpr, tpr, 0.30)
    return dict(acc=acc, auc=auc, r50=r50, r30=r30)


def main():
    ce_results = {}
    for tag in CE_TAGS:
        try:
            ce_results[tag] = compute(tag)
        except FileNotFoundError as e:
            print(f"WARNING: {e}")

    print("\n=== CE-only micro reps (this run) ===")
    print(f"{'Rep':<8}{'Acc':>8}{'AUC':>10}{'1/FPR@50%':>12}{'1/FPR@30%':>12}")
    for tag, m in ce_results.items():
        rep = tag.split("_")[-1]
        print(f"{rep:<8}{m['acc']:>7.2f}%{m['auc']:>10.4f}{m['r50']:>12.1f}{m['r30']:>12.1f}")

    if ce_results:
        arr = lambda k: np.array([m[k] for m in ce_results.values()])
        acc_m, acc_s = arr("acc").mean(), arr("acc").std(ddof=1) if len(ce_results) > 1 else 0.0
        auc_m, auc_s = arr("auc").mean(), arr("auc").std(ddof=1) if len(ce_results) > 1 else 0.0
        r50_m, r50_s = arr("r50").mean(), arr("r50").std(ddof=1) if len(ce_results) > 1 else 0.0
        r30_m, r30_s = arr("r30").mean(), arr("r30").std(ddof=1) if len(ce_results) > 1 else 0.0
        print(f"{'mean':<8}{acc_m:>7.2f}%{auc_m:>10.4f}{r50_m:>12.1f}{r30_m:>12.1f}")
        print(f"{'std':<8}{acc_s:>7.2f}%{auc_s:>10.4f}{r50_s:>12.1f}{r30_s:>12.1f}")

    kd_arr = lambda k: np.array([m[k] for m in KD_MICRO_REPS.values()])

    print("\n=== Full comparison (mean +/- std where applicable) ===")
    print(f"{'Model':<48}{'Acc':>10}{'AUC':>12}{'1/FPR@50%':>14}{'1/FPR@30%':>14}")
    for name, m in PRIOR.items():
        print(f"{name:<48}{m['acc']:>9.2f}%{m['auc']:>12.4f}{m['r50']:>14.1f}{m['r30']:>14.1f}")
    print(
        f"{'distill_top_micro_scratch_a05_T4 (micro KD, n=5)':<48}"
        f"{kd_arr('acc').mean():>8.2f}% ({kd_arr('acc').std(ddof=1):.2f})"
        f"{kd_arr('auc').mean():>8.4f} ({kd_arr('auc').std(ddof=1):.4f})"
        f"{kd_arr('r50').mean():>8.0f} ({kd_arr('r50').std(ddof=1):.0f})"
        f"{kd_arr('r30').mean():>8.0f} ({kd_arr('r30').std(ddof=1):.0f})"
    )
    if ce_results:
        print(
            f"{'train_top_micro_ce_scratch (CE-only micro, n=' + str(len(ce_results)) + ')':<48}"
            f"{acc_m:>8.2f}% ({acc_s:.2f})"
            f"{auc_m:>8.4f} ({auc_s:.4f})"
            f"{r50_m:>8.0f} ({r50_s:.0f})"
            f"{r30_m:>8.0f} ({r30_s:.0f})"
        )


if __name__ == "__main__":
    main()
