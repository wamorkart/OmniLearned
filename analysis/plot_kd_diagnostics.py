"""Diagnostics comparing pure KD (alpha=0, beta=1) vs mixed KD (alpha=0.5, beta=0.5)
on top-tagging, to understand why pure KD wins the ablation.

Loads four models evaluated on the identical, identically-ordered 404k test set:
  - teacher (fine_tune_top_l)
  - CE-only baseline student (fine_tune_top_s)
  - mixed-KD student (distill_top_small_scratch_a05_T4)
  - pure-KD student (distill_top_small_scratch_a00_b10_T4)

Produces a 4-panel PDF:
  A. Overlaid P(top) output distributions for all four models.
  B. Reliability (calibration) diagram: predicted P(top) vs empirical accuracy.
  C. |student - teacher| per-event disagreement, mixed vs pure KD.
  D. Student output distributions restricted to events the teacher gets WRONG
     (teacher argmax != true label) -- does each student follow the teacher's
     mistake or the true label on exactly the events where they conflict?

Usage:
    python plot_kd_diagnostics.py --outfile /path/to/kd_diagnostics.pdf
"""

import argparse
import glob
import re

import numpy as np
import matplotlib.pyplot as plt

SIGNAL_CLASS = 1
ROOT = "/global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned"
EVAL_DIR = "/pscratch/sd/t/twamorka/omnilearned/eval/top_distill"

MODELS = {
    "teacher (fine_tune_top_l)": f"{ROOT}/outputs_fine_tune_top_l_top_*.npz",
    "CE-only (fine_tune_top_s)": f"{ROOT}/outputs_fine_tune_top_s_top_*.npz",
    "mixed KD (a=0.5, b=0.5)": f"{EVAL_DIR}/outputs_distill_top_small_scratch_a05_T4_top_test_rank*.npz",
    "pure KD (a=0, b=1)": f"{EVAL_DIR}/outputs_distill_top_small_scratch_a00_b10_T4_top_test_rank*.npz",
}
COLORS = {
    "teacher (fine_tune_top_l)": "#141413",
    "CE-only (fine_tune_top_s)": "#898781",
    "mixed KD (a=0.5, b=0.5)": "#d6822a",
    "pure KD (a=0, b=1)": "#2a78d6",
}


def load_glob(pattern):
    files = glob.glob(pattern)
    files = sorted(files, key=lambda f: int(re.search(r"(\d+)\.npz$", f).group(1)))
    preds, pids = [], []
    for f in files:
        d = np.load(f)
        preds.append(d["prediction"])
        pids.append(d["pid"])
    return np.concatenate(preds)[:, SIGNAL_CLASS], np.concatenate(pids)


def reliability_curve(p, labels, n_bins=15):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, n_bins - 1)
    mean_pred, frac_pos = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() == 0:
            continue
        mean_pred.append(p[mask].mean())
        frac_pos.append(labels[mask].mean())
    return np.array(mean_pred), np.array(frac_pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outfile", default="kd_diagnostics.pdf")
    args = ap.parse_args()

    data = {}
    labels_ref = None
    for name, pattern in MODELS.items():
        p, labels = load_glob(pattern)
        data[name] = p
        if labels_ref is None:
            labels_ref = labels
        else:
            assert np.array_equal(labels, labels_ref), f"{name}: label mismatch"

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # A. Overlaid output distributions
    ax = axes[0, 0]
    bins = np.linspace(0, 1, 51)
    for name, p in data.items():
        ax.hist(p, bins=bins, histtype="step", lw=1.8, color=COLORS[name], label=name)
    ax.set_yscale("log")
    ax.set_xlabel("predicted P(top)")
    ax.set_ylabel("count (log scale)")
    ax.set_title("A. Output distribution, all four models")
    ax.legend(fontsize=8)

    # B. Reliability diagram
    ax = axes[0, 1]
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for name, p in data.items():
        mp, fp = reliability_curve(p, labels_ref)
        ax.plot(mp, fp, marker="o", ms=4, color=COLORS[name], label=name)
    ax.set_xlabel("mean predicted P(top) in bin")
    ax.set_ylabel("empirical fraction top-labeled")
    ax.set_title("B. Reliability / calibration")
    ax.legend(fontsize=8)

    # C. Per-event disagreement with teacher
    ax = axes[1, 0]
    p_teacher = data["teacher (fine_tune_top_l)"]
    diff_bins = np.linspace(0, 1, 51)
    for name in ["mixed KD (a=0.5, b=0.5)", "pure KD (a=0, b=1)"]:
        diff = np.abs(data[name] - p_teacher)
        ax.hist(diff, bins=diff_bins, histtype="step", lw=1.8, color=COLORS[name],
                 label=f"{name}  (mean={diff.mean():.4f})")
    ax.set_yscale("log")
    ax.set_xlabel("|student P(top) - teacher P(top)|")
    ax.set_ylabel("count (log scale)")
    ax.set_title("C. Per-event disagreement with teacher")
    ax.legend(fontsize=8)

    # D. Restricted to events the teacher gets wrong
    ax = axes[1, 1]
    teacher_wrong = (p_teacher >= 0.5).astype(int) != labels_ref
    n_wrong = teacher_wrong.sum()
    for name in ["mixed KD (a=0.5, b=0.5)", "pure KD (a=0, b=1)"]:
        ax.hist(data[name][teacher_wrong], bins=bins, histtype="step", lw=1.8,
                 color=COLORS[name], label=name, density=True)
    ax.hist(p_teacher[teacher_wrong], bins=bins, histtype="step", lw=1.8,
             color=COLORS["teacher (fine_tune_top_l)"], label="teacher", density=True)
    ax.set_xlabel("predicted P(top)")
    ax.set_ylabel("density")
    ax.set_title(f"D. On the {n_wrong:,} events teacher gets WRONG")
    ax.axvline(0.5, color="gray", lw=1, ls="--")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(args.outfile)
    print(f"Saved {args.outfile}")
    print(f"\nTeacher wrong on {n_wrong:,} / {len(labels_ref):,} events "
          f"({100*n_wrong/len(labels_ref):.2f}%)")
    for name in ["mixed KD (a=0.5, b=0.5)", "pure KD (a=0, b=1)"]:
        sub_true = labels_ref[teacher_wrong]
        sub_pred_class = (data[name][teacher_wrong] >= 0.5).astype(int)
        agree_teacher = (sub_pred_class != sub_true).mean()
        print(f"  {name}: agrees with teacher's WRONG call on "
              f"{100*agree_teacher:.2f}% of those events")


if __name__ == "__main__":
    main()
