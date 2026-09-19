"""Verify the "steep-tail" ROC-shape hypothesis for pure-KD distillation on
top tagging: does the pure-KD student (distill_top_small_scratch_a00_b10_T4)
inherit its teacher's (fine_tune_top_l) unusually steep rejection-vs-efficiency
curve, while mixed-KD (a05_T4) and CE-only (fine_tune_top_s) do not?

CPU-only (loads small npz prediction files, sklearn ROC) -- no GPU needed, run
directly. Does not train or evaluate any model.
"""

import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

MODELS = {
    "fine_tune_top_l (teacher)": "/pscratch/sd/t/twamorka/omnilearned/teacher_logits/TEST/outputs_fine_tune_top_l_top_test_rank*.npz",
    "fine_tune_top_s (CE-only)": "/pscratch/sd/t/twamorka/omnilearned/teacher_logits/outputs_fine_tune_top_s_top_test_*.npz",
    "distill_top_small_scratch_a00_b10_T4 (pure KD)": "/pscratch/sd/t/twamorka/omnilearned/eval/top_distill/outputs_distill_top_small_scratch_a00_b10_T4_top_test_rank*.npz",
    "distill_top_small_scratch_a05_T4 (mixed KD)": "/pscratch/sd/t/twamorka/omnilearned/eval/top_distill/outputs_distill_top_small_scratch_a05_T4_top_test_rank*.npz",
}

EFFS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


def load(pattern):
    preds, labels = [], []
    for f in sorted(glob.glob(pattern)):
        z = np.load(f)
        preds.append(z["prediction"].astype(np.float64))
        labels.append(z["pid"])
    return np.concatenate(preds), np.concatenate(labels)


def rej_at_eff(fpr, tpr, eff):
    idx = np.searchsorted(tpr, eff)
    if idx >= len(fpr) or fpr[idx] == 0:
        return float("inf")
    return 1.0 / fpr[idx]


OUTDIR = "/pscratch/sd/t/twamorka/omnilearned/results"
COLORS = {
    "fine_tune_top_l (teacher)": "black",
    "fine_tune_top_s (CE-only)": "tab:red",
    "distill_top_small_scratch_a00_b10_T4 (pure KD)": "tab:blue",
    "distill_top_small_scratch_a05_T4 (mixed KD)": "tab:orange",
}

results = {}
curves = {}
for name, pattern in MODELS.items():
    preds, labels = load(pattern)
    scores = preds[:, 1]
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    rejs = {eff: rej_at_eff(fpr, tpr, eff) for eff in EFFS}
    results[name] = {"n": len(labels), "auc": auc, "rejs": rejs}
    curves[name] = (fpr, tpr)
    print(f"\n=== {name} (N={len(labels):,}, AUC={auc:.4f}) ===")
    for eff in EFFS:
        print(f"  1/FPR @ eff={eff:.1f}: {rejs[eff]:.1f}")

print("\n\n=== Ratio table: 1/FPR@eff / 1/FPR@0.5 (shape, normalized to the 50%-eff point) ===")
header = "eff  " + "  ".join(f"{name.split(' ')[0]:>32s}" for name in results)
print(header)
for eff in EFFS:
    row = [f"{eff:.1f}"]
    for name in results:
        ref = results[name]["rejs"][0.5]
        val = results[name]["rejs"][eff]
        ratio = val / ref if ref not in (0, float("inf")) else float("nan")
        row.append(f"{ratio:32.3f}")
    print("  ".join(row))

print("\n\n=== Sanity check against known summary numbers ===")
for name in results:
    r = results[name]["rejs"]
    print(f"{name}: AUC={results[name]['auc']:.4f}  1/FPR@50%={r[0.5]:.1f}  1/FPR@30%={r[0.3]:.1f}")

# --- Plots ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

MARK_EFFS = [0.5, 0.3]
SHORT = {name: name.split(" (")[0] for name in results}

ax = axes[0]
for name, (fpr, tpr) in curves.items():
    with np.errstate(divide="ignore"):
        rej = np.where(fpr > 0, 1.0 / fpr, np.inf)
    ax.plot(tpr, rej, label=SHORT[name], color=COLORS[name], lw=1.8)
    for eff in MARK_EFFS:
        val = results[name]["rejs"][eff]
        ax.scatter([eff], [val], color=COLORS[name], s=45, zorder=5, edgecolor="white", linewidth=0.6)
for eff in MARK_EFFS:
    ax.axvline(eff, color="gray", ls=":", lw=1)
ax.set_yscale("log")
ax.set_xlim(0.0, 1.0)
ax.set_xlabel(r"Signal efficiency $\epsilon_{sig}$")
ax.set_ylabel(r"Background rejection $1/\epsilon_{bkg}$")
ax.set_title("Rejection vs. signal efficiency\n(dotted lines mark 50%/30% eff)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)

ax = axes[1]
for name in results:
    ref = results[name]["rejs"][0.5]
    xs = sorted(EFFS)
    ys = [results[name]["rejs"][e] / ref for e in xs]
    ax.plot(xs, ys, marker="o", ms=4, label=SHORT[name], color=COLORS[name], lw=1.8)
for eff in MARK_EFFS:
    ax.axvline(eff, color="gray", ls=":", lw=1)
ax.set_yscale("log")
ax.set_xlabel(r"Signal efficiency $\epsilon_{sig}$")
ax.set_ylabel(r"$(1/\epsilon_{bkg})$ / $(1/\epsilon_{bkg}$ @ 50% eff$)$")
ax.set_title("Rejection-curve shape\n(normalized to 50% eff.)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)

# Bar chart: the actual difference in performance at exactly 50%/30% eff.
ax = axes[2]
names = list(results.keys())
n = len(names)
group_w = 0.8
bar_w = group_w / n
x_groups = np.arange(len(MARK_EFFS))
for i, name in enumerate(names):
    heights = [results[name]["rejs"][eff] for eff in MARK_EFFS]
    xpos = x_groups + (i - (n - 1) / 2) * bar_w
    bars = ax.bar(xpos, heights, width=bar_w * 0.92, color=COLORS[name], label=SHORT[name])
    for b, h in zip(bars, heights):
        ax.annotate(
            f"{h:.0f}", (b.get_x() + b.get_width() / 2, h), textcoords="offset points",
            xytext=(0, 3), fontsize=7.5, ha="center", color=COLORS[name],
        )
ax.set_xticks(x_groups)
ax.set_xticklabels([f"{int(e*100)}% eff." for e in MARK_EFFS])
ax.set_ylabel(r"Background rejection $1/\epsilon_{bkg}$")
ax.set_title("Direct comparison at 50% and 30% signal eff.")
ax.legend(fontsize=8, loc="upper left")
ax.grid(alpha=0.3, axis="y")

fig.suptitle("fine_tune_top_l teacher vs. CE-only / pure-KD / mixed-KD students (top tagging, 404k test events)")
fig.tight_layout()

import os
os.makedirs(OUTDIR, exist_ok=True)
outpath_pdf = os.path.join(OUTDIR, "roc_shape_comparison.pdf")
outpath_png = os.path.join(OUTDIR, "roc_shape_comparison.png")
fig.savefig(outpath_pdf)
fig.savefig(outpath_png, dpi=150)
print(f"\nSaved plot to {outpath_pdf} and {outpath_png}")
