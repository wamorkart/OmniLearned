"""Plot AUC and 1/FPR@30% for the T-sweep and alpha/beta-sweep KD runs on top tagging.

Data below is transcribed from top_T_sweep_eval.log and top_distill_sweep_eval.log
(both already fully evaluated). Produces a 2x2 comparison figure.
"""
import numpy as np
import matplotlib.pyplot as plt

# --- T-sweep (alpha=0, beta=1, varying temperature) ---
T_vals = [1, 2, 4, 8, 16]
T_auc = [0.9879, 0.9879, 0.9879, 0.9879, 0.9879]
T_fpr30 = [3257, 3205, 3205, 3481, 3106]

# T=4 replicate variance (base + r1 + r2)
T4_auc_reps = [0.9879, 0.9879, 0.9879]
T4_fpr30_reps = [3205, 3481, 3205]
T4_auc_mean, T4_auc_std = np.mean(T4_auc_reps), np.std(T4_auc_reps)
T4_fpr30_mean, T4_fpr30_std = np.mean(T4_fpr30_reps), np.std(T4_fpr30_reps)

# --- alpha/beta sweep (T=4, varying loss weight alpha) ---
alpha_vals = [0.0, 0.25, 0.5]
alpha_auc = [0.9879, 0.9877, 0.9875]
alpha_fpr30 = [3205, 3205, 2804]

# alpha=0.5 replicate variance (base + r1..r4)
a05_auc_reps = [0.9875, 0.9874, 0.9875, 0.9875, 0.9875]
a05_fpr30_reps = [2804, 2692, 3059, 3106, 3106]
a05_auc_mean, a05_auc_std = np.mean(a05_auc_reps), np.std(a05_auc_reps)
a05_fpr30_mean, a05_fpr30_std = np.mean(a05_fpr30_reps), np.std(a05_fpr30_reps)

# --- references ---
ce_small = dict(auc=0.9875, fpr30=2556, label="fine_tune_top_s (CE-only small)")
ce_large = dict(auc=0.9880, fpr30=3365, label="fine_tune_top_l (CE-only large)")

fig, axes = plt.subplots(2, 2, figsize=(11, 8))

# Panel 1: AUC vs T
ax = axes[0, 0]
ax.plot(T_vals, T_auc, "o-", color="tab:blue", label="pure KD (a=0,b=1)")
ax.errorbar([4], [T4_auc_mean], yerr=[T4_auc_std], fmt="s", color="tab:blue",
            capsize=4, label="T=4 mean ± std (n=3)")
ax.axhline(ce_small["auc"], ls="--", color="gray", label=ce_small["label"])
ax.axhline(ce_large["auc"], ls=":", color="black", label=ce_large["label"])
ax.set_xscale("log", base=2)
ax.set_xticks(T_vals)
ax.set_xticklabels(T_vals)
ax.set_xlabel("Temperature T")
ax.set_ylabel("AUC")
ax.set_ylim(0.9865, 0.9885)
ax.set_title("T-sweep: AUC")
ax.legend(fontsize=8)

# Panel 2: 1/FPR@30% vs T
ax = axes[0, 1]
ax.plot(T_vals, T_fpr30, "o-", color="tab:blue")
ax.errorbar([4], [T4_fpr30_mean], yerr=[T4_fpr30_std], fmt="s", color="tab:blue", capsize=4)
ax.axhline(ce_small["fpr30"], ls="--", color="gray")
ax.axhline(ce_large["fpr30"], ls=":", color="black")
ax.set_xscale("log", base=2)
ax.set_xticks(T_vals)
ax.set_xticklabels(T_vals)
ax.set_xlabel("Temperature T")
ax.set_ylabel("1/FPR@30%")
ax.set_title("T-sweep: 1/FPR@30%")

# Panel 3: AUC vs alpha
ax = axes[1, 0]
ax.plot(alpha_vals, alpha_auc, "o-", color="tab:red", label=r"T=4, varying $\alpha$")
ax.errorbar([0.5], [a05_auc_mean], yerr=[a05_auc_std], fmt="s", color="tab:red",
            capsize=4, label=r"$\alpha$=0.5 mean ± std (n=5)")
ax.axhline(ce_small["auc"], ls="--", color="gray", label=ce_small["label"])
ax.axhline(ce_large["auc"], ls=":", color="black", label=ce_large["label"])
ax.set_xlabel(r"$\alpha$ (CE weight)")
ax.set_ylabel("AUC")
ax.set_title(r"$\alpha/\beta$-sweep: AUC")
ax.legend(fontsize=8)

# Panel 4: 1/FPR@30% vs alpha
ax = axes[1, 1]
ax.plot(alpha_vals, alpha_fpr30, "o-", color="tab:red")
ax.errorbar([0.5], [a05_fpr30_mean], yerr=[a05_fpr30_std], fmt="s", color="tab:red", capsize=4)
ax.axhline(ce_small["fpr30"], ls="--", color="gray")
ax.axhline(ce_large["fpr30"], ls=":", color="black")
ax.set_xlabel(r"$\alpha$ (CE weight)")
ax.set_ylabel("1/FPR@30%")
ax.set_title(r"$\alpha/\beta$-sweep: 1/FPR@30%")

fig.suptitle("Top-tagging KD distillation: T-sweep and alpha/beta-sweep", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.96))

out_pdf = "/pscratch/sd/t/twamorka/omnilearned/results/top_sweep_comparison.pdf"
out_png = "/pscratch/sd/t/twamorka/omnilearned/results/top_sweep_comparison.png"
fig.savefig(out_pdf)
fig.savefig(out_png, dpi=150)
print(f"Saved: {out_pdf}\nSaved: {out_png}")
