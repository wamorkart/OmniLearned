"""Distribution of a top-tagging classifier's output logits on the 404k top test split.

Two pre-saved sources, no GPU / re-eval needed:

  large   : the large OmniLearned model pretrained on JetClass then fine-tuned to
            top (`fine_tune_top_l`) -- this is the KD teacher for the FPGA study.
            logits from  labels/teacher_labels_fine_tune_top_l_top_test_merged.npz
  gnn     : the float distillnet + 1 GNN block student
            (`distill_top_deepsets_distillnet_scratch_a05_T4_gnn1_k64`)
            logits from  eval/top_distill_deepsets/outputs_..._top_test_rank*.npz

Usage:  python plot_top_logits.py [large|gnn]   (default: large)
"""
import glob
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

WHICH = sys.argv[1] if len(sys.argv) > 1 else "large"
SP = "/pscratch/sd/t/twamorka/omnilearned"
OUT_DIR = "/global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned/analysis"

if WHICH == "large":
    d = np.load(f"{SP}/labels/teacher_labels_fine_tune_top_l_top_test_merged.npz")
    logits, pid = d["logits"].astype(np.float32), d["labels"]
    title_model = "fine_tune_top_l  (large model, JetClass-pretrained -> fine-tuned to top)"
    out = f"{OUT_DIR}/top_logit_distribution_large.png"
elif WHICH == "gnn":
    fs = sorted(glob.glob(f"{SP}/eval/top_distill_deepsets/"
                          "outputs_distill_top_deepsets_distillnet_scratch_a05_T4_gnn1_k64_top_test_rank*.npz"))
    logits = np.concatenate([np.load(f)["logits"].astype(np.float32) for f in fs])
    pid = np.concatenate([np.load(f)["pid"] for f in fs])
    title_model = "distill_top_deepsets_distillnet_scratch_a05_T4_gnn1_k64  (distillnet + GNN student)"
    out = f"{OUT_DIR}/top_logit_distribution_gnn.png"
else:
    raise SystemExit("arg must be 'large' or 'gnn'")

print(f"{WHICH}: {len(pid):,} events, logits {logits.shape}")
L_qcd, L_top = logits[:, 0], logits[:, 1]
D = L_top - L_qcd
is_top, is_qcd = pid == 1, pid == 0

try:
    from sklearn.metrics import roc_auc_score, roc_curve
    score = 1.0 / (1.0 + np.exp(-D))
    auc = roc_auc_score(pid, score)
    fpr, tpr, _ = roc_curve(pid, score)
    rej50 = 1.0 / fpr[tpr.searchsorted(0.50)]
    rej30 = 1.0 / fpr[tpr.searchsorted(0.30)]
    acc = ((D > 0).astype(int) == pid).mean()
    ann = f"acc {acc*100:.2f}%   AUC {auc:.4f}   1/FPR@50% {rej50:.0f}   1/FPR@30% {rej30:.0f}"
except Exception as e:  # noqa
    ann = f"(metrics skipped: {e})"
print(ann)

C_TOP, C_QCD = "#c1121f", "#264653"
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))

b = np.linspace(-20, 20, 130)
ax[0].hist(L_top, bins=b, histtype="stepfilled", alpha=.55, color=C_TOP,
           label=r"$L_\mathrm{top}$ (class-1 node)")
ax[0].hist(L_qcd, bins=b, histtype="stepfilled", alpha=.55, color=C_QCD,
           label=r"$L_\mathrm{QCD}$ (class-0 node)")
ax[0].set_title("Raw output logits, per class node\n(all 404k test jets)")
ax[0].set_xlabel("logit value"); ax[0].set_ylabel("jets"); ax[0].legend(fontsize=9)

ax[1].hist(L_top[is_top], bins=b, histtype="stepfilled", alpha=.55, color=C_TOP, label="true top")
ax[1].hist(L_top[is_qcd], bins=b, histtype="stepfilled", alpha=.55, color=C_QCD, label="true QCD")
ax[1].set_title(r"Top-node logit $L_\mathrm{top}$, by true class")
ax[1].set_xlabel(r"$L_\mathrm{top}$"); ax[1].set_ylabel("jets"); ax[1].legend(fontsize=9)

bd = np.linspace(-35, 35, 130)
ax[2].hist(D[is_top], bins=bd, histtype="stepfilled", alpha=.55, color=C_TOP, label="true top")
ax[2].hist(D[is_qcd], bins=bd, histtype="stepfilled", alpha=.55, color=C_QCD, label="true QCD")
ax[2].axvline(0, color="k", lw=.8, ls="--")
ax[2].set_yscale("log")
ax[2].set_title(r"Discriminant $L_\mathrm{top}-L_\mathrm{QCD}$ (log-odds), log $y$")
ax[2].set_xlabel(r"$L_\mathrm{top}-L_\mathrm{QCD}$"); ax[2].set_ylabel("jets"); ax[2].legend(fontsize=9)

fig.suptitle(f"Top-tagging output-logit distributions  |  {title_model}\n{ann}", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(out, dpi=140)
print("wrote", out)
