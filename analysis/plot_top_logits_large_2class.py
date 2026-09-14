"""Normalized distributions of the two output-node logits (class 0 = QCD,
class 1 = top) from the large model fine-tuned to top (`fine_tune_top_l`),
on the 404k top test split. Step outlines, no fill.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "/pscratch/sd/t/twamorka/omnilearned/labels/teacher_labels_fine_tune_top_l_top_test_merged.npz"
OUT = "/global/cfs/cdirs/m3246/twamorka/omnilearned_test/OmniLearned/analysis/top_logits_large_2class.png"

d = np.load(SRC)
logits = d["logits"].astype(np.float32)
L_qcd, L_top = logits[:, 0], logits[:, 1]

bins = np.linspace(-12, 12, 100)
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.hist(L_qcd, bins=bins, density=True, histtype="step", lw=1.8,
        color="#264653", label="class 0  (QCD)")
ax.hist(L_top, bins=bins, density=True, histtype="step", lw=1.8,
        color="#c1121f", label="class 1  (top)")
ax.set_yscale("log")
ax.set_xlabel("output logit")
ax.set_ylabel("normalized density")
ax.set_title("Output-node logits, large model fine-tuned to top\n(fine_tune_top_l, 404k top test jets)")
ax.legend()
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
