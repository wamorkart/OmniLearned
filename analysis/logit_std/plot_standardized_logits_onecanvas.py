"""Standardized teacher-logit value distribution for every pretrain-mixture
dataset, all on one axes.

Companion to plot_pretrain_logits.py (which shows the *raw* spread that motivates
the transform). Here each sample's 210-dim logit vector is per-sample Z-scored
exactly as src/omnilearned/utils.py::_standardize_logits does in the KD path
(--distill-standardize), then all values are pooled into one step histogram per
dataset. Same tab10 dataset colors and style as pretrain_logits_by_dataset.png.
"""
import os

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

COMPANION = "/pscratch/sd/t/twamorka/omnilearned/teacher_logits/companion"
DATASETS = ["atlas", "aspen", "jetclass", "jetclass2", "h1", "cms_qcd", "cms_bsm"]
N_PER = 20000          # contiguous rows read per dataset (matches plot_pretrain_logits.py)
EPS = 1e-7             # matches _standardize_logits

colors = plt.cm.tab10(np.linspace(0, 1, len(DATASETS)))
cmap = dict(zip(DATASETS, colors))


def standardize(z):
    """Per-sample Z-score along the class dim; ddof=1 to match torch.std default."""
    z = z - z.mean(axis=1, keepdims=True)
    return z / (z.std(axis=1, ddof=1, keepdims=True) + EPS)


std_vals = {}
for ds in DATASETS:
    d = os.path.join(COMPANION, ds, "train")
    files = sorted(f for f in os.listdir(d) if f.endswith((".h5", ".hdf5")))
    with h5py.File(os.path.join(d, files[0]), "r") as hf:
        n = min(N_PER, hf["teacher_logits"].shape[0])
        lg = hf["teacher_logits"][:n].astype(np.float32)
    zs = standardize(lg)
    std_vals[ds] = zs.reshape(-1)
    print(f"{ds:10s} n={n:6d}  standardized mean {zs.mean():+.3f}  std {zs.std():.3f}  "
          f"min {zs.min():+.2f}  max {zs.max():+.2f}  ({files[0]})")

fig, ax = plt.subplots(figsize=(9, 6))
bins = np.linspace(-6, 10, 160)
for ds in DATASETS:
    ax.hist(std_vals[ds], bins=bins, density=True, histtype="step", lw=1.6,
            color=cmap[ds], label=ds)

xs = np.linspace(-6, 10, 400)
ax.plot(xs, np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi), "k--", lw=1.0,
        alpha=0.7, label=r"$\mathcal{N}(0,1)$")

ax.set_yscale("log")
ax.set_ylim(1e-6, 3e1)
ax.set_xlim(-6, 10)
ax.set_xlabel("standardized logit value")
ax.set_ylabel("density (log)")
ax.set_title(
    "pretrain_l teacher logits after per-sample standardization\n"
    f"by source dataset  (first {N_PER} train rows each, 210 dims pooled)",
    fontsize=12)
ax.legend(fontsize=9, ncol=2)
ax.grid(alpha=0.3)

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "pretrain_standardized_logits_onecanvas.png")
fig.savefig(out, dpi=130)
print("wrote", out)
