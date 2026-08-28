"""Plot the large teacher's (fine_tune_top_l) soft outputs on the top-tagging test set.

Panel 1: raw P(top) from the teacher's softmax (T=1), log-scaled counts.
Panel 2: temperature-softened targets softmax(logit/T) for T in
{0.125, 0.25, 0.5, 1, 2, 4, 8, 16}, reconstructed by inverting the T=1 probability
back to a logit and rescaling. T<1 sharpens back toward the raw bimodal T=1 shape
(matching the T-sweep's observed collapse below T=1); T>1 flattens it.

Usage:
    python plot_teacher_soft_outputs.py --indir . --tag fine_tune_top_l \
        --outfile /pscratch/sd/t/twamorka/omnilearned/results/teacher_soft_outputs_top_l.pdf
"""

import argparse
import glob
import os

import numpy as np
import matplotlib.pyplot as plt

SIGNAL_CLASS = 1
TEMPERATURES = [0.125, 0.25, 0.5, 1, 2, 4, 8, 16]
EPS = 1e-7


def load_teacher_probs(indir, tag):
    pattern = os.path.join(indir, f"outputs_{tag}_top_*.npz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching: {pattern}")
    probs = []
    for p in paths:
        z = np.load(p)
        probs.append(z["prediction"][:, SIGNAL_CLASS].astype(np.float64))
    return np.concatenate(probs)


def softmax_temp(p1, T):
    p1 = np.clip(p1, EPS, 1 - EPS)
    logit = np.log(p1 / (1 - p1))
    return 1.0 / (1.0 + np.exp(-logit / T))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default=".", help="Directory with per-shard outputs_<tag>_top_*.npz files")
    ap.add_argument("--tag", default="fine_tune_top_l", help="save_tag used during evaluate")
    ap.add_argument("--outfile", default="teacher_soft_outputs.pdf")
    args = ap.parse_args()

    p1 = load_teacher_probs(args.indir, args.tag)
    n = len(p1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9))

    # Panel 1: raw teacher output, T=1
    bins = np.linspace(0, 1, 51)
    ax1.hist(p1, bins=bins, color="#2a78d6", alpha=0.9)
    ax1.set_yscale("log")
    ax1.set_xlabel("predicted P(top)")
    ax1.set_ylabel("count (log scale)")
    ax1.set_title(f"Raw teacher output, T=1  ({args.tag}, n={n:,})")
    ax1.text(
        0.02, 0.95,
        f"mean = {p1.mean():.4f}\nstd = {p1.std():.4f}",
        transform=ax1.transAxes, va="top", fontsize=9,
    )

    # Panel 2: temperature-softened targets
    bin_edges = np.linspace(0, 1, 31)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    colors = {
        0.125: "#7a0000",
        0.25: "#b3221a",
        0.5: "#e8813a",
        1: "#444444",
        2: "#86b6ef",
        4: "#5598e7",
        8: "#2a78d6",
        16: "#0d366b",
    }
    for T in TEMPERATURES:
        p_T = softmax_temp(p1, T)
        counts, _ = np.histogram(p_T, bins=bin_edges)
        style = "--" if T < 1 else "-"
        ax2.plot(centers, counts, style, marker="o", markersize=3, color=colors[T], label=f"T = {T}")
    ax2.set_yscale("log")
    ax2.set_xlabel("softened score = softmax(logit / T)")
    ax2.set_ylabel("count (log scale)")
    ax2.set_title("Temperature-softened teacher targets (dashed: T<1 sharpening)")
    ax2.legend(ncol=2, fontsize=8)

    fig.tight_layout()
    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    fig.savefig(args.outfile)
    print(f"Saved {args.outfile}")


if __name__ == "__main__":
    main()
