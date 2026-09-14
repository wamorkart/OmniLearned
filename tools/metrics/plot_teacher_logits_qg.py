"""Plot the qg teacher's logit/probability distribution, split by true class.

Diagnostic for "why doesn't distillation help": if the teacher is
overconfident/saturated (probabilities jammed against 0/1, or huge logit
gaps), its soft labels carry little more information than the hard labels
already do, so KD has little extra signal to transfer to the student.

Reads the SAME per-rank outputs_<tag>_qg_<split>_rank*.npz files evaluate.py
writes -- by default the `train` split, since that's what build_teacher_h5.py
converts into the teacher_logits companions actually used during distillation.

Usage:
    python plot_teacher_logits_qg.py --indir /pscratch/sd/m/mbenyas --tag fine_tune_qg_pretrain_l
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def load_rank_files(indir, tag, dataset, split):
    """Concatenate prediction/logits/pid from every per-rank npz for one tag/split."""
    pattern = os.path.join(indir, f"outputs_{tag}_{dataset}_{split}_rank*.npz")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching: {pattern}")
    print(f"Found {len(paths)} rank files")
    pred, logits, pid = [], [], []
    for p in paths:
        z = np.load(p)
        pred.append(z["prediction"].astype(np.float32))
        logits.append(z["logits"].astype(np.float32))
        pid.append(z["pid"])
    return np.concatenate(pred), np.concatenate(logits), np.concatenate(pid)


def report_saturation(pred, pid, thresh=0.99):
    """Fraction of examples where the teacher's top-class probability exceeds
    `thresh` -- a single number for how close to hard-label the soft labels
    already are."""
    top_prob = pred.max(axis=1)
    frac = (top_prob > thresh).mean()
    print(f"P(top class) > {thresh}: {frac * 100:.2f}% of examples")
    print(f"P(top class) mean={top_prob.mean():.4f} median={np.median(top_prob):.4f}")


def plot(pred, logits, pid, outfile):
    # One curve per output channel, pooled across all examples (no true-label
    # split) -- matches the style of the pretraining per-dataset logit plot:
    # raw, un-collapsed logit value on the x-axis, not a derived quantity like
    # a class-1-minus-class-0 gap.
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    lo, hi = np.percentile(logits, [0.5, 99.5])
    for cls in range(logits.shape[1]):
        ax.hist(logits[:, cls], bins=100, range=(lo, hi), histtype="step", density=True, label=f"logit[{cls}]")
    ax.set_xlabel("logit value")
    ax.set_ylabel("density (log)")
    ax.set_yscale("log")
    ax.legend()
    ax.set_title("Raw logit value distribution")

    fig.tight_layout()
    fig.savefig(outfile, dpi=150)
    print(f"Saved plot to {outfile}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--indir", required=True, help="Directory with per-rank npz files")
    ap.add_argument("--tag", required=True, help="save_tag used during evaluate")
    ap.add_argument("--dataset", default="qg")
    ap.add_argument(
        "--split",
        default="train",
        help="Which split to plot (default train, since that's what feeds distillation)",
    )
    ap.add_argument("--outfile", default=None, help="Default: teacher_logits_<tag>_<split>.png")
    args = ap.parse_args()

    outfile = args.outfile or f"teacher_logits_{args.tag}_{args.split}.png"

    pred, logits, pid = load_rank_files(args.indir, args.tag, args.dataset, args.split)
    report_saturation(pred, pid)
    plot(pred, logits, pid, outfile)


if __name__ == "__main__":
    main()
