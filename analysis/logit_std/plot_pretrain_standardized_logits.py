#!/usr/bin/env python3
"""Teacher-logit distributions, raw vs logit-standardized, for every dataset in
the pretrain mixture.

The pretrain KD path (src/omnilearned/utils.py::_standardize_logits, added in
73a2f67) Z-scores each sample's logit vector along the class dim before the KD
softmax. This script applies the *same* transform to the pre-saved teacher
logits (`teacher_logits`, (N, 210) float16, merged into each source .h5) and
plots, per pretrain dataset:

  row 1  raw teacher logits          -- per-panel autoscale; the point is that
                                        the scale differs wildly per dataset
  row 2  standardized teacher logits -- shared x, overlaid unit normal ref

All 210 dims (200 jet-class + 10 event-class) are pooled, matching what the KD
loss standardizes over. Reads from the val split (fully reproducible).
"""
import argparse
import glob
import os

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PRETRAIN = ["atlas", "aspen", "jetclass", "jetclass2", "h1", "cms_qcd", "cms_bsm"]
EPS = 1e-7  # matches _standardize_logits


def standardize(z):
    """Per-sample Z-score along the class dim. ddof=1 to match torch.std default."""
    z = z - z.mean(axis=-1, keepdims=True)
    return z / (z.std(axis=-1, ddof=1, keepdims=True) + EPS)


def load_logits(data_path, dataset, split, n_max):
    """First n_max rows of teacher_logits for dataset/split, walking files in
    sorted order until n_max is reached."""
    files = sorted(
        glob.glob(os.path.join(data_path, dataset, split, "*.h5"))
        + glob.glob(os.path.join(data_path, dataset, split, "*.hdf5"))
    )
    if not files:
        raise FileNotFoundError(f"no h5 for {dataset}/{split} under {data_path}")
    chunks, got = [], 0
    for f in files:
        with h5py.File(f, "r") as hf:
            if "teacher_logits" not in hf:
                raise KeyError(f"{f} has no teacher_logits dataset")
            need = n_max - got
            arr = np.asarray(hf["teacher_logits"][:need], dtype=np.float32)
        chunks.append(arr)
        got += arr.shape[0]
        if got >= n_max:
            break
    return np.concatenate(chunks, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default="/pscratch/sd/t/twamorka/omnilearned/datasets")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n-per-dataset", type=int, default=150_000)
    ap.add_argument("--bins", type=int, default=200)
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "pretrain_standardized_logits.png"),
    )
    args = ap.parse_args()

    ncol = len(PRETRAIN)
    fig, axes = plt.subplots(2, ncol, figsize=(3.0 * ncol, 6.2))

    std_range = (-5.0, 5.0)
    xs = np.linspace(*std_range, 400)
    unit_normal = np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi)

    print(f"{'dataset':10s} {'n_jets':>9s} {'raw mean':>10s} {'raw std':>10s} "
          f"{'raw min':>9s} {'raw max':>9s}")
    for j, ds in enumerate(PRETRAIN):
        z = load_logits(args.data_path, ds, args.split, args.n_per_dataset)
        zs = standardize(z)
        print(f"{ds:10s} {z.shape[0]:9d} {z.mean():10.3f} {z.std():10.3f} "
              f"{z.min():9.2f} {z.max():9.2f}")

        ax_raw, ax_std = axes[0, j], axes[1, j]
        ax_raw.hist(z.reshape(-1), bins=args.bins, color="#4c72b0", density=True)
        ax_raw.set_title(ds, fontsize=11)
        ax_raw.set_yscale("log")

        ax_std.hist(
            zs.reshape(-1), bins=args.bins, range=std_range,
            color="#c44e52", density=True,
        )
        ax_std.plot(xs, unit_normal, "k--", lw=1.0, alpha=0.7)
        ax_std.set_xlim(*std_range)
        ax_std.set_yscale("log")

        if j == 0:
            ax_raw.set_ylabel("raw teacher logits\n(density, log)")
            ax_std.set_ylabel("standardized\n(density, log)")

    for ax in axes[1]:
        ax.set_xlabel("logit value")
    fig.suptitle(
        f"Pretrain teacher logits: raw vs per-sample standardized "
        f"(split={args.split}, {args.n_per_dataset:,} jets/dataset, 210 dims pooled)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=130)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
