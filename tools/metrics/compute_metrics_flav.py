"""Score the atlas_flav jet-flavour head: accuracy, OvR AUC, and flavour
rejections at fixed b-/c-tagging working points.

The atlas_flav label map is NOT the GN2 convention -- it was derived
empirically in analysis/characterize_atlas_flav.py:

    0 = light    1 = c    2 = b    3 = tau

Scoring reproduces the notebook used for the OmniLearned paper numbers
(https://gist.github.com/ViniciusMikuni/dbcfda9da5f082194bf996298dedff0b):

  * jets are required to have pT = exp(cond[:, 0]) > 20 GeV;
  * discriminants are the ATLAS FTAG log-likelihood ratios

        D_b = log( p_b / (f_c p_c + f_tau p_tau + (1 - f_c - f_tau) p_light) )
        D_c = log( p_c / (f_b p_b + f_tau p_tau + (1 - f_b - f_tau) p_light) )

    with f_c = 0.2, f_b = 0.2, f_tau = 0.01, and log of a non-positive ratio
    set to 0 (np.ma.log(...).filled(0));
  * rejection of a background flavour is taken from the signal-vs-that-
    background ROC curve at the first point with tpr > working point.

Usage:
    python compute_metrics_flav.py --indir /path/to/eval/dir \
        --tag fine_tune_atlas_flav_m_localint
    python compute_metrics_flav.py --file /path/to/concat.npz
"""

import argparse
import glob
import os

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

CLASS_NAMES = ["light", "c", "b", "tau"]
LIGHT, C, B, TAU = 0, 1, 2, 3

B_WPS = [0.60, 0.70, 0.77, 0.85]
C_WPS = [0.20, 0.30, 0.40]

# Background composition priors for the LLR discriminants (paper values).
F_C = 0.2      # c fraction in the b-tagging background hypothesis
F_B = 0.2      # b fraction in the c-tagging background hypothesis
F_TAU = 0.01   # tau fraction in both background hypotheses

PT_CUT = 20.0  # GeV, applied to exp(cond[:, 0])


def load_rank_files(indir, tag):
    pat = os.path.join(indir, f"outputs_{tag}_atlas_flav_test*rank*.npz")
    paths = sorted(glob.glob(pat))
    if not paths:
        raise FileNotFoundError(f"no files matching {pat}")
    print(f"Found {len(paths)} rank files")
    preds, labels, pts = [], [], []
    for p in paths:
        z = np.load(p)
        preds.append(z["prediction"].astype(np.float64))
        labels.append(z["pid"].astype(np.int64))
        pts.append(np.exp(z["cond"][:, 0].astype(np.float64)))
    return np.concatenate(preds), np.concatenate(labels), np.concatenate(pts)


def load_concat_file(path):
    z = np.load(path)
    pt = np.exp(z["cond"][:, 0].astype(np.float64))
    return z["prediction"].astype(np.float64), z["pid"].astype(np.int64), pt


def rej_at_wp(sig_score, bkg_score, sig_eff):
    """1 / bkg_eff at the first ROC point with signal efficiency > sig_eff."""
    y = np.concatenate([np.ones(len(sig_score)), np.zeros(len(bkg_score))])
    fpr, tpr, _ = roc_curve(y, np.concatenate([sig_score, bkg_score]))
    i = np.argmax(tpr > sig_eff)
    return np.inf if fpr[i] == 0 else 1.0 / fpr[i]


def llr(num, *dens_terms):
    den = np.zeros_like(num)
    for frac, p in dens_terms:
        den = den + frac * p
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.ma.log(num / den).filled(0)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--indir", help="dir with per-rank outputs_*.npz")
    g.add_argument("--file", help="single concatenated npz")
    ap.add_argument("--tag", default="fine_tune_atlas_flav_m_localint")
    args = ap.parse_args()

    if args.indir:
        preds, labels, pt = load_rank_files(args.indir, args.tag)
    else:
        preds, labels, pt = load_concat_file(args.file)

    n_all = len(labels)
    keep = pt > PT_CUT
    preds, labels = preds[keep, :4], labels[keep]
    p_light, p_c, p_b, p_tau = preds[:, LIGHT], preds[:, C], preds[:, B], preds[:, TAU]

    n = len(labels)
    counts = np.bincount(labels, minlength=4)
    pred_cls = preds.argmax(axis=1)
    acc = (pred_cls == labels).mean()

    print(f"\n=== atlas_flav jet-flavour head  |  tag: {args.tag} ===")
    print(f"  N jets    : {n:,}  (pT > {PT_CUT:g} GeV, of {n_all:,})")
    for i, name in enumerate(CLASS_NAMES):
        frac = counts[i] / n
        recall = (pred_cls[labels == i] == i).mean() if counts[i] else float("nan")
        prec = (labels[pred_cls == i] == i).mean() if (pred_cls == i).any() else float("nan")
        print(f"    {name:<5}: {counts[i]:>10,}  ({frac:6.2%} of sample)   "
              f"recall {recall:6.2%}   precision {prec:6.2%}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")

    # One-vs-rest AUC per class.
    print("\n  One-vs-rest AUC (softmax prob):")
    for i, name in enumerate(CLASS_NAMES):
        auc = roc_auc_score((labels == i).astype(int), preds[:, i])
        print(f"    {name:<5}: {auc:.4f}")

    # b-tagging: signal = b, discriminant D_b.
    D_b = llr(p_b, (F_C, p_c), (F_TAU, p_tau), (1.0 - F_C - F_TAU, p_light))
    is_b = labels == B
    sig = D_b[is_b]
    bkgs = {"light": D_b[labels == LIGHT], "c": D_b[labels == C], "tau": D_b[labels == TAU]}
    print(f"\n  b-tagging  (D_b LLR, f_c={F_C}, f_tau={F_TAU})   "
          f"AUC(b vs all)={roc_auc_score(is_b.astype(int), D_b):.4f}")
    print(f"    {'b-eff':>7} " + "".join(f"{'rej_'+k:>12}" for k in bkgs))
    for wp in B_WPS:
        cells = "".join(f"{rej_at_wp(sig, bkgs[k], wp):>12.1f}" for k in bkgs)
        print(f"    {wp:>7.0%} {cells}")

    # c-tagging: signal = c, discriminant D_c.
    D_c = llr(p_c, (F_B, p_b), (F_TAU, p_tau), (1.0 - F_B - F_TAU, p_light))
    is_c = labels == C
    sig_c = D_c[is_c]
    bkgs_c = {"light": D_c[labels == LIGHT], "b": D_c[labels == B], "tau": D_c[labels == TAU]}
    print(f"\n  c-tagging  (D_c LLR, f_b={F_B}, f_tau={F_TAU})   "
          f"AUC(c vs all)={roc_auc_score(is_c.astype(int), D_c):.4f}")
    print(f"    {'c-eff':>7} " + "".join(f"{'rej_'+k:>12}" for k in bkgs_c))
    for wp in C_WPS:
        cells = "".join(f"{rej_at_wp(sig_c, bkgs_c[k], wp):>12.1f}" for k in bkgs_c)
        print(f"    {wp:>7.0%} {cells}")
    print()


if __name__ == "__main__":
    main()
