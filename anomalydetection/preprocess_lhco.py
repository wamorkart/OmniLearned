#!/usr/bin/env python
"""preprocess_lhco.py - Convert the OmniLearn LHCO h5 files into OmniLearned's
HEPDataset schema so they can be used with `omnilearned train --dataset custom`.

Input (produced upstream by OmniLearn, e.g.
/global/cfs/cdirs/m3246/vmikuni/PET/LHCO/{train,val}_{background,signal}_SR*.h5):

    data (N, 2, 279, 7)  two jets x up to 279 constituents x 7 features
    jet  (N, 2, 5)       [pT, eta, phi, mass, multiplicity] per jet
    pid  (N,)            despite the name this is mjj (GeV), not a particle id

    The 7 constituent features are already relative/log transformed:
        0 d_eta wrt its own jet axis   4 log(E_rel)
        1 d_phi wrt its own jet axis   5 log(E)
        2 log(pT_rel)                  6 dR
        3 log(pT)

Output (one .h5 per source file per split, under
<out-dir>/custom/{train,val,test}/ -- `custom` is load_data's generic dataset
slot, dataloader.py:381; any other name has to be added to supported_datasets):

    data   (N, K, F)  flattened dijet point cloud, both jets in one cloud
    pid    (N,)       int64 class label, see --mode
    global (N, 5)     event-level conditioning, [mjj_scaled, pT1, m1, pT2, m2]
    mjj    (N,)       raw dijet mass in GeV, kept for AD binning / bump hunts

Conventions carried over verbatim from torch_lhco.py (the L-GATr LHCO loader) so
the two trainings see the same events:
  * min-pT cut  X *= (X[:, :, :, 3:4] > 0)   -> pT > 1 GeV   (torch_lhco.py:47)
  * validity    mask = X[:, :, :, 2] != 0                     (torch_lhco.py:48)
  * mjj scaling prep_mjj, linear map [2300, 5000] -> [-1, 1]  (torch_lhco.py:249)

Deliberate differences from torch_lhco.py's make_ptdata(), all forced by what
OmniLearned's model actually reads:

  * Features are written RAW, not standardized. OmniLearned masks on
    `x[:, :, 2:3] != 0` (network.py:332,543,681) so column 2 must be exactly 0
    on padding; subtracting mean_part/std_part would make padding nonzero and
    silently turn every pad slot into a real particle.
  * Column order is kept as [d_eta, d_phi, log(pT_rel), log(pT), ...]. The
    interaction matrix reads x[..., 0] as eta, x[..., 1] as phi and
    exp(x[..., 2]) as pT (layers.py:65 get_mass, :81 get_dr), so those three
    slots are not free to reorder.
  * The jet-id flag goes LAST, because HEPDataset pulls extra features off the
    end of the array: `sample["add_info"] = sample["X"][:, -num_add:]`
    (dataloader.py:281). That is what makes `--use-add --num-add N` work.
  * No per-jet summary token is prepended. torch_lhco.py adds one because
    L-GATr wants it; OmniLearned adds its own class token internally
    (network.py:581) and a summary token would break the index-2 mask
    convention (its column 2 is a jet feature, not log(pT_rel)).

Usage:
    python anomalydetection/preprocess_lhco.py \
        --data-dir /global/cfs/cdirs/m3246/vmikuni/PET/LHCO \
        --out-dir  $SCRATCH/omnilearned/datasets \
        --mode idealized

    python anomalydetection/preprocess_lhco.py ... --mode cwola --nsig 1000
"""

import argparse
import os
from pathlib import Path

import h5py
import numpy as np

# Feature columns in the raw LHCO `data` array.
IDX_DETA, IDX_DPHI, IDX_LOGPTREL, IDX_LOGPT, IDX_LOGEREL, IDX_LOGE, IDX_DR = range(7)
# Columns in the raw `jet` array.
JET_PT, JET_ETA, JET_PHI, JET_MASS, JET_MULT = range(5)


def prep_mjj(mjj, mjjmin=2300.0, mjjmax=5000.0):
    """Linear map of mjj onto [-1, 1]. Same as torch_lhco.py:249."""
    return (2.0 * (mjj - mjjmin) / (mjjmax - mjjmin) - 1.0).astype(np.float32)


def get_dimass(jet):
    """Dijet mass from the per-jet [pT, eta, phi, mass, mult] array.

    Same as torch_lhco.py:288, used only when a file has no `pid` dataset.
    """
    e = np.sqrt(jet[:, 0, JET_MASS] ** 2
                + jet[:, 0, JET_PT] ** 2 * np.cosh(jet[:, 0, JET_ETA]) ** 2)
    e += np.sqrt(jet[:, 1, JET_MASS] ** 2
                 + jet[:, 1, JET_PT] ** 2 * np.cosh(jet[:, 1, JET_ETA]) ** 2)
    px = (jet[:, 0, JET_PT] * np.cos(jet[:, 0, JET_PHI])
          + jet[:, 1, JET_PT] * np.cos(jet[:, 1, JET_PHI]))
    py = (jet[:, 0, JET_PT] * np.sin(jet[:, 0, JET_PHI])
          + jet[:, 1, JET_PT] * np.sin(jet[:, 1, JET_PHI]))
    pz = (jet[:, 0, JET_PT] * np.sinh(jet[:, 0, JET_ETA])
          + jet[:, 1, JET_PT] * np.sinh(jet[:, 1, JET_ETA]))
    return np.sqrt(np.abs(px**2 + py**2 + pz**2 - e**2))


def build_point_cloud(X, jet, num_kin, max_part, coords, jet_id):
    """Flatten (n, 2, P, 7) two-jet constituents into one (n, K, F) point cloud.

    Returns float32 of shape (n, max_part, num_kin + n_id), where n_id is 2 for
    a true one-hot jet tag and 1 for a single binary column.
    """
    X = X * (X[:, :, :, IDX_LOGPT : IDX_LOGPT + 1] > 0.0)   # pT > 1 GeV
    mask = X[:, :, :, IDX_LOGPTREL] != 0                    # (n, 2, P)

    feats = X[:, :, :, :num_kin].copy()

    if coords == "event":
        # Re-express the angles in the event frame: eta becomes absolute, phi
        # becomes an offset from the LEADING jet's phi (so the whole cloud is
        # still rotationally aligned, but the two jets no longer sit on top of
        # each other at the origin). Without this, a jet-1 and a jet-2
        # constituent that are physically far apart get d_eta ~ 0, which feeds
        # nonsense into the pairwise interaction terms.
        feats[:, :, :, IDX_DETA] = X[:, :, :, IDX_DETA] + jet[:, :, None, JET_ETA]
        dphi = (X[:, :, :, IDX_DPHI] + jet[:, :, None, JET_PHI]
                - jet[:, 0:1, None, JET_PHI])
        feats[:, :, :, IDX_DPHI] = (dphi + np.pi) % (2 * np.pi) - np.pi

    n, _, n_part, _ = X.shape
    if jet_id == "onehot":
        tag = np.zeros((n, 2, n_part, 2), dtype=X.dtype)
        tag[:, 0, :, 0] = 1.0   # particle belongs to jet 1
        tag[:, 1, :, 1] = 1.0   # particle belongs to jet 2
    else:
        tag = np.zeros((n, 2, n_part, 1), dtype=X.dtype)
        tag[:, 1, :, 0] = 1.0

    out = np.concatenate([feats, tag], axis=-1) * mask[..., None]

    # Merge the two jets into a single sequence, dropping padding. Ordering by
    # descending pT means the truncation to max_part throws away the softest
    # constituents rather than an arbitrary tail; the model itself is
    # permutation equivariant, so the order carries no other meaning.
    n_feat = out.shape[-1]
    flat = out.reshape(n, 2 * n_part, n_feat)
    key = np.where(mask.reshape(n, 2 * n_part),
                   X[:, :, :, IDX_LOGPT].reshape(n, 2 * n_part), -np.inf)
    order = np.argsort(-key, axis=1)[:, :max_part]
    flat = np.take_along_axis(flat, order[:, :, None], axis=1)

    if flat.shape[1] < max_part:
        pad = np.zeros((n, max_part - flat.shape[1], n_feat), dtype=flat.dtype)
        flat = np.concatenate([flat, pad], axis=1)
    return flat.astype(np.float32), mask.sum(axis=(1, 2))


def build_global(jet, mjj):
    """Event-level conditioning: [mjj_scaled, pT1, m1, pT2, m2].

    Read only when training with `--conditional --num-cond 5`. Note mjj is the
    first column on purpose: for a CWoLa/bump-hunt setup that is the variable
    you condition on, and for the idealized classifier you should leave
    --conditional off entirely so the label does not leak through it.
    """
    return np.stack([
        prep_mjj(mjj),
        jet[:, 0, JET_PT], jet[:, 0, JET_MASS],
        jet[:, 1, JET_PT], jet[:, 1, JET_MASS],
    ], axis=-1).astype(np.float32)


def convert_file(src, dest, label, args, start=0, stop=None):
    """Stream events [start, stop) of `src` into a new OmniLearned-schema h5.

    Chunked because train_background_SR_extended.h5 is 14.9 GB of float64 and
    does not fit in memory the way torch_lhco.py's constructor loads it.
    """
    with h5py.File(src, "r") as f:
        n_total = f["data"].shape[0]
        stop = n_total if stop is None else min(stop, n_total)
        if args.nevts > 0:
            stop = min(stop, start + args.nevts)
        n_out = max(0, stop - start)
        if n_out == 0:
            print(f"  skip {Path(src).name}: no events in range")
            return 0

        has_mjj = "pid" in f
        n_id = 2 if args.jet_id == "onehot" else 1
        n_feat = args.num_kin + n_id

        dest.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(dest, "w") as g:
            d = g.create_dataset(
                "data", shape=(n_out, args.max_part, n_feat), dtype="float32",
                chunks=(min(256, n_out), args.max_part, n_feat),
                compression="gzip", compression_opts=1,
            )
            p = g.create_dataset("pid", shape=(n_out,), dtype="int64")
            gl = g.create_dataset("global", shape=(n_out, 5), dtype="float32")
            mj = g.create_dataset("mjj", shape=(n_out,), dtype="float32")

            n_trunc = 0
            for lo in range(start, stop, args.chunk):
                hi = min(lo + args.chunk, stop)
                X = f["data"][lo:hi]
                jet = f["jet"][lo:hi]
                mjj = f["pid"][lo:hi] if has_mjj else get_dimass(jet)

                cloud, mult = build_point_cloud(
                    X, jet, args.num_kin, args.max_part, args.coords, args.jet_id
                )
                n_trunc += int((mult > args.max_part).sum())

                d[lo - start : hi - start] = cloud
                p[lo - start : hi - start] = label
                gl[lo - start : hi - start] = build_global(jet, mjj)
                mj[lo - start : hi - start] = mjj.astype(np.float32)

            g.attrs["source"] = str(src)
            g.attrs["label"] = label
            g.attrs["coords"] = args.coords
            g.attrs["jet_id"] = args.jet_id
            g.attrs["num_kin"] = args.num_kin
            g.attrs["feature_names"] = ",".join(
                ["d_eta", "d_phi", "log_pt_rel", "log_pt",
                 "log_e_rel", "log_e", "dr"][: args.num_kin]
                + (["is_jet1", "is_jet2"] if n_id == 2 else ["jet_id"])
            )

    msg = f"  {Path(src).name} -> {dest.name}: {n_out} events, label {label}"
    if n_trunc:
        msg += f"  ({n_trunc} events truncated at max_part={args.max_part})"
    print(msg)
    return n_out


def plan(args):
    """Return {split: [(src_filename, label, start_frac, stop_frac), ...]}.

    idealized: truth labels, 0 = background, 1 = signal. Fully supervised, the
        setup LHCO_ANOMALY_DETECTION_PROCEDURE.md's Pipeline A/B assume.
    cwola: weak supervision, 0 = background template (the *_extended sample),
        1 = "data" (background + injected signal). This is what lgatr_train.py
        trains on, so it is the apples-to-apples comparison against L-GATr.
    """
    if args.mode == "idealized":
        return {
            "train": [("train_background_SR.h5", 0, 0.0, 1.0),
                      ("train_signal_SR.h5", 1, 0.0, 1.0)],
            "val":   [("val_background_SR.h5", 0, 0.0, 1.0 - args.test_frac),
                      ("val_signal_SR.h5", 1, 0.0, 1.0 - args.test_frac)],
            "test":  [("val_background_SR.h5", 0, 1.0 - args.test_frac, 1.0),
                      ("val_signal_SR.h5", 1, 1.0 - args.test_frac, 1.0)],
        }
    return {
        "train": [("train_background_SR_extended.h5", 0, 0.0, 1.0),
                  ("train_background_SR.h5", 1, 0.0, 1.0),
                  ("train_signal_SR.h5", 1, 0.0, None)],
        "val":   [("val_background_SR_extended.h5", 0, 0.0, 1.0 - args.test_frac),
                  ("val_background_SR.h5", 1, 0.0, 1.0 - args.test_frac),
                  ("val_signal_SR.h5", 1, 0.0, None)],
        "test":  [("val_background_SR_extended.h5", 0, 1.0 - args.test_frac, 1.0),
                  ("val_background_SR.h5", 1, 1.0 - args.test_frac, 1.0),
                  ("val_signal_SR.h5", 1, None, None)],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="/global/cfs/cdirs/m3246/vmikuni/PET/LHCO",
                    help="Directory holding the OmniLearn LHCO h5 files")
    ap.add_argument("--out-dir", required=True,
                    help="Dataset root; files land in <out-dir>/<name>/{train,val,test}")
    ap.add_argument("--name", default="custom",
                    help="Dataset directory name, passed to omnilearned as --dataset. "
                         "Must be one of load_data's supported_datasets "
                         "(dataloader.py:381); 'custom' is the generic slot")
    ap.add_argument("--mode", choices=["idealized", "cwola"], default="idealized",
                    help="Label scheme, see plan() docstring")
    ap.add_argument("--nsig", type=int, default=1000,
                    help="cwola only: signal events injected into the data class")
    ap.add_argument("--num-kin", type=int, default=4,
                    help="Kinematic features kept, from the front of the 7 raw ones. "
                         "4 matches omnilearned's --num-feat default")
    ap.add_argument("--max-part", type=int, default=320,
                    help="Point cloud length. Measured max over both jets is 303")
    ap.add_argument("--coords", choices=["jet", "event"], default="jet",
                    help="'jet' keeps d_eta/d_phi wrt each constituent's own jet "
                         "(matches the pretrain input distribution); 'event' puts "
                         "both jets in a common frame (physically correct pairwise "
                         "distances, but off-distribution for a pretrained backbone)")
    ap.add_argument("--jet-id", choices=["onehot", "binary"], default="onehot",
                    help="'onehot' writes two columns (is_jet1, is_jet2); 'binary' "
                         "writes one 0/1 column, matching torch_lhco.py")
    ap.add_argument("--test-frac", type=float, default=0.5,
                    help="Fraction of the val_* files held out as the test split")
    ap.add_argument("--nevts", type=int, default=-1,
                    help="Cap events per source file, for smoke tests")
    ap.add_argument("--chunk", type=int, default=20000,
                    help="Events read per h5 slice")
    args = ap.parse_args()

    root = Path(args.out_dir) / args.name
    print(f"mode={args.mode}  coords={args.coords}  jet_id={args.jet_id}  "
          f"num_kin={args.num_kin}  max_part={args.max_part}")
    print(f"writing to {root}/{{train,val,test}}\n")

    totals = {}
    for split, entries in plan(args).items():
        print(f"[{split}]")
        n = 0
        for fname, label, lo_frac, hi_frac in entries:
            src = Path(args.data_dir) / fname
            if not src.is_file():
                print(f"  MISSING {src}, skipping")
                continue
            with h5py.File(src, "r") as f:
                n_avail = f["data"].shape[0]

            if "signal" in fname and args.mode == "cwola":
                # The injected signal is a fixed count, not a fraction: train
                # takes the first 90% of --nsig, val the next 5%, test the rest,
                # mirroring lgatr_train.py's 90/10 train/test signal split.
                n_tr = int(args.nsig * 0.9)
                n_va = int(args.nsig * 0.05)
                bounds = {"train": (0, n_tr),
                          "val": (n_tr, n_tr + n_va),
                          "test": (n_tr + n_va, args.nsig)}[split]
                start, stop = bounds
            else:
                start = int(n_avail * lo_frac)
                stop = int(n_avail * hi_frac)

            dest = root / split / f"{src.stem}_{split}.h5"
            n += convert_file(src, dest, label, args, start=start, stop=stop)
        totals[split] = n
        print()

    n_id = 2 if args.jet_id == "onehot" else 1
    print("Totals:", ", ".join(f"{k}={v}" for k, v in totals.items()))
    print(f"\nBuild the index cache:\n"
          f"  omnilearned dataloader -d {args.name} -f {args.out_dir}\n")
    print(f"Train (feature layout needs --num-feat {args.num_kin} "
          f"--use-add --num-add {n_id}):\n"
          f"  omnilearned train --dataset {args.name} --path {args.out_dir} \\\n"
          f"    --mode classifier --num-classes 2 \\\n"
          f"    --num-feat {args.num_kin} --use-add --num-add {n_id} \\\n"
          f"    --size small --interaction --local-interaction \\\n"
          f"    --batch 128 --iterations 1000 --epoch 50 --save-tag {args.name}_{args.mode}")


if __name__ == "__main__":
    main()
