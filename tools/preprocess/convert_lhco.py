"""
Pipeline: 'fastjet_awkward.ipynb' 
                    --> 'preprocess_lhco.py' from OmniLearn 
                    --> convert_lhco.py

`salloc -N 1 -C cpu -q interactive -t 01:00:00 -A m3246` before running.
Run as 'python convert_lhco.py --nsig {NSIG}

`fastjet_awkward.ipynb` clusters raw LHCO events as follows:
    jet_data     (N, 2, 4)      N events, 2 jets/event, 4 numbers/jet = (pt, eta, phi, m)
    constituents (N, 2, P, 3)   N events, 2 jets/event, P particles/jet, 3 numbers/particle = (pt, eta, phi)
    mask         (N, 2, P, 1)   N events, 2 jets/event, P particles/jet, 1 number/particle 
                                                                                = (1 if constituent is real, 0 if padding)
saved separately for background, signal, and an independent extended-background
sample as processed_data_{background,signal,background_extended}.h5.

Then 'preprocess_lhco.py' turns files into following format:
    jet  (N, 2, 5)      N events, 2 jets/event, 5 numbers/jet = (pt, eta, phi, mass, multiplicity)
    data (N, 2, P, 7)   N events, 2 jets/event, P particles/jet, 7 numbers/particle
                                                                    = (delta_eta, delta_phi, log(1-pT_rel), log_pt, log(1-E_rel), log_e, deltaR)
saved as {train,val}_{background,signal}_{SB,SR,SR_extended}.h5
                                                                    
OmniLearned to take in one row per EVENT, both jets' particles merged together
(so a one-hot is needed to mark which jet each particle came from):
    data   (N, 2P, 6)           N events, max 2P merged particles, 6 numbers/particle
                                                                    = (delta_eta, delta_phi, log_pt, log_e,
                                                                       onehot_jet0, onehot_jet1)
    pid    (N,)                 N events, 1 number/event = (0 if pure background, 1 if data)
    global (N, 11)              N events, 11 numbers/event = mjj, then per jet (log pT, eta, phi, log mass,
                                                                    multiplicity / 100), jet0 then jet1
                                                                    -- jet-level observables per torch_lhco.py's `jet` array

Idealized CWoLa setup: no generative model is trained, background sampled from predefined distribution.

Builds dataset "lhco_ad" with two separate files per split (data.h5,
bkg.h5). Each file gets an independent 80/10/10 train/val/test split:
    bkg.h5   the independent extended-background file, pid=0 -- the
             "true background" stand-in a generator would otherwise have
             produced, kept fully disjoint from data.h5 so the classifier
             isn't trained to separate a sample from itself
    data.h5  background (all) + signal (up to --nsig injected events), pid=1


"""

import argparse
from pathlib import Path

import h5py
import numpy as np

OUT_DIR = Path("/global/cfs/cdirs/m3246/mbenyas/OmniLearned_distillation/LHCO")
INPUT_DIR = Path("/global/cfs/cdirs/m3246/mbenyas/OmniLearn/LHCO")

BACKGROUND_FILES = [INPUT_DIR / "train_background_SR.h5", INPUT_DIR / "val_background_SR.h5"]
SIGNAL_FILES = [INPUT_DIR / "train_signal_SR.h5", INPUT_DIR / "val_signal_SR.h5"]
BACKGROUND_EXTENDED_FILES = [
    INPUT_DIR / "train_background_SR_extended.h5",
    INPUT_DIR / "val_background_SR_extended.h5",
]


def load_source(paths):
    """Concatenate jet/data/mjj across files (source's train+val; redoing the
    split in this script). Source's own `pid` field holds mjj (verified:
    values sit in the 3300-3700 GeV SR window)."""
    jets, datas, mjjs = [], [], []
    for path in paths:
        with h5py.File(path, "r") as f:
            jets.append(f["jet"][:])
            datas.append(f["data"][:])
            mjjs.append(f["pid"][:])
    return np.concatenate(jets), np.concatenate(datas), np.concatenate(mjjs)


def apply_pt_cut(data):
    """Zero constituents with log(pT) <= 0 (pT <= 1 GeV); padding (already 0) goes too."""
    valid = data[..., 3] > 0.0
    return data * valid[..., None], valid


def build_rows(paths, pid_label, nsig=None, rng=None):
    """Load one sample, apply the pT cut, optionally cap events at `nsig`
    (sampled w/o replacement), and merge each event's two jets into one row.
    `data`'s last 2 columns are a one-hot marking which jet each particle
    came from. `global` = mjj + each jet's own (log pT, eta, phi, log mass,
    multiplicity/100). pid_label is the classifier's training label
    (0=background, 1=data), one per event."""
    jet, data, mjj = load_source(paths)
    data, valid = apply_pt_cut(data)

    n = jet.shape[0]
    evt_sel = np.ones(n, dtype=bool)
    if nsig is not None and nsig < n:
        evt_sel[:] = False
        evt_sel[rng.choice(n, size=nsig, replace=False)] = True
    pid = np.full(evt_sel.sum(), pid_label, dtype=np.int64)

    data_blocks = []
    global_cols = [mjj[evt_sel][:, None]]
    for j in range(2):
        deta, dphi, log_pt, log_e = (data[evt_sel, j, :, k] for k in (0, 1, 3, 5)) # extract 4 of 7 features
        onehot = np.zeros(deta.shape + (2,), dtype=np.float32)
        onehot[..., j] = 1.0
        onehot *= valid[evt_sel, j][..., None]  # zero on padding, same as everything else
        data_blocks.append(
            np.concatenate([np.stack([deta, dphi, log_pt, log_e], -1), onehot], -1).astype(
                np.float32
            )
        )

        jet_pt, jet_eta, jet_phi = (jet[evt_sel, j, k] for k in (0, 1, 2))
        jet_m = np.clip(jet[evt_sel, j, 3], 1e-3, None)  # avoid log(0)
        jet_mult = valid[evt_sel, j].sum(-1)
        global_cols.append(
            np.stack([np.log(jet_pt), jet_eta, jet_phi, np.log(jet_m), jet_mult / 100.0], -1)
        )

    return {
        "data": np.concatenate(data_blocks, axis=1),
        "pid": pid,
        "global": np.concatenate(global_cols, axis=-1).astype(np.float32),
    }


def split_indices(n, val_frac, test_frac, rng):
    """Shuffle [0, n) into train/val/test index arrays with exact counts."""
    perm = rng.permutation(n)
    n_val, n_test = int(round(n * val_frac)), int(round(n * test_frac))
    return {"val": perm[:n_val], "test": perm[n_val : n_val + n_test], "train": perm[n_val + n_test :]}


def write_pool(dataset, filename, out_dir, rows, val_frac, test_frac, rng):
    """Split `rows` into train/val/test and write each as
    out_dir/dataset/<split>/filename.h5."""
    n = rows["data"].shape[0]
    counts = {}
    for split, idx in split_indices(n, val_frac, test_frac, rng).items():
        path = Path(out_dir) / dataset / split / f"{filename}.h5"
        path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(path, "w") as f:
            for key, arr in rows.items():
                f.create_dataset(key, data=arr[idx])
        counts[split] = len(idx)
    return counts

def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--val-frac", type=float, default=0.1, help="Val fraction")
    ap.add_argument("--test-frac", type=float, default=0.1, help="Test fraction")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--nsig", type=int, default=None, help="Signal events to inject (None = all)")
    return ap.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    bkg = build_rows(BACKGROUND_FILES, pid_label=1)
    sig = build_rows(SIGNAL_FILES, pid_label=1, nsig=args.nsig, rng=rng)
    data_rows = {k: np.concatenate([bkg[k], sig[k]]) for k in bkg}
    bkg_rows = build_rows(BACKGROUND_EXTENDED_FILES, pid_label=0)

    # one subfolder per --nsig value, so a sweep doesn't overwrite prior runs
    nsig_dir = f"nsig_{args.nsig}" if args.nsig is not None else "nsig_all"
    out_dir = OUT_DIR / nsig_dir

    for filename, rows in (("data", data_rows), ("bkg", bkg_rows)):
        counts = write_pool("lhco_ad", filename, out_dir, rows, args.val_frac, args.test_frac, rng)
        print(f"{nsig_dir}/lhco_ad/{filename}: {counts}, total={sum(counts.values())}")


if __name__ == "__main__":
    main()
