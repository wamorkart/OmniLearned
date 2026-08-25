"""
Pipeline: 'fastjet_awkward.ipynb' 
                    --> 'preprocess_lhco.py' from OmniLearn 
                    --> convert_lhco.py

`salloc -N 1 -C cpu -q interactive -t 01:00:00 -A m3246` before running.

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
                                                                    
OmniLearned expects one row per jet (not per dijet event):
    data   (N, M, 4)            N events, max M particles/jet, 4-5 numbers/particle 
                                                                    = (delta_eta, delta_phi, log_pt, log_e, optionally PID)
    pid    (N,)                 N events, 1 number/event = (0 if data, 1 if pure background)
    global (N, 3)               N events, 3 numbers/event = (log jet pT, log jet mass, multiplicity / 100)

Idealized CWoLa setup: no generative model is trained, background sampled from predefined distribution.

Builds two independent pools, each with 80/10/10 train/val/test split:
    lhco_ad_data  background (all) + signal (up to --nsig injected events),
                  both from the base files, pid=0
    lhco_ad_bkg   the independent extended-background file, pid=1 -- the
                  "true background" stand-in a generator would otherwise have
                  produced, kept fully disjoint from lhco_ad_data so the
                  classifier isn't trained to separate a sample from itself


"""

import argparse
from pathlib import Path

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
OUT_DIR = REPO_ROOT / "LHCO"
INPUT_DIR = Path("/global/cfs/cdirs/m3246/mbenyas/OmniLearn/LHCO")

BACKGROUND_FILES = [INPUT_DIR / "train_background_SR.h5", INPUT_DIR / "val_background_SR.h5"]
SIGNAL_FILES = [INPUT_DIR / "train_signal_SR.h5", INPUT_DIR / "val_signal_SR.h5"]
BACKGROUND_EXTENDED_FILES = [
    INPUT_DIR / "train_background_SR_extended.h5",
    INPUT_DIR / "val_background_SR_extended.h5",
]


def load_source(paths):
    """Concatenate jet/data across files (source's train+val; redoing the split in this script)."""
    jets, datas = [], []
    for path in paths:
        with h5py.File(path, "r") as f:
            jets.append(f["jet"][:])
            datas.append(f["data"][:])
    return np.concatenate(jets), np.concatenate(datas)


def apply_pt_cut(data):
    """Zero constituents with log(pT) <= 0 (pT <= 1 GeV); padding (already 0) goes too."""
    valid = data[..., 3] > 0.0
    return data * valid[..., None], valid


def build_rows(paths, pid_label, nsig=None, rng=None):
    """Load one sample, apply the pT cut, optionally cap events at `nsig`
    (sampled w/o replacement), and flatten each event into 2 per-jet rows.
    pid_label is the classifier's training label (0=data, 1=background)."""
    jet, data = load_source(paths)
    data, valid = apply_pt_cut(data)

    n = jet.shape[0]
    evt_sel = np.ones(n, dtype=bool)
    if nsig is not None and nsig < n:
        evt_sel[:] = False
        evt_sel[rng.choice(n, size=nsig, replace=False)] = True
    pid = np.full(evt_sel.sum(), pid_label, dtype=np.int64)

    data_rows, global_rows = [], []
    for j in range(2):
        deta, dphi, log_pt, log_e = (data[evt_sel, j, :, k] for k in (0, 1, 3, 5)) # extract 4 of 7 features
        data_rows.append(np.stack([deta, dphi, log_pt, log_e], -1).astype(np.float32))

        jet_pt = jet[evt_sel, j, 0]
        jet_m = np.clip(jet[evt_sel, j, 3], 1e-3, None)  # avoid log(0)
        jet_mult = valid[evt_sel, j].sum(-1)
        global_rows.append(
            np.stack([np.log(jet_pt), np.log(jet_m), jet_mult / 100.0], -1).astype(np.float32)
        )

    return {
        "data": np.concatenate(data_rows),
        "pid": np.concatenate([pid, pid]),
        "global": np.concatenate(global_rows),
    }


def split_indices(n, val_frac, test_frac, rng):
    """Shuffle [0, n) into train/val/test index arrays with exact counts."""
    perm = rng.permutation(n)
    n_val, n_test = int(round(n * val_frac)), int(round(n * test_frac))
    return {"val": perm[:n_val], "test": perm[n_val : n_val + n_test], "train": perm[n_val + n_test :]}


def write_pool(name, out_dir, rows, val_frac, test_frac, rng):
    """Split `rows` into train/val/test and write each as out_dir/name/<split>/name.h5."""
    n = rows["data"].shape[0]
    counts = {}
    for split, idx in split_indices(n, val_frac, test_frac, rng).items():
        path = Path(out_dir) / name / split / f"{name}.h5"
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

    bkg = build_rows(BACKGROUND_FILES, pid_label=0)
    sig = build_rows(SIGNAL_FILES, pid_label=0, nsig=args.nsig, rng=rng)
    data_rows = {k: np.concatenate([bkg[k], sig[k]]) for k in bkg}
    bkg_rows = build_rows(BACKGROUND_EXTENDED_FILES, pid_label=1)

    for name, rows in (("lhco_ad_data", data_rows), ("lhco_ad_bkg", bkg_rows)):
        counts = write_pool(name, OUT_DIR, rows, args.val_frac, args.test_frac, rng)
        print(f"{name}: {counts}, total={sum(counts.values())}")


if __name__ == "__main__":
    main()
