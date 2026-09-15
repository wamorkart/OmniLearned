"""Build a small, held-out, PURE-signal eval file for a given convert_lhco.py
--nsig run, from signal events never injected into that run's training data.
Paired with an EXISTING evaluate run's pid==0 rows (already pure background,
already held out -- see tools/metrics/compute_roc_sic_lhco.py), this gives a
genuinely truth-labeled test set for a real ROC/AUC/max-SIC, with no
retraining: run `omnilearned evaluate` on the file this writes, using the
already-fine-tuned checkpoint for that nsig.

The dataloader just globs every *.h5 under <path>/<dataset>/<split>/ and reads
each file's own `pid` column, so this single file is a complete, valid
`--dataset lhco_ad --dataset-type test` input on its own -- no companion file
needed.

Must match the EXACT --seed/--val-frac/--test-frac the target nsig run used
(defaults match convert_lhco.py's), since this replays that run's RNG stream
to recover which signal indices it actually injected.

`salloc -N 1 -C cpu -q interactive -t 01:00:00 -A m3246` before running.
Run (from this directory): python build_lhco_eval_signal.py --nsig 500
"""

import argparse

import h5py
import numpy as np

from convert_lhco import (
    OUT_DIR,
    BACKGROUND_FILES,
    SIGNAL_FILES,
    apply_pt_cut,
    global_zscore_stats,
    load_source,
    rows_from_selection,
    split_indices,
)


def replay_main_rng(nsig, seed, val_frac, test_frac):
    """Replay main()'s RNG sequence -- background_SR split+stats fit, then
    signal nsig-capping, sharing one rng stream in that order -- to recover
    the exact same background-train stats and 'used' signal indices that
    nsig run produced. Loads all of background_SR (~4GB)."""
    rng = np.random.default_rng(seed)

    bkg_jet, bkg_data, bkg_mjj = load_source(BACKGROUND_FILES)
    bkg_data, bkg_valid = apply_pt_cut(bkg_data)
    bkg_rows = rows_from_selection(
        bkg_jet, bkg_data, bkg_valid, bkg_mjj, np.ones(bkg_jet.shape[0], dtype=bool), pid_label=1
    )
    bkg_split = split_indices(bkg_rows["data"].shape[0], val_frac, test_frac, rng)
    mean, std = global_zscore_stats(bkg_rows["global"][bkg_split["train"]])

    n_sig_total = sum(h5py.File(p)["jet"].shape[0] for p in SIGNAL_FILES)
    used = np.zeros(n_sig_total, dtype=bool)
    if nsig is not None and nsig < n_sig_total:
        used[rng.choice(n_sig_total, size=nsig, replace=False)] = True
    else:
        used[:] = True  # nsig=None ("nsig_all"): nothing left over

    return mean, std, ~used


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nsig", type=int, required=True, help="Must match an existing nsig_<N> run")
    ap.add_argument("--seed", type=int, default=0, help="Must match that run's --seed")
    ap.add_argument("--val-frac", type=float, default=0.1, help="Must match that run's --val-frac")
    ap.add_argument("--test-frac", type=float, default=0.1, help="Must match that run's --test-frac")
    args = ap.parse_args()

    mean, std, unused_mask = replay_main_rng(args.nsig, args.seed, args.val_frac, args.test_frac)
    if not unused_mask.any():
        raise SystemExit(f"No leftover signal for --nsig {args.nsig}: all of it was injected.")
    print(f"{unused_mask.sum():,} / {unused_mask.size:,} signal events never injected at nsig={args.nsig}")

    jet, data, mjj = load_source(SIGNAL_FILES)
    data, valid = apply_pt_cut(data)
    rows = rows_from_selection(jet, data, valid, mjj, unused_mask, pid_label=1)
    rows["global"] = (rows["global"] - mean) / std

    nsig_dir = f"nsig_{args.nsig}" if args.nsig is not None else "nsig_all"
    out_path = OUT_DIR / nsig_dir / "eval_signal" / "lhco_ad" / "test" / "leftover_signal.h5"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        for key, arr in rows.items():
            f.create_dataset(key, data=arr)
    print(f"Wrote {rows['data'].shape[0]:,} pure-signal events to {out_path}")


if __name__ == "__main__":
    main()
