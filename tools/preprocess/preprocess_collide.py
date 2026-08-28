#!/usr/bin/env python
"""Preprocess fastmachinelearning/collide-1m (HL-LHC mu=200 Delphes events,
https://huggingface.co/datasets/fastmachinelearning/collide-1m) into this
project's standard classification h5 schema: `data` (N, K, F) padded particle
point cloud, `pid` (N,) integer class label, `global` (N, G) event-level
conditioning -- the same schema `HEPDataset`/`load_data` in
src/omnilearned/dataloader.py already read for every other dataset, so
training/eval with `--dataset collide --arch deep-sets` needs no other code
change once files land under `<path>/collide/{train,val,test}/*.h5`.

Config-driven per user request (2026-08-21): --task selects a default
process/label mapping (`multiclass` or `binary_ad`), or pass --processes to
override with an explicit list. Point-cloud choice and feature layout below
are the result of scoping against the real dataset, not the raw schema --
see the notes inline.

Requires (NOT part of the live omnilearned-clean/omnilearned-fpga envs --
install into a throwaway/dedicated env first, same convention as
preprocess_omnifold.py's "conda activate energyflow" note):
    pip install pyarrow fsspec aiohttp huggingface_hub

Usage:
    python preprocess_collide.py --task multiclass --out-dir /pscratch/sd/t/twamorka/omnilearned/datasets \
        --k-part 32 --events-per-process 20000

    python preprocess_collide.py --task binary_ad --out-dir ... --events-per-process 20000
"""

import argparse
import os
import random
from pathlib import Path

import fsspec
import h5py
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import HfApi

HF_REPO = "fastmachinelearning/collide-1m"
HF_RESOLVE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"

# --- Particle collection choice -------------------------------------------
# Scoping (2026-08-21) pulled real rows from the `minbias` process: many
# events have ZERO reconstructed JetPuppiAK4 jets (Delphes finds no jet above
# threshold in a lot of pure-pileup events), which rules out a "leading jet's
# constituents" point cloud -- it would be empty for a large fraction of the
# background class. FullReco_PUPPIPart is a much better fit: it's already the
# small, PUPPI-cleaned, per-event candidate collection (observed ~16-60
# particles/event, vs ~3800-4200 raw FullReco_PFCand/event at mu=200) that
# fastmachinelearning's own COLLIDE-1m_streaming_classifier_tutorial.ipynb
# also builds its reference 6-class MLP from -- so this follows their own
# precedent rather than inventing a new convention.
# NOTE: the real per-process parquet files (checked against the actual
# minbias/*.parquet, not the HF-auto-converted refs/convert/parquet mirror,
# which has a stale/different schema including a PuppiW field that does NOT
# exist on the real files) expose PT/Eta/Phi/Charge/Mass/PID/D0/DZ/ErrorD0/
# ErrorDZ/fUniqueID for PUPPIPart -- no PuppiW. Verify with a fresh schema
# read (see scoping notes) before trusting this if HF re-converts the repo.
PART_PREFIX = "FullReco_PUPPIPart"
PART_FIELDS = ["PT", "Eta", "Phi", "Charge", "Mass", "PID"]

MET_FIELDS = ["FullReco_PUPPIMET_MET", "FullReco_PUPPIMET_Phi"]
PV_FIELDS = ["FullReco_PrimaryVertex_Z", "FullReco_PrimaryVertex_SumPT2"]

# --- Feature layout ---------------------------------------------------------
# DeepSetsBody.forward (src/omnilearned/network.py) hardcodes
# `mask = (x[:, :, 2:3] != 0)` -- feature index 2 MUST be nonzero for every
# real particle and exactly 0 for padding. Raw pT (GeV, always > 0 for a
# reconstructed candidate) is the safest choice, matching the "always-positive
# kinematic quantity at index 2" convention already used for the mask column
# elsewhere in this project (see collate_point_cloud in dataloader.py).
# Padding is enforced explicitly below via a per-event particle-count mask,
# not by relying on any feature happening to equal 0 naturally.
FEATURE_NAMES = ["eta", "phi", "pt", "log_pt", "charge", "mass", "pid_raw"]
NUM_FEAT = len(FEATURE_NAMES)

# NOTE on PID: FullReco_PUPPIPart_PID is stored as int8. Cross-checked against
# real events: values like -92 decode as 2212 (proton) truncated mod 256 into
# signed int8 (2212 % 256 = 164 -> -92 as int8), and +-45 decode as the two
# charged-pion codes (+-211 wrapped the same way). This wraparound is LOSSY
# for anything with |PDG code| >= 128 (most hadrons alias with other hadrons),
# so `pid_raw` here is included as an auxiliary continuous feature only --
# it is NOT wired through DeepSetsBody's `use_pid`/pid_embed categorical path
# (that path expects a small clean vocabulary; this field doesn't provide
# one). Charge is stored separately and is NOT truncated -- treat charge, not
# PID, as the reliable particle-identity feature here.

# --- Default process -> label mappings -------------------------------------
# multiclass: same 6 physics families as fastmachinelearning's own reference
# notebook (COLLIDE-1m_streaming_classifier_tutorial.ipynb), so results are
# comparable to their published baseline.
MULTICLASS_DEFAULT = {
    "DYJetsToLL_13TeV-madgraphMLM-pythia8": 0,   # DY
    "QCD_HT50toInf": 1,                          # QCD
    "VBFHtautau": 2,                             # SingleHiggs
    "tt0123j_5f_ckm_LO_MLM_leptonic": 3,          # top
    "WZ_semileptonic": 4,                         # diboson
    "HH_bbtautau": 5,                             # diHiggs
}
MULTICLASS_NAMES = ["DY", "QCD", "SingleHiggs", "top", "diboson", "diHiggs"]

# binary_ad: signal-region-style class (rare diHiggs final state) vs a
# QCD/minbias background mix, mirroring the aspen_bsm_ad_sb/ad_sr naming
# already stubbed (unused) in dataloader.py's supported_datasets list.
BINARY_AD_DEFAULT = {
    "HH_bbtautau": 1,
    "QCD_HT50toInf": 0,
    "minbias": 0,
}
BINARY_AD_NAMES = ["background", "signal"]


def list_process_parquet_files(process_folder):
    api = HfApi()
    files = api.list_repo_files(repo_id=HF_REPO, repo_type="dataset")
    prefix = f"{process_folder.strip('/')}/"
    return sorted(f for f in files if f.startswith(prefix) and f.endswith(".parquet"))


def _pad_particles(part_arrays, k_part):
    """part_arrays: dict field -> list of python lists (one per event, ragged).
    Returns (n_events, k_part, NUM_FEAT) float32, sorted by pT descending,
    zero-padded/truncated to k_part.
    """
    n_events = len(part_arrays["PT"])
    out = np.zeros((n_events, k_part, NUM_FEAT), dtype=np.float32)
    for i in range(n_events):
        pt = np.asarray(part_arrays["PT"][i], dtype=np.float32)
        if pt.size == 0:
            continue
        order = np.argsort(-pt)[:k_part]
        n_use = order.size

        eta = np.asarray(part_arrays["Eta"][i], dtype=np.float32)[order]
        phi = np.asarray(part_arrays["Phi"][i], dtype=np.float32)[order]
        pt_use = pt[order]
        charge = np.asarray(part_arrays["Charge"][i], dtype=np.float32)[order]
        mass = np.asarray(part_arrays["Mass"][i], dtype=np.float32)[order]
        pid_raw = np.asarray(part_arrays["PID"][i], dtype=np.float32)[order]

        out[i, :n_use, 0] = eta
        out[i, :n_use, 1] = phi
        out[i, :n_use, 2] = pt_use
        out[i, :n_use, 3] = np.log(np.clip(pt_use, 1e-3, None))
        out[i, :n_use, 4] = charge
        out[i, :n_use, 5] = mass
        out[i, :n_use, 6] = pid_raw
        # explicit padding mask, independent of any single column's natural
        # value -- zero every feature beyond n_use (already true from
        # np.zeros, kept here as documentation of the invariant, not dead code:
        # a future feature added at the end must respect this too).
        out[i, n_use:, :] = 0.0
    return out


def _event_globals(tbl_dict, part_pt, n_events):
    g = np.zeros((n_events, 5), dtype=np.float32)
    for i in range(n_events):
        met = tbl_dict["FullReco_PUPPIMET_MET"][i]
        met_phi = tbl_dict["FullReco_PUPPIMET_Phi"][i]
        pv_z = tbl_dict["FullReco_PrimaryVertex_Z"][i]
        pv_sumpt2 = tbl_dict["FullReco_PrimaryVertex_SumPT2"][i]
        g[i, 0] = met[0] if len(met) else 0.0
        g[i, 1] = met_phi[0] if len(met_phi) else 0.0
        g[i, 2] = pv_z[0] if len(pv_z) else 0.0
        g[i, 3] = pv_sumpt2[0] if len(pv_sumpt2) else 0.0
        g[i, 4] = float(len(part_pt[i]))
    return g


def load_process_events(process_folder, max_events, seed):
    """Stream parquet files for one process folder, column-pruned (never pulls
    FullReco_PFCand_*, which is ~200x larger per event than PUPPIPart), and
    return (particles (n,k,F) unpacked later, globals, n_events).
    """
    files = list_process_parquet_files(process_folder)
    if not files:
        raise ValueError(f"No parquet files found under '{process_folder}' in {HF_REPO}")
    rng = random.Random(seed)
    rng.shuffle(files)

    cols = [f"{PART_PREFIX}_{f}" for f in PART_FIELDS] + MET_FIELDS + PV_FIELDS

    part_arrays = {f: [] for f in PART_FIELDS}
    glob_rows = {c: [] for c in MET_FIELDS + PV_FIELDS}
    n_collected = 0

    for fname in files:
        if n_collected >= max_events:
            break
        url = f"{HF_RESOLVE}/{fname}"
        fs = fsspec.filesystem("http")
        with fs.open(url, "rb") as fh:
            pf = pq.ParquetFile(fh)
            for rg in range(pf.num_row_groups):
                if n_collected >= max_events:
                    break
                tbl = pf.read_row_group(rg, columns=cols).to_pydict()
                n_rows = len(tbl[f"{PART_PREFIX}_PT"])
                take = min(n_rows, max_events - n_collected)
                for f in PART_FIELDS:
                    part_arrays[f].extend(tbl[f"{PART_PREFIX}_{f}"][:take])
                for c in MET_FIELDS + PV_FIELDS:
                    glob_rows[c].extend(tbl[c][:take])
                n_collected += take

    print(f"  {process_folder}: collected {n_collected} events from "
          f"{len(files)} available parquet file(s)")
    return part_arrays, glob_rows, n_collected


def write_h5(out_path, data, pid, glob):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("data", data=data, compression="gzip", compression_opts=4)
        f.create_dataset("pid", data=pid)
        f.create_dataset("global", data=glob, compression="gzip", compression_opts=4)
    print(f"Wrote {out_path}: data{data.shape}, pid{pid.shape}, global{glob.shape}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["multiclass", "binary_ad"], default="multiclass")
    p.add_argument("--processes", type=str, default=None,
                    help="Override default mapping: comma-separated "
                         "'process_folder:label_int' pairs, e.g. "
                         "'minbias:0,HH_bbtautau:1'")
    p.add_argument("--out-dir", type=str, required=True,
                    help="Files land in <out-dir>/collide/{train,val,test}/data.h5")
    p.add_argument("--k-part", type=int, default=32,
                    help="Particles per event to keep (top-K by pT); real "
                         "PUPPIPart events observed with 16-60 particles, so "
                         "32 covers the typical case with room to spare")
    p.add_argument("--events-per-process", type=int, default=20000)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.processes:
        proc_map = {}
        for pair in args.processes.split(","):
            folder, label = pair.split(":")
            proc_map[folder.strip()] = int(label)
    elif args.task == "multiclass":
        proc_map = MULTICLASS_DEFAULT
    else:
        proc_map = BINARY_AD_DEFAULT

    print(f"Task: {args.task}; processes: {proc_map}")

    all_data, all_pid, all_glob = [], [], []
    for folder, label in proc_map.items():
        part_arrays, glob_rows, n = load_process_events(
            folder, args.events_per_process, seed=args.seed
        )
        if n == 0:
            print(f"  WARNING: 0 events collected for '{folder}', skipping")
            continue
        data = _pad_particles(part_arrays, args.k_part)
        glob = _event_globals(glob_rows, part_arrays["PT"], n)
        all_data.append(data)
        all_pid.append(np.full(n, label, dtype=np.int64))
        all_glob.append(glob)

    data = np.concatenate(all_data, axis=0)
    pid = np.concatenate(all_pid, axis=0)
    glob = np.concatenate(all_glob, axis=0)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(pid))
    data, pid, glob = data[perm], pid[perm], glob[perm]

    n = len(pid)
    n_test = int(n * args.test_frac)
    n_val = int(n * args.val_frac)
    n_train = n - n_val - n_test

    out_dir = Path(args.out_dir) / "collide"
    write_h5(out_dir / "train" / "data.h5", data[:n_train], pid[:n_train], glob[:n_train])
    write_h5(
        out_dir / "val" / "data.h5",
        data[n_train:n_train + n_val], pid[n_train:n_train + n_val], glob[n_train:n_train + n_val],
    )
    write_h5(
        out_dir / "test" / "data.h5",
        data[n_train + n_val:], pid[n_train + n_val:], glob[n_train + n_val:],
    )
    print(f"Total: {n} events, {n_train} train / {n_val} val / {n - n_train - n_val} test")


if __name__ == "__main__":
    main()
