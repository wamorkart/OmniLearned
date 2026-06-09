"""One-time conversion: sharded teacher-logit .npz -> per-source-h5 companions.

The evaluate pipeline writes teacher logits as many sharded .npz files, e.g.
    outputs_<tag>_<dataset>_<split>_chunk<K>of<N>_rank<r>.npz
    outputs_<tag>_<dataset>_<split>_rank<r>.npz
    outputs_<tag>_<dataset>_<split>_<r>.npz
Each holds `logits` (N, C) fp16 and `sample_keys` (N, 2) int64 = (file_idx,
sample_idx), where file_idx is the *local* (per-dataset) index into the
sorted list of source h5 files for that dataset/split.

This script builds, for every source h5 file, a companion
    <out-dir>/<dataset>/<split>/<source_stem>.h5
holding a `teacher_logits` dataset of shape (N_f, C) fp16, indexed by
sample_idx -- so training can read one row on demand (`tf["teacher_logits"][i]`)
exactly the way HEPDataset reads `f["data"][i]`, with no in-RAM array or dict.

Source-file enumeration mirrors dataloader.load_data EXACTLY:
    sorted(glob("*.h5") + glob("*.hdf5"))
over <data-path>/<dataset>/<split>/, so file_idx <-> filename stays aligned.

Companions are initialized to NaN; a coverage pass fails loudly if any row is
left unfilled (a NaN teacher logit would poison the KD loss).
"""

import argparse
import glob
import os
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

PRETRAIN_NAMES = ["atlas", "aspen", "jetclass", "jetclass2", "h1", "cms_qcd", "cms_bsm"]
NAN_SCAN_CHUNK = 1_000_000


def expand_datasets(spec):
    out = []
    for name in spec.split(","):
        name = name.strip()
        if not name:
            continue
        if name == "pretrain":
            out.extend(PRETRAIN_NAMES)
        else:
            out.append(name)
    # de-dup, preserve order
    seen = set()
    return [d for d in out if not (d in seen or seen.add(d))]


def source_files(data_path, dataset, split):
    """Same ordering as dataloader.load_data."""
    d = Path(data_path) / dataset / split
    files = sorted(list(d.glob("*.h5")) + list(d.glob("*.hdf5")))
    return files


def npz_files(npz_dir, tag, dataset, split):
    """All sharded npz for one dataset/split, regardless of chunk/rank suffix.

    The `_*` after `_<split>` matches `chunk<K>of<N>_rank<r>`, `rank<r>`, and
    plain `<r>` naming. `<split>` is anchored by the literal `_<split>_` so e.g.
    `jetclass` does not pick up `jetclass2` files.
    """
    pattern = os.path.join(npz_dir, f"outputs_{tag}_{dataset}_{split}_*.npz")
    return sorted(glob.glob(pattern))


def companions_complete(comp_paths, n_rows, num_classes):
    """True iff every companion exists, has the right shape, and zero NaN rows.

    Returns (ok, reason). Used by --skip-existing to avoid re-truncating and
    rebuilding a dataset/split that already converted cleanly.
    """
    for comp, N_f in zip(comp_paths, n_rows):
        if not comp.exists():
            return False, f"missing {comp.name}"
        try:
            with h5py.File(comp, "r") as cf:
                if "teacher_logits" not in cf:
                    return False, f"{comp.name}: no teacher_logits dataset"
                dset = cf["teacher_logits"]
                if dset.shape != (N_f, num_classes):
                    return False, (
                        f"{comp.name}: shape {tuple(dset.shape)} != "
                        f"({N_f}, {num_classes})"
                    )
                for start in range(0, N_f, NAN_SCAN_CHUNK):
                    end = min(start + NAN_SCAN_CHUNK, N_f)
                    if np.isnan(dset[start:end]).any():
                        return False, f"{comp.name}: NaN rows present"
        except OSError as e:
            return False, f"{comp.name}: unreadable ({e})"
    return True, "all companions present, correct shape, zero NaN"


def build_one(npz_dir, tag, data_path, out_dir, dataset, split, skip_existing=False):
    print(f"\n=== {dataset}/{split} ===")
    srcs = source_files(data_path, dataset, split)
    if not srcs:
        raise FileNotFoundError(
            f"No source h5 found in {Path(data_path) / dataset / split}"
        )
    npzs = npz_files(npz_dir, tag, dataset, split)
    if not npzs:
        raise FileNotFoundError(
            f"No npz matching outputs_{tag}_{dataset}_{split}_*.npz in {npz_dir}"
        )
    print(f"  source files: {len(srcs)}   npz shards: {len(npzs)}")

    # num_classes from the first shard.
    with np.load(npzs[0]) as z:
        num_classes = int(z["logits"].shape[1])
    print(f"  num_classes (C): {num_classes}")

    # Row count per source file and companion paths (no writes yet).
    out_split_dir = Path(out_dir) / dataset / split
    out_split_dir.mkdir(parents=True, exist_ok=True)
    n_rows = []
    comp_paths = []
    for src in srcs:
        with h5py.File(src, "r") as f:
            N_f = int(len(f["data"]))
        n_rows.append(N_f)
        comp_paths.append(out_split_dir / (src.stem + ".h5"))
    print(f"  total rows: {sum(n_rows)}")

    # Skip a clean, already-converted dataset/split before truncating anything.
    if skip_existing:
        ok, reason = companions_complete(comp_paths, n_rows, num_classes)
        if ok:
            print(f"  SKIP {dataset}/{split}: {reason}")
            return len(srcs), sum(n_rows), num_classes
        print(f"  (re)building {dataset}/{split}: {reason}")

    # Pre-create companions initialized to NaN (truncates any existing file).
    coverage = []  # bool mask per file (which rows have been written)
    for comp, N_f in zip(comp_paths, n_rows):
        with h5py.File(comp, "w") as cf:
            dset = cf.create_dataset(
                "teacher_logits",
                shape=(N_f, num_classes),
                dtype=np.float16,
                chunks=(min(N_f, 4096), num_classes) if N_f > 0 else None,
            )
            # Fill with NaN sentinel in chunks (avoid a big in-RAM array).
            nan_row = np.full((1, num_classes), np.nan, dtype=np.float16)
            for start in range(0, N_f, NAN_SCAN_CHUNK):
                end = min(start + NAN_SCAN_CHUNK, N_f)
                dset[start:end] = np.broadcast_to(
                    nan_row, (end - start, num_classes)
                )
        coverage.append(np.zeros(N_f, dtype=bool))

    # Open companions for writing (one handle each).
    handles = [h5py.File(p, "a") for p in comp_paths]
    try:
        n_dup_diff = 0
        for npz_path in npzs:
            with np.load(npz_path) as z:
                logits = z["logits"]  # (n, C) fp16
                keys = z["sample_keys"]  # (n, 2) int64
                if logits.shape[1] != num_classes:
                    raise ValueError(
                        f"{npz_path}: C={logits.shape[1]} != {num_classes}"
                    )
                fids = keys[:, 0]
                sids = keys[:, 1]
                for fid in np.unique(fids):
                    fid = int(fid)
                    if fid >= len(handles) or fid < 0:
                        raise ValueError(
                            f"{npz_path}: file_idx {fid} out of range "
                            f"(have {len(handles)} source files for {dataset}/{split})"
                        )
                    rows = np.where(fids == fid)[0]
                    sidx = sids[rows]
                    if sidx.max() >= n_rows[fid]:
                        raise ValueError(
                            f"{npz_path}: sample_idx {int(sidx.max())} >= "
                            f"{n_rows[fid]} rows in {comp_paths[fid].name}"
                        )
                    order = np.argsort(sidx, kind="stable")
                    sidx_sorted = sidx[order]
                    vals = logits[rows][order]

                    # Detect double-writes with differing values (stray/dup npz).
                    already = coverage[fid][sidx_sorted]
                    if already.any():
                        dup_pos = sidx_sorted[already]
                        existing = handles[fid]["teacher_logits"][
                            np.sort(np.unique(dup_pos))
                        ]
                        # cheap heads-up; differing values flagged below in detail
                        n_dup = int(already.sum())
                        new_for_dup = vals[already]
                        # compare against just-read existing for the unique set
                        diff = ~np.allclose(
                            np.nan_to_num(existing[0]), np.nan_to_num(new_for_dup[0])
                        ) if n_dup else False
                        if diff:
                            n_dup_diff += n_dup
                            print(
                                f"  WARN: {n_dup} cells in {comp_paths[fid].name} "
                                f"rewritten with DIFFERING values (from {os.path.basename(npz_path)})"
                            )

                    handles[fid]["teacher_logits"][sidx_sorted] = vals
                    coverage[fid][sidx_sorted] = True
        if n_dup_diff:
            print(f"  WARN total differing double-written cells: {n_dup_diff}")
    finally:
        for h in handles:
            h.close()

    # Coverage check: re-scan each companion for any remaining NaN rows.
    missing_total = 0
    for src, comp, N_f in zip(srcs, comp_paths, n_rows):
        n_missing = 0
        first_missing = None
        with h5py.File(comp, "r") as cf:
            dset = cf["teacher_logits"]
            for start in range(0, N_f, NAN_SCAN_CHUNK):
                end = min(start + NAN_SCAN_CHUNK, N_f)
                block = dset[start:end]
                bad = np.isnan(block).any(axis=1)
                if bad.any():
                    n_missing += int(bad.sum())
                    if first_missing is None:
                        first_missing = start + int(np.where(bad)[0][0])
        if n_missing:
            missing_total += n_missing
            print(
                f"  MISSING: {comp.name}: {n_missing}/{N_f} rows still NaN "
                f"(first at sample_idx {first_missing})"
            )
    if missing_total:
        raise RuntimeError(
            f"{dataset}/{split}: {missing_total} rows have no teacher logits. "
            "Conversion incomplete -- do NOT use for distillation."
        )
    print(
        f"  OK {dataset}/{split}: {len(srcs)} companions, "
        f"{sum(n_rows)} rows, C={num_classes}, zero NaN."
    )
    return len(srcs), sum(n_rows), num_classes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--npz-dir", required=True, help="Directory with sharded teacher npz files"
    )
    ap.add_argument(
        "--tag", required=True, help="Teacher save tag, e.g. pretrain_l"
    )
    ap.add_argument(
        "--data-path",
        default="/global/cfs/cdirs/m4567/www/",
        help="Root of source h5 datasets (<data-path>/<dataset>/<split>)",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help="Companion root (default: <npz-dir>/companion)",
    )
    ap.add_argument(
        "--dataset",
        required=True,
        help="Dataset name, comma-list, or 'pretrain' to expand to the 7",
    )
    ap.add_argument(
        "--split",
        required=True,
        help="train / val / test (comma-list allowed)",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a dataset/split whose companions all exist with the right "
        "shape and zero NaN (safe to re-run after an interrupted job)",
    )
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.npz_dir, "companion")
    datasets = expand_datasets(args.dataset)
    splits = [s.strip() for s in args.split.split(",") if s.strip()]
    print(f"npz-dir : {args.npz_dir}")
    print(f"out-dir : {out_dir}")
    print(f"tag     : {args.tag}")
    print(f"datasets: {datasets}")
    print(f"splits  : {splits}")

    summary = []
    for ds in datasets:
        for sp in splits:
            nfiles, nrows, C = build_one(
                args.npz_dir, args.tag, args.data_path, out_dir, ds, sp,
                skip_existing=args.skip_existing,
            )
            summary.append((ds, sp, nfiles, nrows, C))

    print("\n=== SUMMARY ===")
    for ds, sp, nfiles, nrows, C in summary:
        print(f"  {ds}/{sp}: files={nfiles} rows={nrows} C={C}")


if __name__ == "__main__":
    main()
