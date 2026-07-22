"""Read-only validation of completed teacher-logit chunk files.

Checks, for chunks 0..MAXK of NUM_CHUNKS:
  - exactly EXPECTED_RANKS rank files present per chunk
  - consistent key set across every file
  - consistent trailing (per-event) shape per key
  - no NaN / Inf in float arrays
  - per-chunk and total event counts
Also estimates peak memory for the final concat_logits.py merge.
"""
import glob
import os
import re
import sys

import numpy as np

OUT = "/pscratch/sd/t/twamorka/omnilearned/teacher_logits/TEST"
TAG = "pretrain_l"
DATASET = "atlas"
SPLIT = "train"
NUM_CHUNKS = 24
EXPECTED_RANKS = 16

pat = re.compile(
    rf"outputs_{TAG}_{DATASET}_{SPLIT}_chunk(\d+)of{NUM_CHUNKS}_rank(\d+)\.npz$"
)

files = sorted(glob.glob(os.path.join(OUT, f"outputs_{TAG}_{DATASET}_{SPLIT}_chunk*of{NUM_CHUNKS}_rank*.npz")))
by_chunk = {}
for f in files:
    m = pat.search(os.path.basename(f))
    if not m:
        continue
    k, r = int(m.group(1)), int(m.group(2))
    by_chunk.setdefault(k, {})[r] = f

ref_keys = None
ref_shapes = None
total_events = 0
total_bytes = 0
problems = []

for k in sorted(by_chunk):
    ranks = by_chunk[k]
    if len(ranks) != EXPECTED_RANKS:
        problems.append(f"chunk {k}: {len(ranks)}/{EXPECTED_RANKS} rank files (missing {sorted(set(range(EXPECTED_RANKS))-set(ranks))})")
    chunk_events = 0
    for r in sorted(ranks):
        f = ranks[r]
        total_bytes += os.path.getsize(f)
        with np.load(f) as z:
            keys = sorted(z.keys())
            if ref_keys is None:
                ref_keys = keys
                ref_shapes = {key: z[key].shape[1:] for key in keys}
            elif keys != ref_keys:
                problems.append(f"{os.path.basename(f)}: keys {keys} != {ref_keys}")
            n_rows = None
            for key in keys:
                a = z[key]
                if a.size == 0:
                    continue
                if key in ref_shapes and a.shape[1:] != ref_shapes[key]:
                    problems.append(f"{os.path.basename(f)}: {key} per-event shape {a.shape[1:]} != {ref_shapes[key]}")
                if n_rows is None:
                    n_rows = a.shape[0]
                elif a.shape[0] != n_rows:
                    problems.append(f"{os.path.basename(f)}: {key} rows {a.shape[0]} != {n_rows}")
                if np.issubdtype(a.dtype, np.floating):
                    if not np.isfinite(a).all():
                        nnan = int(np.isnan(a).sum()); ninf = int(np.isinf(a).sum())
                        problems.append(f"{os.path.basename(f)}: {key} has {nnan} NaN, {ninf} Inf")
            chunk_events += (n_rows or 0)
    total_events += chunk_events
    print(f"chunk {k:2d}: {len(ranks):2d} ranks, {chunk_events:>10,} events")

print("\n=== summary ===")
print(f"chunks present : {sorted(by_chunk)}")
print(f"keys           : {ref_keys}")
print(f"per-event shape: {ref_shapes}")
print(f"total events (chunks present): {total_events:,}")
gb = total_bytes / 1e9
print(f"on-disk size (present chunks): {gb:.2f} GB")
# Final merge loads ALL chunks (0..23) for the split, holds every array in a
# list, then concatenates -> ~2x the full-dataset in-memory size at peak.
full_est = gb * (NUM_CHUNKS / max(len(by_chunk), 1))
print(f"est. full-split on disk (all {NUM_CHUNKS} chunks): {full_est:.2f} GB")
print(f"est. PEAK RAM for concat_logits.py merge (~2x): {2*full_est:.2f} GB")

if problems:
    print(f"\n!!! {len(problems)} PROBLEM(S):")
    for p in problems:
        print("  -", p)
    sys.exit(1)
else:
    print("\nOK: all present chunks are well-formed (keys/shapes consistent, no NaN/Inf).")
