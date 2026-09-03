"""One-off diagnostic: build a lhco_ad dataset with NO mixing at all --
100% signal (pid=1) vs 100% background (pid=0), for testing whether the
classifier can learn anything on the easiest possible version of this task.

Not part of the real pipeline -- writes to LHCO/pure_test/lhco_ad/, separate
from the real nsig_<N>/lhco_ad/ folders convert_lhco.py produces.

Run (from this directory, so the plain import below resolves):
    python convert_lhco_pure_test.py
"""

import numpy as np

from convert_lhco import (
    OUT_DIR,
    SIGNAL_FILES,
    BACKGROUND_EXTENDED_FILES,
    build_rows,
    write_pool,
)

rng = np.random.default_rng(0)

print(f"Loading signal (100%, pid=1): {SIGNAL_FILES}")
sig_rows = build_rows(SIGNAL_FILES, pid_label=1)  # nsig=None -> every available signal event

print(f"Loading background (100%, pid=0): {BACKGROUND_EXTENDED_FILES}")
bkg_rows = build_rows(BACKGROUND_EXTENDED_FILES, pid_label=0)

out_dir = OUT_DIR / "pure_test"
for filename, rows in (("data", sig_rows), ("bkg", bkg_rows)):
    counts = write_pool("lhco_ad", filename, out_dir, rows, val_frac=0.1, test_frac=0.1, rng=rng)
    print(f"pure_test/lhco_ad/{filename}: {counts}, total={sum(counts.values())}")
