"""Characterize the atlas_flav (ATLAS flavour tagging) dataset before committing
GPU time to an ftag fine-tune.

Resolves, empirically rather than by assumption:
  1. which jet-label index is b / c / light / tau, by correlating the jet label
     with the 8-class per-track ORIGIN labels (data_pid). ATLAS FTAG track
     origin classes distinguish tracks fromB / fromC / fromTau, so a jet class
     whose tracks are dominated by a given origin identifies that flavour.
  2. class balance, track multiplicity, padding convention, feature layout.

Deliberately light on the shared login node: reads bounded slices only.
"""
import h5py, numpy as np

P = "/global/cfs/cdirs/m4567/www/atlas_flav/val/val_flav_0.h5"
N_JET = 300_000      # jet-level stats
N_TRK = 60_000       # track-level stats (heavier: 40x21 per jet)

with h5py.File(P, "r") as f:
    y   = f["pid"][:N_JET].astype(int)
    dpi = f["data_pid"][:N_TRK].astype(int).squeeze(-1)   # (N,40)
    X   = f["data"][:N_TRK]                                # (N,40,21)
    g   = f["global"][:N_JET]

print("="*78)
print("1. JET CLASS BALANCE  (pid, N=%d)" % N_JET)
u, c = np.unique(y, return_counts=True)
for ui, ci in zip(u, c):
    print(f"   class {ui}: {ci:>8,}  ({100*ci/len(y):5.2f}%)")
print(f"   imbalance ratio max/min = {c.max()/c.min():.1f}x")

print("\n" + "="*78)
print("2. PADDING / TRACK MULTIPLICITY")
# a padded track should be all-zero in features; data_pid uses -1
pad_by_pid = (dpi == -1)
allzero = (X == 0).all(axis=2)
print(f"   data_pid == -1 fraction : {pad_by_pid.mean():.4f}")
print(f"   all-zero feature rows   : {allzero.mean():.4f}")
print(f"   agreement (-1 <-> allzero): {(pad_by_pid == allzero).mean():.4f}")
ntrk = (~pad_by_pid).sum(axis=1)
print(f"   real tracks/jet: min={ntrk.min()} med={int(np.median(ntrk))} "
      f"mean={ntrk.mean():.1f} max={ntrk.max()}  (slot cap = {dpi.shape[1]})")
print(f"   jets at the 40-track cap: {100*(ntrk==dpi.shape[1]).mean():.2f}%")

print("\n" + "="*78)
print("3. TRACK-ORIGIN (data_pid) GLOBAL DISTRIBUTION, padding excluded")
uo, co = np.unique(dpi[~pad_by_pid], return_counts=True)
for ui, ci in zip(uo, co):
    print(f"   origin {ui}: {ci:>9,}  ({100*ci/co.sum():5.2f}%)")

print("\n" + "="*78)
print("4. JET LABEL  x  TRACK ORIGIN  -> identifies the flavour of each label")
yt = y[:N_TRK]
n_orig = int(dpi.max()) + 1
print("   rows = jet label, cols = track-origin share within that jet class (%)")
hdr = "        " + "".join(f"  org{o:>2}" for o in range(n_orig))
print(hdr)
prof = np.zeros((len(u), n_orig))
for i, ui in enumerate(u):
    m = (yt[:, None] == ui) & (~pad_by_pid)
    tot = m.sum()
    if tot == 0:
        continue
    row = np.array([((dpi == o) & m).sum() for o in range(n_orig)], float) / tot
    prof[i] = row
    print(f"   lab {ui} " + "".join(f" {100*v:5.1f}" for v in row))
    print(f"          mean real tracks/jet = {ntrk[yt==ui].mean():.1f}")

print("\n   --> Interpretation: under the ATLAS FTAG convention the 8 origin")
print("       classes are {pileup, fake, primary, fromB, fromBC, fromC,")
print("       fromTau, otherSecondary}. The jet class most enriched in the")
print("       fromB/fromBC origins is b; the one enriched in fromC is c;")
print("       the one enriched in fromTau is tau; the flattest/primary-")
print("       dominated one with the fewest tracks is light.")
for i, ui in enumerate(u):
    j = int(np.argmax(prof[i][2:])) + 2   # skip pileup/fake
    print(f"       label {ui}: most-enriched non-pileup origin = {j}"
          f"  ({100*prof[i][j]:.1f}%)")

print("\n" + "="*78)
print("5. CONDITIONAL FEATURES (global, --num-cond 4)")
for i in range(g.shape[1]):
    col = g[:, i]
    print(f"   cond[{i}]: min={col.min():9.3f} max={col.max():9.3f} "
          f"mean={col.mean():8.3f} std={col.std():7.3f} "
          f"nuniq={len(np.unique(col[:20000]))}")

print("\n" + "="*78)
print("6. FEATURE LAYOUT (21 cols; --num-add 17 => last 17 are 'add_info')")
real = ~pad_by_pid
for i in range(X.shape[2]):
    col = X[:, :, i][real]
    tail = "  <- add_info" if i >= X.shape[2] - 17 else "  <- kinematic"
    print(f"   feat[{i:>2}]: min={col.min():9.3f} max={col.max():9.3f} "
          f"mean={col.mean():8.3f} std={col.std():7.3f} "
          f"nuniq={len(np.unique(col[:5000])):>6}{tail}")
