"""Semantic sanity check on teacher logits (fast: samples a few rank files).

Validates per-event properties that tell you the logits are *real model output*,
not corrupted/truncated/collapsed:

  (2) prediction & event_prediction are valid softmaxes (rows sum ~1, in [0,1])
  (3) softmax(logits) reproduces the saved prediction  <-- strongest check
  (4) argmax(prediction) is spread across classes (teacher hasn't collapsed)

Run before trusting the pipeline enough to launch the other datasets.
"""
import glob
import os
import sys

import numpy as np

OUT = "/pscratch/sd/t/twamorka/omnilearned/teacher_logits/TEST"
PAT = "outputs_pretrain_l_atlas_train_chunk*of24_rank*.npz"
N_SAMPLE_FILES = 6          # spread across chunks
N_JET = 200                 # width of `prediction`
N_EVT = 10                  # width of `event_prediction`


def softmax(x):
    x = x.astype(np.float32)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


files = sorted(glob.glob(os.path.join(OUT, PAT)))
if not files:
    print("No files found:", PAT); sys.exit(1)
# even sample across the sorted list
idx = np.linspace(0, len(files) - 1, min(N_SAMPLE_FILES, len(files))).astype(int)
sample = [files[i] for i in idx]
print(f"{len(files)} files present; sampling {len(sample)}:")
for f in sample:
    print("   ", os.path.basename(f))
print()

ok = True
argmax_counts = np.zeros(N_JET, dtype=np.int64)
total = 0
for f in sample:
    d = np.load(f)
    pred = d["prediction"].astype(np.float32)
    evt = d["event_prediction"].astype(np.float32)
    logits = d["logits"].astype(np.float32)
    n = pred.shape[0]; total += n

    # (2) valid softmaxes
    for name, a in (("prediction", pred), ("event_prediction", evt)):
        s = a.sum(axis=1)
        if not (np.all(a >= -1e-4) and np.all(a <= 1 + 1e-4)):
            print(f"  [FAIL] {name}: values outside [0,1] ({a.min():.3g}..{a.max():.3g})"); ok = False
        if not np.allclose(s, 1.0, atol=1e-2):
            print(f"  [FAIL] {name}: rows not normalized (sum range {s.min():.4f}..{s.max():.4f})"); ok = False

    # (3) softmax(logits) reproduces saved predictions
    # try layout [jet(200) | event(10)]
    sm_jet = softmax(logits[:, :N_JET])
    sm_evt = softmax(logits[:, N_JET:N_JET + N_EVT])
    d_jet = np.abs(sm_jet - pred).max()
    d_evt = np.abs(sm_evt - evt).max()
    flag_jet = "OK" if d_jet < 2e-2 else "MISMATCH"
    flag_evt = "OK" if d_evt < 2e-2 else "MISMATCH"
    if d_jet >= 2e-2 or d_evt >= 2e-2:
        ok = False
    print(f"  {os.path.basename(f)}: n={n:,}  "
          f"softmax(logits[:200])->pred maxdiff={d_jet:.3g} [{flag_jet}]  "
          f"softmax(logits[200:210])->evt maxdiff={d_evt:.3g} [{flag_evt}]")

    # (4) collapse check on jet head
    am = pred.argmax(axis=1)
    np.add.at(argmax_counts, am, 1)

print()
nz = (argmax_counts > 0).sum()
top = argmax_counts.max() / max(total, 1)
print(f"(4) collapse check: {nz}/{N_JET} jet classes ever predicted; "
      f"most-frequent class = {100*top:.1f}% of events")
if nz < 2 or top > 0.99:
    print("  [WARN] predictions look collapsed onto one class"); ok = False

print()
print("RESULT:", "PASS — logits are valid model output" if ok else "FAIL — see flags above")
sys.exit(0 if ok else 1)
