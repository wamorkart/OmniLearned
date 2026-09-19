"""Compare fp32 vs bf16 teacher logits produced by the A/B harness.

Both runs process the identical event slice in the identical order
(same chunk, same rank, shuffle=False), so rows line up 1:1.
"""
import glob
import os
import sys

import numpy as np

FP32_DIR = sys.argv[1] if len(sys.argv) > 1 else "checks/ab_fp32"
BF16_DIR = sys.argv[2] if len(sys.argv) > 2 else "checks/ab_bf16"


def load_one(d):
    fs = sorted(glob.glob(os.path.join(d, "outputs_*_rank*.npz")))
    if not fs:
        sys.exit(f"No npz in {d}")
    return np.load(fs[0]), os.path.basename(fs[0])


a, na = load_one(FP32_DIR)
b, nb = load_one(BF16_DIR)
print(f"fp32: {na}\nbf16: {nb}\n")

# sanity: same events, same order
ka, kb = a["sample_keys"], b["sample_keys"]
if ka.shape != kb.shape or not np.array_equal(ka, kb):
    print("[WARN] sample_keys differ between runs — rows may not align!")
else:
    print(f"sample_keys identical ({ka.shape[0]:,} events, aligned)\n")

la = a["logits"].astype(np.float32)
lb = b["logits"].astype(np.float32)
abs_d = np.abs(la - lb)
print("=== raw logits (210-dim) ===")
print(f"  max |Δ|  = {abs_d.max():.4f}")
print(f"  mean |Δ| = {abs_d.mean():.5f}")
print(f"  logit scale (fp32 std) = {la.std():.3f}")

# what actually matters for distillation: the softmax targets + the argmax label
for name, sl in (("jet (200)", slice(0, 200)), ("event (10)", slice(200, 210))):
    pa = a["prediction"] if sl.start == 0 else a["event_prediction"]
    pb = b["prediction"] if sl.start == 0 else b["event_prediction"]
    pa = pa.astype(np.float32); pb = pb.astype(np.float32)
    arga, argb = pa.argmax(1), pb.argmax(1)
    agree = (arga == argb).mean()
    # KL(fp32 || bf16) per event, averaged
    eps = 1e-9
    kl = (pa * (np.log(pa + eps) - np.log(pb + eps))).sum(1).mean()
    print(f"\n=== {name} head ===")
    print(f"  prob max|Δ|   = {np.abs(pa - pb).max():.5f}")
    print(f"  argmax agree  = {100*agree:.3f}%")
    print(f"  mean KL(fp32||bf16) = {kl:.2e} nats")

print("\nInterpretation: argmax agreement ~100% and tiny KL => bf16 logits are")
print("equivalent distillation targets. Compare the printed forward times in the")
print("two run logs for the speedup.")
