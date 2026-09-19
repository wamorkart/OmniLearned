"""
One-off check: with the current PET_body attention wiring (nn.MultiheadAttention
fed a non-None additive attn_mask on every call, built at network.py:581-596),
does PyTorch's SDPA dispatcher ever actually pick the flash-attention backend,
with vs without --interaction (use_int)?

Two probes:
  1. Profiler trace of a real forward pass -> which aten::_scaled_dot_product_*
     kernel actually fires.
  2. Force each SDPA backend one at a time via sdpa_kernel(...) and see which
     ones accept vs reject the mask PET_body builds (RuntimeError = rejected).

Run on a GPU node:
  module load conda && conda activate /global/homes/t/twamorka/omnilearned-clean/env
  module load pytorch
  python flash_attention_check.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import torch
import torch.nn as nn

print(f"torch version: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("Need a GPU node for this check (flash/efficient SDPA backends are CUDA-only).")

device = "cuda"

from omnilearned.network import PET_body

B, N = 16, 64
input_dim = 4
num_coord = 2

torch.manual_seed(0)


def make_batch(B, N, input_dim, pad_frac=0.3):
    x = torch.randn(B, N, input_dim, device=device)
    # x[:, :, 2] is pT-like; PET_body masks on x[:, :, 2:3] != 0
    x[:, :, 2] = torch.rand(B, N, device=device) + 0.1
    n_pad = int(N * pad_frac)
    if n_pad > 0:
        x[:, -n_pad:, :] = 0.0
    return x


def build_body(use_int):
    return PET_body(
        input_dim=input_dim,
        base_dim=128,
        num_transformers=8,
        num_transf_local=2,
        num_heads=8,
        mlp_ratio=2,
        num_tokens=4,
        K=10,
        use_int=use_int,
        local_int=False,
        int_type="lhc",
        num_coord=num_coord,
    ).to(device).eval()


def profile_forward(use_int, x):
    model = build_body(use_int)
    with torch.no_grad():
        # warmup
        for _ in range(3):
            model(x)
        torch.cuda.synchronize()
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        ) as prof:
            model(x)
            torch.cuda.synchronize()

    def device_time_total(r):
        return getattr(r, "cuda_time_total", None) or getattr(r, "device_time_total", 0)

    rows = prof.key_averages()
    hits = [r for r in rows if "scaled_dot_product" in r.key or "attention" in r.key.lower()]
    print(f"\n--- use_int={use_int}: attention-related ops in trace ---")
    for r in sorted(hits, key=lambda r: -device_time_total(r)):
        print(f"  {r.key:55s} count={r.count:4d}  device_time_total={device_time_total(r)/1e3:.3f}ms")
    if not hits:
        print("  (no scaled_dot_product / attention ops found in trace)")


def force_backend_probe(use_int, x):
    print(f"\n--- use_int={use_int}: forcing each SDPA backend ---")
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:
        print("  torch.nn.attention.sdpa_kernel not available in this torch version, skipping")
        return

    backends = {
        "FLASH_ATTENTION": SDPBackend.FLASH_ATTENTION,
        "EFFICIENT_ATTENTION": SDPBackend.EFFICIENT_ATTENTION,
        "MATH": SDPBackend.MATH,
    }
    for name, backend in backends.items():
        model = build_body(use_int)
        try:
            with torch.no_grad(), sdpa_kernel(backend):
                model(x)
                torch.cuda.synchronize()
            print(f"  {name:20s} -> OK (accepted the mask PET_body builds)")
        except RuntimeError as e:
            msg = str(e).splitlines()[0]
            print(f"  {name:20s} -> REJECTED: {msg}")


if __name__ == "__main__":
    x = make_batch(B, N, input_dim)

    for use_int in [False, True]:
        profile_forward(use_int, x)

    for use_int in [False, True]:
        force_backend_probe(use_int, x)
