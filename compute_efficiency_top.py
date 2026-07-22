"""
Training-time compute/memory comparison for the top-tagging KD models.

Benchmarks the exact architectures used in the KD experiments (see
distill_train_top_micro.sh / distill_train_top_rep.sh / train.sh's
fine_tune_top_l line): PET2 with input_dim=4, num_coord=2, K=10,
interaction=True, num_classes=2, at each model's real training batch size
and N=150 particles (matches the padded shape of top/train/*.h5).

Reports, per model:
  - parameter count and fp32 weight memory
  - Lion optimizer state memory (1 momentum buffer per param, same as weights)
  - forward-only and forward+backward FLOPs (torch.utils.flop_counter;
    counts mm/addmm/bmm-family ops only -- the KNN search in
    LocalEmbeddingBlock is not matmul-based and is undercounted, but it is
    a small fraction of total cost next to attention/MLP layers)
  - peak CUDA memory allocated over a few full train steps
    (zero_grad -> forward -> backward -> optimizer.step)

Run inside a GPU allocation with the omnilearned-clean conda env active:
  module load conda && conda activate /global/homes/t/twamorka/omnilearned-clean/env
  module load pytorch
  python compute_efficiency_top.py
"""

import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode
from pytorch_optimizer import Lion

from omnilearned.network import PET2
from omnilearned.utils import get_model_parameters

N_PART = 150  # matches top/train/*.h5 padded particle-multiplicity axis
NUM_FEAT = 4
NUM_CLASSES = 2

# (size, interaction, local_interaction, batch) -- matches the actual scripts:
#   micro/small students: distill_train_top_micro.sh / distill_train_top_rep.sh
#   large teacher: train.sh's `fine_tune_top_l` line
CONFIGS = [
    ("micro", True, True, 128),
    ("small", True, True, 128),
    ("large", True, False, 8),
]


def make_batch(batch, device):
    x = torch.randn(batch, N_PART, NUM_FEAT, device=device)
    x[:, :, 2] = x[:, :, 2].abs() + 0.1  # nonzero -> "valid particle" per the mask convention
    y = torch.randint(0, NUM_CLASSES, (batch,), device=device)
    return x, y


def build_model(size, interaction, local_interaction, device):
    params = get_model_parameters(size)
    model = PET2(
        input_dim=NUM_FEAT,
        use_int=interaction,
        local_int=local_interaction,
        num_classes=NUM_CLASSES,
        mode="classifier",
        num_coord=2,
        K=10,
        **params,
    ).to(device)
    return model


def count_flops(model, x, y, criterion):
    model.zero_grad(set_to_none=True)
    with FlopCounterMode(model, display=False) as fwd_counter:
        out = model(x, y)
    fwd_flops = fwd_counter.get_total_flops()

    model.zero_grad(set_to_none=True)
    with FlopCounterMode(model, display=False) as fb_counter:
        out = model(x, y)
        loss = criterion(out["y_pred"], y)
        loss.backward()
    fwd_bwd_flops = fb_counter.get_total_flops()

    model.zero_grad(set_to_none=True)
    return fwd_flops, fwd_bwd_flops


def peak_train_step_memory(model, make_batch_fn, batch, device, steps=3):
    optimizer = Lion(model.parameters(), betas=(0.95, 0.98))
    criterion = nn.CrossEntropyLoss()

    # one warmup step so Lion's momentum buffers are allocated before we
    # start measuring peak memory (otherwise step 1 looks artificially high
    # and later steps artificially low, rather than a stable steady state)
    x, y = make_batch_fn(batch, device)
    optimizer.zero_grad(set_to_none=True)
    out = model(x, y)
    criterion(out["y_pred"], y).backward()
    optimizer.step()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    for _ in range(steps):
        x, y = make_batch_fn(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(x, y)
        loss = criterion(out["y_pred"], y)
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated(device)


def human_bytes(n):
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    free_bytes, total_bytes = torch.cuda.mem_get_info(device) if device.type == "cuda" else (0, 0)
    if device.type == "cuda":
        print(
            f"GPU free/total: {human_bytes(free_bytes)}/{human_bytes(total_bytes)} "
            f"(shared node -- another process may already hold memory)\n"
        )

    rows = []
    for size, interaction, local_interaction, requested_batch in CONFIGS:
        torch.cuda.empty_cache()
        model = build_model(size, interaction, local_interaction, device)
        n_params = sum(p.numel() for p in model.parameters())
        weight_bytes = n_params * 4
        optim_state_bytes = n_params * 4  # Lion: 1 fp32 momentum buffer/param

        # Actual training batch size may not fit alongside whatever else is
        # resident on a shared GPU; fall back to the largest power-of-two
        # <= requested_batch that fits, and report what was actually used.
        batch = requested_batch
        while True:
            try:
                x, y = make_batch(batch, device)
                fwd_flops, fwd_bwd_flops = count_flops(model, x, y, criterion)
                peak_mem = peak_train_step_memory(model, make_batch, batch, device)
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if batch <= 1:
                    raise
                batch //= 2
                print(f"  [{size}] OOM at batch={batch*2}, retrying at batch={batch}")

        rows.append(
            dict(
                size=size,
                requested_batch=requested_batch,
                batch=batch,
                params=n_params,
                weight_bytes=weight_bytes,
                optim_state_bytes=optim_state_bytes,
                fwd_gflops_per_batch=fwd_flops / 1e9,
                fwd_gflops_per_sample=fwd_flops / batch / 1e9,
                train_gflops_per_sample=fwd_bwd_flops / batch / 1e9,
                peak_mem_bytes=peak_mem,
            )
        )

        del model, x, y
        torch.cuda.empty_cache()

    hdr = (
        f"{'size':6} {'batch':>6} {'params':>10} {'weights':>9} {'optim':>9} "
        f"{'fwd GFLOP/samp':>15} {'train GFLOP/samp':>17} {'peak mem @ batch':>22}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        batch_note = str(r["batch"])
        if r["batch"] != r["requested_batch"]:
            batch_note += f" (req {r['requested_batch']})"
        print(
            f"{r['size']:6} {batch_note:>6} {r['params']:>10,} "
            f"{human_bytes(r['weight_bytes']):>9} {human_bytes(r['optim_state_bytes']):>9} "
            f"{r['fwd_gflops_per_sample']:>15.3f} {r['train_gflops_per_sample']:>17.3f} "
            f"{human_bytes(r['peak_mem_bytes']):>22}"
        )


if __name__ == "__main__":
    main()
