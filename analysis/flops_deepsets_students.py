"""FLOPs of the three Deep Sets students vs. jet multiplicity N.

Source of the student FLOP numbers in docs/paper_draft/ml4ps2026_deepsets_fpga.tex
(Table 2 and the Cost accounting paragraph). Same convention as
analysis/benchmark_inference.py, which produced the teacher numbers:
torch.utils.flop_counter.FlopCounterMode, batch 1, all N slots valid, no
padding. Table 2 quotes N=47 (median test-jet multiplicity); N=8/114 are the
observed min/max; N=150 is the padded file width.

CPU only, seconds to run:
    module load conda && conda activate ~/omnilearned-clean/env
    python analysis/flops_deepsets_students.py
"""
import torch
torch.set_num_threads(2)
from torch.utils.flop_counter import FlopCounterMode
from omnilearned.network import DeepSets
from omnilearned.utils import get_deepsets_parameters


def build(size, gnn=False):
    p = get_deepsets_parameters(size)
    kw = dict(num_interaction_layers=1, interaction_k=64) if gnn else {}
    m = DeepSets(input_dim=4, num_classes=2, mode="classifier", **p, **kw).eval()
    return m, sum(t.numel() for t in m.parameters())


def flops(m, n):
    x = torch.randn(1, n, 4)
    x[..., 2] = x[..., 2].abs() + 0.1  # log pT != 0 -> every slot counts as a real particle
    y = torch.zeros(1, dtype=torch.long)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        m(x, y)
    return fc.get_total_flops()


if __name__ == "__main__":
    for name, size, gnn in [("small", "small", False),
                            ("distillnet", "distillnet", False),
                            ("distillnet+GNN", "distillnet", True)]:
        m, npar = build(size, gnn)
        row = "  ".join(f"N={n}: {flops(m, n)/1e6:7.2f}M" for n in (8, 47, 64, 114, 150))
        print(f"{name:15s} params={npar:7d}  {row}")
