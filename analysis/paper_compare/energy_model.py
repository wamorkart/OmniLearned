"""Comparison #2 -- apply Petitjean et al.'s energy model (their Table 5,
7nm process, pJ per add / mul) to our models' forward-pass op counts, and
place them on the axes of their Fig. 5 (energy per jet vs background
rejection).

Their method (paper Sec 3.3): "estimate the total energy consumption of a
forward pass by counting all operations in the network and weighting them
with the corresponding energy cost". A matmul-dominated forward pass of
M MACs costs M multiplies + M adds, so

    E_forward ~= MACs * (E_mul[dtype] + E_add[dtype]).

We report three points per model:
    float32     - everything in fp32
    bfloat16    - everything in bf16 (their AMP baseline)
    int8-linear - inner linear layers in int8, a residual fraction (attention
                  softmax, norms, pooling, embed/output) left at bf16. We use
                  a coarse INT8_COVERAGE fraction; the paper keeps input/output
                  layers and equivariance-critical ops higher precision and
                  finds ~2x per quant step, ~10x combined.

This is an estimate on top of an estimate (analytic MACs, generic pJ/op) --
good for a frontier plot, not a hardware number. Real LUT/DSP/latency needs
the hls4ml/QONNX export.

  python analysis/paper_compare/energy_model.py --outdir <dir>
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_data import (
    THEIR_ENERGY_PJ, THEIR_TABLE1, OUR_MODELS, OUR_FLOPS_AT_MEDIAN_N, MEDIAN_N,
)

# fraction of MACs assumed to move to int8 in the "int8-linear" point.
# DeepSets is almost entirely stacked linear layers; PET2 has attention +
# the pairwise interaction block that we keep at bf16.
INT8_COVERAGE = {"deepsets": 0.95, "pet2": 0.75}

# map an OUR_MODELS key to its FLOPS key
FLOPS_KEY = {
    "teacher_L": "teacher_L",
    "pet2_small_ce": "pet2_small", "pet2_small_kd_pureKD": "pet2_small",
    "pet2_small_kd_a05": "pet2_small", "pet2_small_pretrainKD_ft": "pet2_small",
    "pet2_micro_kd_a05": "pet2_micro", "pet2_micro_ce": "pet2_micro",
    "deepsets_small_kd_a05": "deepsets_small", "deepsets_small_ce": "deepsets_small",
    "deepsets_small_kd_teachS": "deepsets_small",
    "deepsets_distillnet_kd": "deepsets_distillnet",
    "deepsets_distillnet_qat8": "deepsets_distillnet",
    "deepsets_distillnet_ptq8": "deepsets_distillnet",
}


def load_flops():
    flops = dict(OUR_FLOPS_AT_MEDIAN_N)
    probe_path = os.path.join(os.path.dirname(__file__), "flops_probe.json")
    if os.path.exists(probe_path):
        probe = json.load(open(probe_path))
        for k, v in probe.items():
            flops[k] = v["fwd_flops_at_N47"]
    return flops


def energy_pj(macs, dtype):
    e_add, e_mul = THEIR_ENERGY_PJ[dtype]
    return macs * (e_add + e_mul)


def mixed_int8_pj(macs, coverage):
    e_add8, e_mul8 = THEIR_ENERGY_PJ["int8"]
    e_add16, e_mul16 = THEIR_ENERGY_PJ["bfloat16"]
    q = macs * coverage
    r = macs * (1.0 - coverage)
    return q * (e_add8 + e_mul8) + r * (e_add16 + e_mul16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/pscratch/sd/t/twamorka/omnilearned/results/paper_compare")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    flops = load_flops()

    rows = []
    for key, arch, params, acc, auc, r50, r30, r30e, kind, _src in OUR_MODELS:
        fk = FLOPS_KEY.get(key)
        f = flops.get(fk)
        if f is None or r30 is None:
            continue
        macs = f / 2.0
        e_f32 = energy_pj(macs, "float32")
        e_bf16 = energy_pj(macs, "bfloat16")
        e_i8 = mixed_int8_pj(macs, INT8_COVERAGE[arch])
        rows.append((key, arch, kind, macs, r30, e_f32, e_bf16, e_i8))

    # ---- table ----
    lines = []
    lines.append(f"{'model':<30} {'arch':<9} {'MACs@N'+str(MEDIAN_N):>14} "
                 f"{'E fp32 (nJ)':>12} {'E bf16 (nJ)':>12} {'E int8* (nJ)':>13} {'rej@0.3':>9}")
    lines.append("-" * 104)
    for key, arch, kind, macs, r30, ef, eb, ei in rows:
        lines.append(f"{key:<30} {arch:<9} {macs:>14,.0f} "
                     f"{ef/1e3:>12.2f} {eb/1e3:>12.2f} {ei/1e3:>13.2f} {r30:>9.0f}")
    lines.append("")
    lines.append("* int8-linear = inner linear layers int8, rest bf16 "
                 f"(coverage: deepsets {INT8_COVERAGE['deepsets']}, pet2 {INT8_COVERAGE['pet2']}). "
                 "pJ/op from Petitjean et al. Table 5 (7nm). 1 nJ = 1000 pJ.")
    lines.append("Energy per jet forward pass at the top-tagging median N=%d." % MEDIAN_N)
    lines.append("Methodology matches their Sec 3.3 (op-count x pJ/op); NOT directly on their")
    lines.append("Fig. 5 curve -- that is JetClass multi-class with ternary weights, this is")
    lines.append("top-tagging binary with int8. Analytic MACs + generic pJ/op = estimate only.")
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(args.outdir, "energy_model.txt"), "w") as fh:
        fh.write(txt + "\n")

    # ---- plot: their Fig. 5 axes (energy/jet vs rejection@0.3) ----
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    col = {"pet2": "#1f77b4", "deepsets": "#d62728"}
    for key, arch, kind, macs, r30, ef, eb, ei in rows:
        xs = [ef / 1e9, eb / 1e9, ei / 1e9]     # pJ -> mJ (matches their Fig. 5 axis)
        ys = [r30, r30, r30]
        ax.plot(xs, ys, "-", color=col[arch], alpha=0.35, lw=1)
        ax.scatter(xs, ys, color=col[arch],
                   marker={"pet2": "o", "deepsets": "v"}[arch], s=45, zorder=4)
        ax.annotate(key.replace("deepsets_", "ds_").replace("pet2_", ""),
                    (ei / 1e9, r30), fontsize=6.5, color=col[arch],
                    xytext=(4, 3), textcoords="offset points")

    ax.set_xscale("log")
    ax.set_xlabel(r"estimated energy per jet forward pass  [mJ]  (fp32 $\rightarrow$ bf16 $\rightarrow$ int8-linear)")
    ax.set_ylabel(r"background rejection  $1/\varepsilon_B$  @  $\varepsilon_S = 0.3$")
    ax.set_title("Energy frontier (our models, Petitjean et al. Table-5 pJ/op model)\n"
                 "cf. their Fig. 5")
    ax.grid(True, which="both", alpha=0.25)
    ax.scatter([], [], color="#1f77b4", marker="o", label="OmniLearned/PET2 student")
    ax.scatter([], [], color="#d62728", marker="v", label="DeepSets student")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = os.path.join(args.outdir, f"energy_model.{ext}")
        fig.savefig(p, dpi=150)
        print("wrote", p)


if __name__ == "__main__":
    main()
