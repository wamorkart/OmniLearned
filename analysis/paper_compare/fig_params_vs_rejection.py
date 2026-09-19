"""Comparison #1 -- put our top-tagging models onto the axes of Petitjean et
al. Fig. 3: background rejection 1/eps_B at eps_S=0.3 versus parameter count
(log x). Their Table 1 points + Fig. 3 ultra-mini band form the reference;
our PET2 and DeepSets students are overlaid.

  python analysis/paper_compare/fig_params_vs_rejection.py --outdir <dir>
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paper_data import (
    THEIR_TABLE1, THEIR_ULTRAMINI_FIG3, THEIR_ULTRAMINI_NOTE,
    OUR_MODELS, OUR_FLOPS_AT_MEDIAN_N,
)

try:
    import json
    _fp = os.path.join(os.path.dirname(__file__), "flops_probe.json")
    PROBE = json.load(open(_fp)) if os.path.exists(_fp) else {}
except Exception:
    PROBE = {}

PARAM_FILL = {  # keys in OUR_MODELS whose params are None until flops_probe runs
    "pet2_micro_kd_a05": "pet2_micro",
    "pet2_micro_ce": "pet2_micro",
}


def our_params(key, listed):
    if listed is not None:
        return listed
    probe_key = PARAM_FILL.get(key)
    if probe_key and probe_key in PROBE:
        return PROBE[probe_key]["params"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/pscratch/sd/t/twamorka/omnilearned/results/paper_compare")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 6.0))

    # ---- their Table 1 ----
    for equiv, pre, marker, label in [
        (False, False, "o", "theirs: plain, from scratch"),
        (True,  False, "s", "theirs: Lorentz-equiv, from scratch"),
        (False, True,  "^", "theirs: plain, pretrained"),
        (True,  True,  "D", "theirs: Lorentz-equiv, pretrained"),
    ]:
        pts = [(p * 1e6, r30) for (_n, p, _a, _u, _r50, r30, _nt, e, q) in THEIR_TABLE1
               if e == equiv and q == pre and r30 is not None]
        if pts:
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, marker=marker, s=55, facecolors="none",
                       edgecolors="0.45", linewidths=1.3, label=label, zorder=3)

    # label the two anchors most relevant to us
    for name, p, *_rest in THEIR_TABLE1:
        if name in ("L-GATr-slim", "L-GATr-slim-f.t.", "OmniLearned-L", "OmniLearned-M"):
            r30 = _rest[3]
            ax.annotate(name, (p * 1e6, r30), fontsize=7, color="0.4",
                        xytext=(4, 4), textcoords="offset points")

    # ---- their Fig. 3 ultra-mini band (approx) ----
    for arch, col in [("LLoCa-Transformer", "0.6"), ("L-GATr-slim", "0.72")]:
        xs = sorted(THEIR_ULTRAMINI_FIG3)
        ys = [THEIR_ULTRAMINI_FIG3[x][arch] for x in xs]
        ax.plot(xs, ys, "--", color=col, lw=1.2, zorder=2,
                label=f"theirs Fig.3 (approx): {arch}")

    # ---- ours ----
    style = {
        "pet2":     dict(color="#1f77b4", marker="o"),
        "deepsets": dict(color="#d62728", marker="v"),
    }
    for key, arch, params, acc, auc, r50, r30, r30e, kind, _src in OUR_MODELS:
        if key == "teacher_L":
            continue  # == OmniLearned-L in their table, drawn there
        p = our_params(key, params)
        if p is None or r30 is None:
            continue
        st = style[arch]
        ax.errorbar(p, r30, yerr=(r30e or 0), fmt=st["marker"], color=st["color"],
                    ms=9, capsize=3, zorder=5,
                    mec="white", mew=0.8)
        ax.annotate(kind.split(",")[0], (p, r30), fontsize=7, color=st["color"],
                    xytext=(6, -3), textcoords="offset points")

    # legend proxies for ours
    ax.scatter([], [], color="#1f77b4", marker="o", s=70, label="ours: OmniLearned/PET2 student")
    ax.scatter([], [], color="#d62728", marker="v", s=70, label="ours: DeepSets student")

    ax.set_xscale("log")
    ax.set_xlabel("number of parameters")
    ax.set_ylabel(r"background rejection  $1/\varepsilon_B$  @  $\varepsilon_S = 0.3$")
    ax.set_title("Top tagging: parameter economy\n"
                 "ours vs Petitjean et al. (arXiv:2512.17011), Table 1 + Fig. 3")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=7.5, loc="lower right", ncol=1)
    ax.text(0.01, 0.01, THEIR_ULTRAMINI_NOTE, transform=ax.transAxes, fontsize=6,
            color="0.5", va="bottom")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(args.outdir, f"params_vs_rejection.{ext}")
        fig.savefig(path, dpi=150)
        print("wrote", path)


if __name__ == "__main__":
    main()
