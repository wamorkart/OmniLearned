"""Comparisons #3 and #4 -- assemble the text tables once the GPU jobs in
run_paper_compare.sh have produced their logs under <outdir>.

#3  Quantization impact in the layout of Petitjean et al. Table 6
    (1/eps_B @ eps_S=0.3):  f32 | i8 inputs-only | i8 + weights.
    Ours: parsed from ptq_deepsets.py logs (--weights-float and default) plus
    the 8-bit QAT numbers from paper_data.

#4  Constituent-count robustness (their Fig. 6 right): full-N vs N=30,
    parsed from <outdir>/nconst_results.tsv written by eval_nconst_top.py.

  python analysis/paper_compare/make_tables.py --outdir <dir>
"""

import argparse
import glob
import os
import re

from paper_data import THEIR_TABLE6, OUR_MODELS

PTQ_LINE = re.compile(
    r"(?P<bits>\d+)-bit PTQ\s+acc=(?P<acc>[\d.]+)\s+auc=(?P<auc>[\d.]+)\s+"
    r"1/FPR@50%=(?P<r50>[\d.]+)\s+1/FPR@30%=(?P<r30>[\d.]+)"
)
FLOAT_LINE = re.compile(
    r"float32\s+acc=(?P<acc>[\d.]+)\s+auc=(?P<auc>[\d.]+)\s+"
    r"1/FPR@50%=(?P<r50>[\d.]+)\s+1/FPR@30%=(?P<r30>[\d.]+)"
)


def parse_ptq_log(path):
    """Return {'float32': r30, '8bit': r30, ...} for one ptq log."""
    if not path or not os.path.exists(path):
        return {}
    txt = open(path).read()
    out = {}
    m = FLOAT_LINE.search(txt)
    if m:
        out["float32"] = float(m.group("r30"))
    for m in PTQ_LINE.finditer(txt):
        out[f"{m.group('bits')}bit"] = float(m.group("r30"))
    return out


def our_qat_r30(tag_key):
    for k, arch, params, acc, auc, r50, r30, r30e, kind, src in OUR_MODELS:
        if k == tag_key:
            return r30
    return None


def table3(outdir):
    L = []
    L.append("=" * 78)
    L.append("#3  Quantization impact -- layout of Petitjean et al. (2512.17011) Table 6")
    L.append("     background rejection  1/eps_B @ eps_S = 0.3")
    L.append("=" * 78)
    L.append("")
    L.append("THEIRS (their Table 6; i8 = int8 inputs, then + ternary weights via STE / PARQ):")
    L.append(f"  {'arch':<20} {'f32':>12} {'i8':>12} {'i8+STE':>12} {'i8+PARQ':>12}")
    for arch, (f32, i8, ste, parq) in THEIR_TABLE6.items():
        L.append(f"  {arch:<20} {f32[0]:>7} ±{f32[1]:<3} {i8[0]:>7} ±{i8[1]:<3} "
                 f"{ste[0]:>7} ±{ste[1]:<3} {parq[0]:>7} ±{parq[1]:<3}")
    L.append("")
    L.append("OURS (DeepSets students; i8_in = int8 inputs only, i8_w = int8 inputs+weights):")
    L.append(f"  {'model':<24} {'f32':>10} {'i8_in (PTQ)':>13} {'i8_w (PTQ)':>12} {'i8_w (QAT)':>12}")
    rows = [
        ("deepsets_small_kd_a05",   "small",      "deepsets_small_kd_a05",   None),
        ("deepsets_distillnet_kd",  "distillnet", "deepsets_distillnet_kd",  "deepsets_distillnet_qat8"),
    ]
    for key, size, base_key, qat_key in rows:
        io = parse_ptq_log(os.path.join(outdir, f"ptq_inputs_only_{size}.log"))
        bw = parse_ptq_log(os.path.join(outdir, f"ptq_both_{size}.log"))
        f32 = io.get("float32") or bw.get("float32")
        r_io = io.get("8bit")
        r_bw = bw.get("8bit")
        r_qat = our_qat_r30(qat_key) if qat_key else None
        def fmt(v):
            return f"{v:>10.0f}" if isinstance(v, (int, float)) else f"{'--':>10}"
        L.append(f"  {key:<24} {fmt(f32)} {fmt(r_io):>13} {fmt(r_bw):>12} {fmt(r_qat):>12}")
    L.append("")
    L.append("note: our i8_w(PTQ) uses percentile-calibrated static PTQ; our i8_w(QAT)")
    L.append("is Brevitas int8 QAT with a KD fine-tune -- closest analog to their i8+PARQ")
    L.append("(weights adapted to the grid) rather than i8+STE (post-hoc).")
    return "\n".join(L)


def table4(outdir):
    path = os.path.join(outdir, "nconst_results.tsv")
    L = []
    L.append("=" * 78)
    L.append("#4  Constituent-count robustness -- cf. Petitjean et al. Fig. 6 (right, N<=30)")
    L.append("=" * 78)
    if not os.path.exists(path):
        L.append(f"(no results yet: {path} not found)")
        return "\n".join(L)
    rows = [l.rstrip("\n").split("\t") for l in open(path)][1:]
    by_tag = {}
    for tag, arch, size, params, nconst, acc, auc, r50, r30 in rows:
        by_tag.setdefault(tag, {})[int(nconst)] = (float(acc), float(auc), float(r50), float(r30))
    L.append(f"  {'tag':<40} {'N':>6} {'acc':>8} {'auc':>8} {'rej@50':>9} {'rej@30':>9}")
    for tag, d in by_tag.items():
        full = d.get(0)
        for nconst in sorted(d):
            acc, auc, r50, r30 = d[nconst]
            tag_lbl = tag if nconst == sorted(d)[0] else ""
            nlbl = "full" if nconst == 0 else str(nconst)
            drop = ""
            if full and nconst != 0:
                drop = f"  (rej@30 {100*(r30/full[3]-1):+.0f}%)"
            L.append(f"  {tag_lbl:<40} {nlbl:>6} {acc:>8.4f} {auc:>8.4f} {r50:>9.1f} {r30:>9.1f}{drop}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/pscratch/sd/t/twamorka/omnilearned/results/paper_compare")
    args = ap.parse_args()
    t3 = table3(args.outdir)
    t4 = table4(args.outdir)
    print(t3)
    print()
    print(t4)
    with open(os.path.join(args.outdir, "tables_3_4.txt"), "w") as f:
        f.write(t3 + "\n\n" + t4 + "\n")
    print("\nwrote", os.path.join(args.outdir, "tables_3_4.txt"))


if __name__ == "__main__":
    main()
