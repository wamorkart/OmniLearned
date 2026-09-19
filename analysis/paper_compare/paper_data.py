"""Single source of truth for the comparison against

    Petitjean, Plehn, Spinner, Kothe,
    "Economical Jet Taggers -- Equivariant, Slim, and Quantized",
    arXiv:2512.17011v2 (SciPost submission, Jan 2026).

All "their" numbers are transcribed from the paper PDF (Tables 1, 5, 6 and
the text of Sec 2.5 / 3.3 / 3.4). All "ours" numbers are pulled from this
project's own eval logs under
/pscratch/sd/t/twamorka/omnilearned/results/ and the FPGA-DeepSets /
distill-lazy-teacher memory notes -- see the per-entry `src` field.

Shared benchmark: the Kasieczka et al. top-tagging reference set
(1.2M train, ~400k test), binary QCD-vs-top, metrics = accuracy, AUC,
and background rejection 1/eps_B at fixed signal efficiency eps_S.
Both efforts report eps_S in {0.5, 0.3}. Directly comparable.
"""

# --------------------------------------------------------------------------
# THEIR Table 1 -- top tagging, accuracy / AUC / 1/eps_B(0.5) / 1/eps_B(0.3)
#   params in millions, n_train in millions (1.2 = from scratch, 100/1058 =
#   pretrained then fine-tuned). asterisk in the paper = Lorentz-equivariant.
# --------------------------------------------------------------------------
THEIR_TABLE1 = [
    # name,                 params_M, acc,     auc,    rej50, rej30, n_train_M, equivariant, pretrained
    ("ParticleNet",            0.4,  0.940,  0.9858,  397,  1615,   1.2,  False, False),
    ("Transformer",            2.0,  0.9393, 0.9855,  389,  1613,   1.2,  False, False),
    ("ParT",                   2.1,  0.940,  0.9858,  413,  1602,   1.2,  False, False),
    ("ParT (longer)",          2.1,  0.9416, 0.9865,  485,  1808,   1.2,  False, False),
    ("MIParT",                 2.2,  0.942,  0.9868,  505,  2010,   1.2,  False, False),
    ("IAFormer",               0.2,  0.942,  0.987,   510,  2012,   1.2,  False, False),
    ("PET v2-s",               3.0,  0.943,  0.987,   505,  2167,   1.2,  False, False),
    ("LorentzNet",             0.2,  0.942,  0.9868,  498,  2195,   1.2,  True,  False),
    ("PELICAN",                0.2,  0.9426, 0.9870,  None, 2250,   1.2,  True,  False),
    ("CGENN",                  0.3,  0.942,  0.9869,  500,  2172,   1.2,  True,  False),
    ("LLoCa-Transformer",      2.0,  0.9416, 0.9866,  492,  2150,   1.2,  True,  False),
    ("L-GATr",                 1.1,  0.9423, 0.9870,  540,  2240,   1.2,  True,  False),
    ("L-GATr-slim",            2.0,  0.9420, 0.9869,  546,  2264,   1.2,  True,  False),
    ("ParticleNet-f.t.",       0.4,  0.942,  0.9866,  487,  1771,   100,  False, True),
    ("OmniLearn",              2.0,  0.942,  0.9872,  568,  2647,   100,  False, True),
    ("ParT-f.t.",              2.1,  0.944,  0.9877,  691,  2766,   100,  False, True),
    ("MIParT-f.t.",            2.3,  0.944,  0.9878,  640,  2789,   100,  False, True),
    ("L-GATr-f.t.",            1.1,  0.9446, 0.9879,  651,  2894,   100,  True,  True),
    ("L-GATr-slim-f.t.",       2.0,  0.9442, 0.9879,  655,  2927,   100,  True,  True),
    ("OmniLearned-M",         58.0,  0.944,  0.9880,  656,  3208,  1058,  False, True),
    ("OmniLearned-L",        423.0,  0.944,  0.9880,  688,  3486,  1058,  False, True),
]

# THEIR Sec 2.5 / Fig 3 ultra-mini regime (read off the figure -- approximate,
# 1/eps_B @ eps_S=0.3). Two down-scaling axes; "fixed 10 blocks, reduce width"
# is the one that keeps a Lorentz-equivariant net above ~1000 at ~1k params.
THEIR_ULTRAMINI_FIG3 = {
    # params : {arch : rej30 (approx, from figure)}
    200_000: {"LLoCa-Transformer": 2000, "L-GATr-slim": 1900, "Transformer": 1750},
    20_000:  {"LLoCa-Transformer": 1500, "L-GATr-slim": 1450, "Transformer": 1150},
    2_000:   {"LLoCa-Transformer": 1150, "L-GATr-slim": 1050, "Transformer": 600},
}
THEIR_ULTRAMINI_NOTE = (
    "approx values read off Fig. 3 (right panel: 10 blocks, reduced width); "
    "paper text: the minimal Lorentz-equivariant tagger 'loses a factor of two "
    "in background rejection' vs full size but still beats all pre-graph taggers."
)

# --------------------------------------------------------------------------
# THEIR Table 5 -- energy per operation, 7nm process (NVIDIA A100 node), pJ.
# --------------------------------------------------------------------------
THEIR_ENERGY_PJ = {
    #  dtype   : (E_add_pJ, E_mul_pJ)
    "float32":  (0.38,  1.31),
    "bfloat16": (0.11,  0.21),
    "int8":     (0.007, 0.07),
}

# --------------------------------------------------------------------------
# THEIR Table 6 -- quantization impact, top tagging, 1/eps_B @ eps_S=0.3.
#   f32        : full precision
#   i8         : int8 inputs to inner linear layers, weights still float
#   i8+STE     : + ternary weights via straight-through estimation
#   i8+PARQ    : + ternary weights via piecewise-affine regularized quant
# --------------------------------------------------------------------------
THEIR_TABLE6 = {
    # arch               : (f32,        i8,         i8_STE,     i8_PARQ)   -- (value, err)
    "Transformer":         ((1694, 69), (1577, 68), (1357, 69), (1302, 55)),
    "ParT":                ((1808, 33), (1768, 81), (1676, 94), (1693, 86)),
    "LLoCa-Transformer":   ((2150, 130),(2019, 79), (1932, 132),(1935, 110)),
    "L-GATr-slim":         ((2264, 93), (2082, 124),(1990, 108),(1872, 67)),
}

# --------------------------------------------------------------------------
# OURS -- top tagging, 404k test events. rej50/rej30 = 1/FPR at 50%/30% sig eff.
#   arch: "pet2" (OmniLearned/PET2 transformer) or "deepsets".
#   n_train_M kept parallel to THEIR_TABLE1: our PET2 students inherit the
#   1058M-jet pretraining of the OmniLearned-L teacher via KD, so they belong
#   in the "pretrained" bucket even when the student itself is random-init.
# --------------------------------------------------------------------------
OUR_MODELS = [
    # key,                         arch,       params,    acc,     auc,    rej50, rej30, rej30_err, kind,             src
    ("teacher_L",                  "pet2",   373_687_000, 0.9444, 0.9880, 645,   3365,  None,  "teacher (OmniLearned-L)",
        "results/top_distill_sweep_eval.log reference line"),
    ("pet2_small_ce",              "pet2",     2_709_000, 0.9438, 0.9875, 577,   2556,  None,  "CE-only (pretrain->ft)",
        "results/deepsets_mlp_comparison.txt"),
    ("pet2_small_kd_pureKD",       "pet2",     2_709_000, 0.9443, 0.9879, 630,   3297,  130,   "KD from L, pure (a=0,b=1,T=4), n=3",
        "results/top_T_sweep_eval.log final table"),
    ("pet2_small_kd_a05",          "pet2",     2_709_000, 0.9431, 0.9875, 596,   2953,  174,   "KD from L, mixed (a=0.5), n=5",
        "results/top_T_sweep_eval.log / top_distill_sweep_eval.log"),
    ("pet2_micro_kd_a05",          "pet2",        None,   0.9434, 0.9876, 605,   3021,  251,   "KD from L, micro PET2, n=5",
        "results/deepsets_mlp_comparison.txt (micro param count filled by flops_probe)"),
    ("pet2_micro_ce",              "pet2",        None,   0.9412, 0.9863, 442,   1639,  71,    "CE-only, micro PET2, n=5",
        "results/deepsets_mlp_comparison.txt"),
    ("pet2_small_pretrainKD_ft",   "pet2",     2_709_000, 0.9417, 0.9867, 478,   2148,  None,  "pipeline-A: pretrain-KD then ft (a00_b10_v2)",
        "results/a00_b10_v2_vs_a05_b05_top_metrics.txt"),
    ("deepsets_small_kd_a05",      "deepsets",   298_887, 0.9330, 0.9828, 283,   1122,  None,  "KD from L (a=0.5), 299k",
        "results/deepsets_mlp_comparison.txt"),
    ("deepsets_small_ce",          "deepsets",   298_887, 0.9284, 0.9799, 202,    697,  8,     "CE-only, 299k, n=3",
        "fpga-deepsets memory 2026-08-28 spread"),
    ("deepsets_small_kd_teachS",   "deepsets",   298_887, 0.9316, 0.9817, 245,    895,  175,   "KD from small teacher, 299k, n=3",
        "fpga-deepsets memory 2026-08-28 spread"),
    ("deepsets_distillnet_kd",     "deepsets",    10_981, 0.9285, 0.9800, 199,    748,  None,  "KD from L (a=0.5), 11k (DistillNet-parity)",
        "fpga-deepsets memory 2026-08-25"),
    ("deepsets_distillnet_qat8",   "deepsets",    10_981, 0.9258, 0.9785, 178,    699,  None,  "11k, 8-bit QAT (Brevitas int8 w+a)",
        "fpga-deepsets memory 2026-08-25 QAT table"),
    ("deepsets_distillnet_ptq8",   "deepsets",    10_981, 0.8290, 0.9151, 19.5,   45.2, None,  "11k, 8-bit PTQ (percentile calib)",
        "fpga-deepsets memory 2026-08-25 PTQ table"),
]

# --------------------------------------------------------------------------
# OURS -- forward-pass FLOPs (analytic, FlopCounterMode) at the top-tagging
# median particle count N=47. From analysis/benchmark_inference.py output
# /pscratch/sd/t/twamorka/omnilearned/results/inference_benchmark.txt.
# Missing entries (None) are filled by analysis/paper_compare/flops_probe.py.
# torch FlopCounterMode reports 2*MACs, so MACs = flops / 2.
# --------------------------------------------------------------------------
OUR_FLOPS_AT_MEDIAN_N = {
    "teacher_L":               75_885_530_000,
    "pet2_small":                  550_800_000,
    "pet2_micro":                          None,
    "deepsets_small":               15_760_000,
    "deepsets_distillnet":                 None,
}

MEDIAN_N = 47
TEST_N_EVENTS = 404_000
