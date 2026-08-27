# Comparison against "Virtues and Vices of Equivariant Transformers" (arXiv:2608.02735)

Favaro, Plehn, Qu, Spinner (Aug 2026). Benchmarks Lorentz-equivariant transformers
(L-GATr, L-GATr-slim, LLoCa-Transformer) against a baseline transformer and ParT
on ATLAS top tagging, JetClass, and ATLAS flavor tagging (JetSet), including a
pretrain-on-JetClass-then-fine-tune-on-top-tagging study (their Section 5, Table 4).

**Their Table 4 uses the same classic top-tagging benchmark this project's
`--dataset top` uses** (Butter/Kasieczka et al. 2019, arXiv:1902.09914 — 1.2M
train / 400k val / 400k test; their reported dataset size and our own repeated
"404,000 test events" figure match exactly), so the comparison below is
apples-to-apples on the eval set, not just architecture-vs-architecture in the
abstract.

## Their baseline table already contains our own numbers

Their Table 4 lists "OmniLearned-M" and "OmniLearned-L" (citing Bhimji, Harris,
Mikuni, Nachman — Mikuni is the upstream author of this codebase) as the
large-pretraining reference point:

| Model (theirs) | Acc | AUC | 1/FPR@50% | 1/FPR@30% | Params | Pretrain jets |
|---|---|---|---|---|---|---|
| OmniLearned-M | 0.944 | 0.9880 | 656±12 | 3208±176 | 58M | 1058M |
| OmniLearned-L | 0.944 | 0.9880 | 688±9 | 3486±157 | 423M | 1058M |

Our own `fine_tune_top_l` (373.7M, the teacher behind all our top-tagging KD
work) lines up closely: **94.44% acc, AUC 0.9880, 1/FPR@50%=645, 1/FPR@30%=3365.**
Good external cross-check that our reproduction of these reference numbers is
consistent with the published literature.

## Our KD students vs. their equivariant fine-tuned models

Comparison at comparable parameter count, no pretraining on our side:

| Model | Acc | AUC | 1/FPR@50% | 1/FPR@30% | Params | Pretrain |
|---|---|---|---|---|---|---|
| Transformer (their baseline, from scratch) | 93.93% | 0.9855 | 389 | 1613 | 2.0M | none |
| L-GATr* (Lorentz-equivariant, from scratch) | 94.23% | 0.9870 | 540 | 2240 | 1.1M | none |
| L-GATr-slim* (from scratch) | 94.20% | 0.9869 | 546 | 2264 | 2.0M | none |
| **`distill_top_small_scratch_a05_T4` (ours, mixed KD)** | 94.32% | 0.9875 | 580 | 2804 | 2.71M | none |
| **`distill_top_small_scratch_a00_b10_T4` (ours, pure KD)** | **94.43%** | **0.9879** | **621** | **3205** | 2.71M | none |
| L-GATr-slim-f.t. (theirs, pretrained+fine-tuned) | 94.42% | 0.9879 | 655±5 | 2927±70 | 2.0M | 100M JetClass jets |
| L-GATr-slim-f.t. 48M (theirs, headline result) | 94.46% | 0.9880 | 693±17 | 3062±84 | 48M | 100M JetClass jets |

**Our simple direct-task pure-KD student (2.71M params, zero pretraining) beats
every from-scratch architecture in their comparison — including their
Lorentz-equivariant ones — on every metric.** Against their expensive
pretrained+fine-tuned category, accuracy/AUC are essentially tied, we lose on
1/FPR@50% (621 vs 655/693), but **we win on 1/FPR@30% against both, including
their 48M-param 100M-jet-pretrained headline result** (3205 vs 2927 and vs
3062) — using a 17x-smaller model with no pretraining corpus at all.

Recipe for the winning checkpoint: `distill_top_small_scratch_a00_b10_T4` —
PET2-small (2.71M params), trained from random init, `distill-alpha=0.0`
`distill-beta=1.0` `distill-t=4` (pure KD, zero weight on hard-label CE),
distilled directly from `fine_tune_top_l` on the top-tagging task itself
(no upstream/pretrain-level distillation stage). This is the winning
configuration from this project's own earlier α/β/T sweep
([[distill-lazy-teacher-progress]] memory).

## Why we win at 30% signal efficiency but lose at 50%

Verified directly (not just from summary-metric memory) via
`roc_shape_comparison.py`, which recomputes ROC curves from the raw
`outputs_*_top_test_rank*.npz` prediction files for `fine_tune_top_l`
(teacher), `fine_tune_top_s` (CE-only baseline), and both KD students.
Plot saved to
`/pscratch/sd/t/twamorka/omnilearned/results/roc_shape_comparison.pdf`
(also `.png`): left panel is the raw rejection-vs-efficiency curves, right
panel normalizes each curve to its own 50%-efficiency value to isolate
shape from overall discrimination quality.

**Sanity check** (recomputed vs. previously recorded values, exact match):

| Model | AUC | 1/FPR@50% | 1/FPR@30% |
|---|---|---|---|
| `fine_tune_top_l` (teacher) | 0.9880 | 645.1 | 3365.2 |
| `fine_tune_top_s` (CE-only) | 0.9875 | 576.9 | 2555.9 |
| `distill_top_small_scratch_a00_b10_T4` (pure KD) | 0.9879 | 621.3 | 3205.0 |
| `distill_top_small_scratch_a05_T4` (mixed KD) | 0.9875 | 580.2 | 2804.4 |

**Rejection-curve shape, normalized to the value at 50% signal efficiency**
(1/FPR@eff ÷ 1/FPR@50%) — this isolates *shape* (how fast rejection grows as
the cut tightens) from *overall* discrimination quality (AUC):

| eff | teacher | CE-only | pure KD | mixed KD |
|---|---|---|---|---|
| 0.5 | 1.000 | 1.000 | 1.000 | 1.000 |
| 0.4 | 1.956 | 2.011 | 2.083 | 1.977 |
| **0.3** | **5.217** | **4.430** | **5.159** | **4.833** |
| 0.2 | 12.038 | 12.069 | 12.037 | 15.130 |

At the 30%-efficiency working point specifically, the **teacher's own
rejection curve is unusually steep** (ratio 5.22 vs. the CE-only baseline's
4.43 — the teacher's tail is ~18% steeper than a typical same-architecture
CE-trained model). **Pure KD transfers this almost intact to the student**
(5.16, within 1% of the teacher), while **mixed KD only partially transfers
it** (4.83, roughly halfway between CE-only and the teacher) — consistent
with mixed KD blending in a hard-label CE term that pulls the shape back
toward the CE-only curve.

Mechanistic explanation: the KD loss (KL-divergence against the teacher's
full softmax distribution, not just its argmax) keeps providing gradient
signal deep into the "already correctly classified" regime, since it's
trying to match the teacher's exact confidence value rather than just get
the right answer. Ordinary CE training saturates once an example is
confidently correct, so it has comparatively less pressure to sharpen the
extreme-confidence tail that 1/FPR@30% is sensitive to. The paper's
equivariant fine-tuned models were trained with ordinary CE against ground
truth (no distillation) — their architectural prior (Lorentz-equivariance)
gives them a real, independent edge in the *bulk*/mid-range separation
(captured by 1/FPR@50%, where they beat us: 655-693 vs our 621), but they
never had a teacher's tail-shape to inherit the way our KD student did.

**Caveat, not yet resolved**: the shape-inheritance story is clean at
eff=0.3-0.4 but **breaks down at eff=0.2**, where mixed-KD (15.13) actually
exceeds both the teacher and pure-KD student (~12.0 each) — so "pure KD
inherits the teacher's steep tail" is not a universal law across all tight
cuts, just a real, verified effect specifically around the 30%-efficiency
point this comparison (and the paper's Table 4) happens to report. Whether
this is genuine curve-shape structure or partly sample-size noise (at
eff=0.2, ~16,800 of ~202,000 background events survive the cut — not a tiny
sample, but the four curves are close enough that ranking can flip) is an
open question; would need bootstrapped uncertainty bands on the ROC curves
themselves to settle definitively.

## What we can't compare yet

Our FPGA-target DeepSets students (298,887 down to 10,981 params) fall below
any size the paper's Table 4 reports — their sweep goes down to ~42K params
(their Figure 3) but only on their own newer ATLASTop dataset (full ATLAS
sim, different from the classic benchmark), not the classic dataset Table 4
uses, so there's no apples-to-apples number at our scale from this paper.

Neither this paper nor its cited quantization companion
(arXiv:2512.17011, "Economical Jet Taggers – Equivariant, Slim, and
Quantized") reports actual FPGA synthesis numbers (LUTs/DSPs/BRAM/latency)
— both report CPU/GPU cost metrics (FLOPs, inference time, memory, energy),
not hls4ml output. Our planned hls4ml step for the DeepSets FPGA work would
be filling a gap this line of research hasn't published concrete silicon
numbers for yet.

## Flavor tagging: architecture choice validated

Their JetSet (ATLAS flavor-tagging) result shows all architectures —
equivariant or not — perform nearly identically, because flavor tagging is
dominated by scalar track-impact-parameter features, not 4-momentum/Lorentz
structure. This supports using a standard PET2 architecture (not an
equivariant one) for `fine_tune_atlas_flav_m` (in progress) — there isn't
much equivariance gain to leave on the table for this specific task.
