# Improving Pipeline A (distill → fine-tune): literature + prioritized experiments

Compiled 2026-08-31. Scope: improving the **distillation stage** that produces
the student before its downstream fine-tune — covers both the FPGA DeepSets-KD
study (`fpga-deepsets-distillation-progress`) and the pretrain-scale KD ablation
(`distill-lazy-teacher-progress`). Pipeline definitions: **A = distill →
fine-tune**, **B = fine-tune → distill** (see `pipeline-ab-definitions`).

## What is already closed on this project (do not re-run)

| Tried | Result | Source |
|---|---|---|
| Pure KD (α=0, β=1) for the DeepSets student | Worse than mixed CE+KD (92.95% vs 93.30%) | fpga memory 2026-08-21 |
| Low weight decay (wd=0.05 vs 0.5) | Worse (92.83%) | fpga memory 2026-08-21 |
| Energy-weighted sum pooling | Model collapsed (50% acc) — raw-pT activation blow-up; retry only with normalized pT weights | fpga memory 2026-08-22 |
| Large (373.7M) vs small (2.71M) teacher | No meaningful difference (93.30 vs 93.24%) | fpga memory 2026-08-24 |
| CE-only vs KD (3-seed spread) | KD real edge: +0.32 acc pts (~1.7× std), clear rejection-power edge | fpga memory 2026-08-28 |

**Key implication:** teacher quality/size is *not* the bottleneck, so
teacher-assistant / TAKD-style methods are low value here. The bottleneck is how
much of the teacher's function a ~0.01–0.3M-param permutation-invariant student
can absorb through soft logits alone. That argues the remaining gains are in
(a) a less rigid KD loss, (b) feature/relational transfer, or (c) more/augmented
distillation data — not more logit-weight tuning.

## Closest prior work

**[Efficient and Robust Jet Tagging at the LHC with KD (arXiv 2311.14160)](https://arxiv.org/abs/2311.14160)**
— almost this exact setup: LorentzNet teacher → Deep Set / MLP student, loss
`(1−λ)·CE + λ·KL(T)`, `T∈{1,3,5}`.
- Deep Set student gained only **+0.2% acc / +0.001 AUC but ~25% better
  Rej₅₀%** — nearly identical to our `a05_T4` vs CE-only result.
- KD acts as a **regularizer** (val loss stops diverging — we see this:
  distillnet doesn't overfit while "small" drifts after epoch ~25).
- Pure KD (λ=1) used only to isolate **inductive-bias transfer under
  augmentation** (Lorentz boosts) — transferred boost-robustness to the student.

## Technique families (filtered to the large-capacity-gap, tiny-student regime)

### 1. Loss-function upgrades — cheap, drop-in, target rigid-KL failure mode

- **Logit Standardization in KD** — [CVPR 2024 highlight](https://github.com/sunshangquan/logit-standardization-KD).
  Z-score teacher & student logits before softmax/KL; per-sample temperature =
  weighted logit std. Lets the student match the *shape* of the teacher
  distribution without matching its range/variance. Explicitly motivated by
  teacher/student capacity discrepancy. Most plug-and-play option.
- **Decoupled KD (DKD)** — [CVPR 2022](https://ieeexplore.ieee.org/document/9879819/).
  Split KL into target-class (TCKD) + non-target-class (NCKD) terms, upweight
  NCKD (where the dark knowledge lives; vanilla KD couples & suppresses it).
  One loss reformulation, consistent +1–2% in vision.
- **DIST** — [correlation matching / stronger teacher](https://arxiv.org/html/2410.06561).
  Replace exact KL matching with Pearson-correlation loss on logits. Built for
  "teacher–student discrepancy is large."
- **NormKD** — [arXiv 2308.00520](https://arxiv.org/pdf/2308.00520). Per-sample
  temperature from logit std; removes most of the T-tuning burden.

### 2. Feature / relational KD — the original untried "item 4"

- **CRD — Contrastive Representation Distillation.** The arch-mismatch-robust
  choice. Surveys note FitNets / Attention-Transfer degrade below a vanilla
  student when architectures differ a lot; CRD holds up (~50% rel. improvement
  over AT). Relevant for a PET2-transformer → DeepSets gap. Cost: projector +
  negative sampling / memory bank.
- **Cheap variant:** MSE hint from the teacher's pooled event embedding to the
  student's pooled ρ-input via a learned linear projector. Repo already has
  `--distill-cls` (CLS-MSE) for PET2→PET2; DeepSets has no CLS token, so add a
  pooled-embedding projector. `--distill-gamma` weighting already exists.

### 3. Data / augmentation — biggest lever for a data-hungry low-bias student

- **More distillation data via teacher soft labels.** Distill on a larger, more
  diverse jet sample (pretrain mixture or JetClass) with teacher soft labels,
  then fine-tune on top-tagging. Pipeline A done more aggressively.
- **Augmentation-consistency KD.** Augment each jet (φ-rotation, constituent
  dropout, soft-drop, mild boost); force student≈teacher on both views. This is
  the mechanism 2311.14160 used to transfer robustness to the Deep Set student.
- **Mixup / CutMix on constituent sets** + teacher labels on the mix —
  [MixSKD / SuperMix](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840527.pdf)
  "notably enhance" distillation.

### 4. Joint quantization + distillation — for the FPGA endgame

- **QKD / quantized distillation** — [QKD arXiv 1911.12491](https://arxiv.org/abs/1911.12491),
  Polino et al. 2018. Do KD and quantization *together* rather than
  distill-fp32 → then-QAT. Distill straight into the Brevitas `QuantLinear`
  student (teacher stays full precision). Most valuable at the untried 6-bit /
  4-bit targets, where PTQ was 79% / random.

### 5. Teacher-side — mostly deprioritized here

- Teacher assistant / TAKD: **skip** — our ablation showed teacher capacity
  doesn't limit the student.
- **Multi-teacher soft-label averaging** (PET2 + LorentzNet + ParticleNet, if
  checkpoints exist): cheap denoising of the target, low effort if checkpoints
  are already on disk.
- **"Adapt the teacher for distillation"** —
  [ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Qian_A_Good_Teacher_Adapts_Their_Knowledge_for_Distillation_ICCV_2025_paper.pdf):
  briefly fine-tune the teacher with a distillation-aware objective so its
  outputs are more student-learnable. More involved.

## Prioritized experiment table

| # | Try | Where in code | Effort | Rationale |
|---|---|---|---|---|
| 1 | **Logit standardization** in KD loss | `get_distill_loss` in `train.py` | ~10 lines | Directly targets the capacity gap; CVPR-24 plug-and-play |
| 2 | **DKD** (TCKD/NCKD split, upweight NCKD) | `get_distill_loss` | small | Free reformulation, robust gains |
| 3 | **Augmentation-consistency KD** (φ-rotation + constituent dropout; student matches teacher on aug view) | dataloader collate + train step | medium | Published precedent for a DeepSet jet student (2311.14160); adds robustness |
| 4 | **DIST** correlation loss (swap/add to KL) | `get_distill_loss` | small | Built for teacher ≫ student |
| 5 | **Pooled-embedding hint** (teacher event embed → student ρ-input, MSE, weight `distill_gamma`) | new projector on `DeepSets`; reuse `--distill-gamma` | medium | The real "item 4"; last structural lever |
| 6 | **CRD** if #5 underdelivers | new module | higher | Arch-mismatch-robust feature KD |
| 7 | **Distill on JetClass / pretrain mix → fine-tune top** | new config + teacher companions | medium | Pipeline A "more data" for the low-bias student |
| 8 | **Joint QAT+KD** (distill into Brevitas-quantized student) | `tools/quantize/qat_deepsets.py` + KD path | medium | For 6/4-bit FPGA targets |
| 9 | **Multi-teacher soft-label averaging** | teacher companion build | low if ckpts exist | Cheap target denoising |

**Recommended sequencing:** 1 → 2 → 4 (all in the same one function, ~a day;
might recover a meaningful fraction of the CE-only ↔ KD ↔ teacher gap), then 3
(robustness + accuracy, published precedent), then 5/6 as the structural push,
with 8 reserved for once the FPGA bit-width scan is underway. Skip
teacher-assistant/TAKD.

## Sources

- [Efficient and Robust Jet Tagging at the LHC with KD (arXiv 2311.14160)](https://arxiv.org/abs/2311.14160)
- [Logit Standardization in KD (CVPR 2024)](https://github.com/sunshangquan/logit-standardization-KD)
- [Decoupled Knowledge Distillation (CVPR 2022)](https://ieeexplore.ieee.org/document/9879819/)
- [Efficient and Robust KD from a Stronger Teacher / correlation matching](https://arxiv.org/html/2410.06561)
- [NormKD: Normalized Logits for KD](https://arxiv.org/pdf/2308.00520)
- [Contrastive Representation Distillation — review](https://liner.com/review/contrastive-representation-distillation)
- [A Comprehensive Survey on Knowledge Distillation](https://openreview.net/pdf?id=3cbJzdR78B)
- [Improved KD via Teacher Assistant (AAAI 2020)](https://ojs.aaai.org/index.php/AAAI/article/view/5963/5819)
- [MixSKD: Self-KD from Mixup (ECCV 2022)](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136840527.pdf)
- [QKD: Quantization-aware Knowledge Distillation (arXiv 1911.12491)](https://arxiv.org/abs/1911.12491)
- [A Good Teacher Adapts Their Knowledge for Distillation (ICCV 2025)](https://openaccess.thecvf.com/content/ICCV2025/papers/Qian_A_Good_Teacher_Adapts_Their_Knowledge_for_Distillation_ICCV_2025_paper.pdf)
